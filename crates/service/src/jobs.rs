//! Server-owned workers. HTTP sessions never own or cancel these tasks.
use graphrag_application::{ActionCancellation, RemoteApplicationOperations, RemoteJobExecution};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::task::{Id, JoinHandle, JoinSet};
use tokio_util::sync::CancellationToken;

pub(crate) struct JobWorkers {
    stop: CancellationToken,
    task: JoinHandle<()>,
}

impl JobWorkers {
    pub(crate) async fn start(
        application: Arc<dyn RemoteApplicationOperations>,
        maximum: usize,
        shutdown: &CancellationToken,
    ) -> Result<Self, crate::ServiceError> {
        let epoch = format!(
            "service-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|_| crate::ServiceError::Jobs)?
                .as_nanos()
        );
        application
            .reconcile_remote_jobs(&epoch)
            .await
            .map_err(|_| crate::ServiceError::Jobs)?;
        let stop = shutdown.child_token();
        let signal = stop.clone();
        let task = tokio::spawn(async move {
            let mut running = JoinSet::new();
            let mut owners = HashMap::<Id, RemoteJobExecution>::new();
            let mut sequence = 0_u64;
            loop {
                if signal.is_cancelled() {
                    break;
                }
                while running.len() < maximum && !signal.is_cancelled() {
                    sequence += 1;
                    let token = format!("{epoch}-{sequence}");
                    let execution = match application.claim_remote_job(&epoch, &token).await {
                        Ok(Some(execution)) => execution,
                        Ok(None) => break,
                        Err(_) => {
                            tracing::warn!("Cannot claim an uploaded job; retrying shortly.");
                            break;
                        }
                    };
                    let owner = execution.clone();
                    let application = Arc::clone(&application);
                    let stop = signal.clone();
                    let handle = running.spawn(async move {
                        let cancel = ActionCancellation::new();
                        let monitor_cancel = cancel.clone();
                        let monitor_application = Arc::clone(&application);
                        let monitored = execution.clone();
                        let monitor = tokio::spawn(async move {
                            loop {
                                tokio::select! {
                                    _ = stop.cancelled() => { monitor_cancel.cancel(); return; }
                                    _ = tokio::time::sleep(Duration::from_millis(100)) => {
                                        match monitor_application.remote_job_cancel_requested(&monitored).await {
                                            Ok(false) => {},
                                            _ => { monitor_cancel.cancel(); return; }
                                        }
                                    }
                                }
                            }
                        });
                        // Never select/drop this future on disconnect, cancel,
                        // or shutdown: each atomic write finishes before exit.
                        let result = application.execute_remote_job(execution.clone(), cancel).await;
                        monitor.abort();
                        let _ = monitor.await;
                        (execution, result)
                    });
                    owners.insert(handle.id(), owner);
                }
                tokio::select! {
                    _ = signal.cancelled() => break,
                    result = running.join_next_with_id(), if !running.is_empty() => {
                        handle_completion(result, &mut owners, &application).await;
                    },
                    _ = tokio::time::sleep(Duration::from_millis(100)) => {},
                }
            }
            // Shutdown wakes preparation cancellation, then drains workers.
            // Their safe checkpoints remain resumable on the next service.
            while let Some(result) = running.join_next_with_id().await {
                handle_completion(Some(result), &mut owners, &application).await;
            }
        });
        Ok(Self { stop, task })
    }

    pub(crate) async fn stop(self) {
        self.stop.cancel();
        let _ = self.task.await;
    }
}

type WorkerOutcome = (
    RemoteJobExecution,
    graphrag_application::ApplicationResult<()>,
);
type WorkerCompletion = Result<(Id, WorkerOutcome), tokio::task::JoinError>;

async fn handle_completion(
    result: Option<WorkerCompletion>,
    owners: &mut HashMap<Id, RemoteJobExecution>,
    application: &Arc<dyn RemoteApplicationOperations>,
) {
    match result {
        Some(Ok((id, (_, outcome)))) => {
            owners.remove(&id);
            if outcome.is_err() {
                tracing::warn!("Uploaded job stopped; inspect its durable status before resuming.");
            }
        }
        Some(Err(error)) => {
            if let Some(execution) = owners.remove(&error.id()) {
                let _ = application.interrupt_remote_job(execution).await;
            }
            tracing::warn!("Uploaded worker interrupted; its durable checkpoint was retained.");
        }
        None => {}
    }
}
