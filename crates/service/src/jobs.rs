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
                        handle_completion(result, &mut running, &mut owners, &application, &signal);
                    },
                    _ = tokio::time::sleep(Duration::from_millis(100)) => {},
                }
            }
            // Shutdown wakes preparation cancellation, then drains workers.
            // Their safe checkpoints remain resumable on the next service.
            while let Some(result) = running.join_next_with_id().await {
                handle_completion(
                    Some(result),
                    &mut running,
                    &mut owners,
                    &application,
                    &signal,
                );
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

fn handle_completion(
    result: Option<WorkerCompletion>,
    running: &mut JoinSet<WorkerOutcome>,
    owners: &mut HashMap<Id, RemoteJobExecution>,
    application: &Arc<dyn RemoteApplicationOperations>,
    shutdown: &CancellationToken,
) {
    match result {
        Some(Ok((id, (execution, outcome)))) => {
            owners.remove(&id);
            if let Err(error) = outcome {
                if !shutdown.is_cancelled() {
                    schedule_recovery(
                        running,
                        owners,
                        application,
                        execution,
                        error.code().into(),
                        shutdown,
                    );
                }
            }
        }
        Some(Err(error)) => {
            if let Some(execution) = owners
                .remove(&error.id())
                .filter(|_| !shutdown.is_cancelled())
            {
                schedule_recovery(
                    running,
                    owners,
                    application,
                    execution,
                    "worker_interrupted".into(),
                    shutdown,
                );
            }
            tracing::warn!("Uploaded worker interrupted; its durable checkpoint was retained.");
        }
        None => {}
    }
}

fn schedule_recovery(
    running: &mut JoinSet<WorkerOutcome>,
    owners: &mut HashMap<Id, RemoteJobExecution>,
    application: &Arc<dyn RemoteApplicationOperations>,
    execution: RemoteJobExecution,
    error_code: String,
    shutdown: &CancellationToken,
) {
    let application = Arc::clone(application);
    let owner = execution.clone();
    let shutdown = shutdown.clone();
    let handle = running.spawn(async move {
        let mut delay = Duration::from_millis(100);
        loop {
            if shutdown.is_cancelled() {
                // Leave the durable running lease for startup reconciliation.
                return (execution, Ok(()));
            }
            // Finish this operation before observing shutdown; do not drop an
            // in-flight atomic write to achieve a timeout.
            match application.recover_remote_job(execution.clone(), error_code.clone()).await {
                Ok(()) => return (execution, Ok(())),
                Err(_) => {
                    tracing::warn!("Uploaded job terminalization is pending; retaining its fenced worker and retrying.");
                    tokio::select! {
                        _ = shutdown.cancelled() => return (execution, Ok(())),
                        _ = tokio::time::sleep(delay) => {},
                    }
                    delay = (delay * 2).min(Duration::from_secs(5));
                }
            }
        }
    });
    // Recovery occupies the same bounded worker slot and is drained during
    // shutdown. Never re-execute providers or abandon a still-running lease.
    owners.insert(handle.id(), owner);
}

#[cfg(test)]
mod tests {
    use super::*;
    use graphrag_agents::{
        DeterministicEmbedder, FixtureEntityExtractor, LibrarianRuntimeConfig, SearchAgent,
    };
    use graphrag_application::{
        ApplicationError, CallerIdentity, EmbeddedApplication, UploadSourceRequest,
    };
    use graphrag_db::{init_memory, Repository};

    #[tokio::test]
    async fn failed_status_reads_and_terminal_writes_keep_recovery_tracked_until_settled() {
        for read_failure in [true, false] {
            let db = init_memory().await.unwrap();
            let repo = Repository::new(db.clone());
            let embedder = Arc::new(DeterministicEmbedder::default());
            let app = Arc::new(EmbeddedApplication::new(
                repo.clone(),
                SearchAgent::new(repo.clone(), embedder.clone()),
                embedder,
                Arc::new(FixtureEntityExtractor::default()),
                LibrarianRuntimeConfig::default(),
            ));
            let admission = app
                .upload_source(
                    CallerIdentity {
                        instance_id: "owner".into(),
                    },
                    UploadSourceRequest {
                        request_id: "retry-settlement".into(),
                        document_key: "synthetic.md".into(),
                        content: "Synthetic pending upload".into(),
                        title: None,
                        provenance: None,
                        extract_entities: false,
                    },
                )
                .await
                .unwrap();
            let execution = app
                .claim_remote_job("epoch", "worker")
                .await
                .unwrap()
                .unwrap();
            assert_eq!(execution.job_id, admission.job_id);
            let job_id = execution.job_id.clone();
            if read_failure {
                // Fault the minimal cancellation state, independently of the
                // immutable input that recovery no longer needs to decode.
                db.query("DEFINE FIELD OVERWRITE remote_cancel_requested ON processing_job TYPE any; UPDATE processing_job SET remote_cancel_requested = 'invalid-bool' WHERE job_type = 'remote_upload'").await.unwrap().check().unwrap();
            } else {
                db.query("DEFINE FIELD OVERWRITE status ON processing_job TYPE string ASSERT $value != 'failed'").await.unwrap().check().unwrap();
            }
            let application: Arc<dyn RemoteApplicationOperations> = app.clone();
            let mut running = JoinSet::new();
            let finished = execution.clone();
            let worker = running.spawn(async move {
                (
                    finished,
                    Err(ApplicationError::ProviderUnavailable(
                        "synthetic provider unavailable".into(),
                    )),
                )
            });
            let mut owners = HashMap::from([(worker.id(), execution.clone())]);
            let outcome = running.join_next_with_id().await;
            handle_completion(
                outcome,
                &mut running,
                &mut owners,
                &application,
                &CancellationToken::new(),
            );
            tokio::time::sleep(Duration::from_millis(130)).await;
            assert_eq!(
                owners.len(),
                1,
                "the failed worker's fenced lease must remain tracked"
            );
            assert_eq!(
                running.len(),
                1,
                "recovery must retain the bounded worker slot"
            );
            assert!(matches!(
                app.recover_remote_job(execution.clone(), "provider_unavailable".into())
                    .await,
                Err(ApplicationError::Internal(_))
            ));
            if read_failure {
                db.query("UPDATE processing_job SET remote_cancel_requested = false WHERE job_type = 'remote_upload'; DEFINE FIELD OVERWRITE remote_cancel_requested ON processing_job TYPE bool").await.unwrap().check().unwrap();
            } else {
                db.query("DEFINE FIELD OVERWRITE status ON processing_job TYPE string")
                    .await
                    .unwrap()
                    .check()
                    .unwrap();
            }
            let outcome = tokio::time::timeout(Duration::from_secs(5), running.join_next_with_id())
                .await
                .unwrap();
            handle_completion(
                outcome,
                &mut running,
                &mut owners,
                &application,
                &CancellationToken::new(),
            );
            assert!(owners.is_empty());
            assert!(running.is_empty());
            let terminal = repo
                .get_remote_upload_job("owner", &job_id)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(terminal.job.status, "failed");
            assert_eq!(
                terminal.job.last_error.as_deref(),
                Some("provider_unavailable")
            );
            assert!(terminal.service_epoch.is_none());
            assert!(terminal.worker_token.is_none());
            // The same process can resume immediately; no epoch restart needed.
            app.resume_remote_job(
                CallerIdentity {
                    instance_id: "owner".into(),
                },
                &job_id,
            )
            .await
            .unwrap();
            assert_eq!(
                repo.get_remote_upload_job("owner", &job_id)
                    .await
                    .unwrap()
                    .unwrap()
                    .job
                    .status,
                "queued"
            );
            assert_eq!(repo.get_stats().await.unwrap().note_count, 0);
        }
    }
    #[tokio::test]
    async fn shutdown_stops_recovery_retries_after_current_write_and_restart_reconciles() {
        let db = init_memory().await.unwrap();
        let repo = Repository::new(db.clone());
        let embedder = Arc::new(DeterministicEmbedder::default());
        let app = Arc::new(EmbeddedApplication::new(
            repo.clone(),
            SearchAgent::new(repo.clone(), embedder.clone()),
            embedder,
            Arc::new(FixtureEntityExtractor::default()),
            LibrarianRuntimeConfig {
                min_chunk_size: 1,
                skip_entity_extraction: true,
                ..Default::default()
            },
        ));
        let caller = CallerIdentity {
            instance_id: "shutdown-owner".into(),
        };
        let admission = app
            .upload_source(
                caller.clone(),
                UploadSourceRequest {
                    request_id: "shutdown-storage-fault".into(),
                    document_key: "shutdown-document".into(),
                    content:
                        "# Synthetic storage fault\n\nCommitted chunks survive graceful shutdown."
                            .into(),
                    title: None,
                    provenance: None,
                    extract_entities: false,
                },
            )
            .await
            .unwrap();
        db.query("DEFINE FIELD OVERWRITE status ON processing_job TYPE string ASSERT $value IN ['queued', 'running']").await.unwrap().check().unwrap();
        let shutdown = CancellationToken::new();
        let application: Arc<dyn RemoteApplicationOperations> = app.clone();
        let workers = JobWorkers::start(application.clone(), 1, &shutdown)
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                if repo
                    .get_remote_upload_job(&caller.instance_id, &admission.job_id)
                    .await
                    .unwrap()
                    .unwrap()
                    .phase
                    == "promoted"
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        // Leave the fault unrepaired: shutdown must finish the current attempt
        // and stop retrying, preserving the running lease for reconciliation.
        tokio::time::timeout(Duration::from_secs(2), workers.stop())
            .await
            .unwrap();
        let pending = repo
            .get_remote_upload_job(&caller.instance_id, &admission.job_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(pending.job.status, "running");
        let ids = pending.job.item_ids.clone();
        assert!(!ids.is_empty());
        db.query("DEFINE FIELD OVERWRITE status ON processing_job TYPE string")
            .await
            .unwrap()
            .check()
            .unwrap();
        let restart = JobWorkers::start(application, 1, &CancellationToken::new())
            .await
            .unwrap();
        let interrupted = repo
            .get_remote_upload_job(&caller.instance_id, &admission.job_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(interrupted.job.status, "failed");
        assert_eq!(interrupted.job.last_error.as_deref(), Some("interrupted"));
        app.resume_remote_job(caller.clone(), &admission.job_id)
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                let job = repo
                    .get_remote_upload_job(&caller.instance_id, &admission.job_id)
                    .await
                    .unwrap()
                    .unwrap();
                if job.job.status == "completed" {
                    assert_eq!(job.job.item_ids, ids);
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        restart.stop().await;
    }
}
