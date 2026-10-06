//! Readiness does not confer upload/job visibility and never invokes inference.
use crate::{Capability, Principal};
use graphrag_application::{
    CallerIdentity, OwnedJobReadiness, Readiness, ReadinessState, RemoteApplicationOperations,
    ServiceReadiness,
};
use std::{collections::BTreeMap, time::Duration};

const JOB_SAMPLE_LIMIT: usize = 10;

pub(crate) async fn report(
    application: &dyn RemoteApplicationOperations,
    principal: &Principal,
) -> ServiceReadiness {
    let application_report =
        match tokio::time::timeout(Duration::from_secs(4), application.service_readiness()).await {
            Ok(Ok(report)) => report,
            _ => Default::default(),
        };
    let mut jobs = OwnedJobReadiness {
        readiness: Readiness::new(
            ReadinessState::Forbidden,
            "Job visibility is not granted to this principal.",
            Some("Ask the owner for jobs permission if needed."),
        ),
        sampled: 0,
        limit: JOB_SAMPLE_LIMIT,
        counts: BTreeMap::new(),
    };
    if principal.allows(Capability::Jobs) {
        jobs.readiness = Readiness::unknown(
            "Owned job evidence is unavailable.",
            "Retry owned jobs status or ask the service owner to inspect it.",
        );
        let caller = CallerIdentity {
            instance_id: principal.instance_id.clone(),
        };
        if let Ok(Ok(list)) = tokio::time::timeout(
            Duration::from_secs(1),
            application.list_remote_jobs(caller, JOB_SAMPLE_LIMIT),
        )
        .await
        {
            // Defense in depth: adapter output must still belong to this principal.
            let owned: Vec<_> = list
                .jobs
                .into_iter()
                .filter(|job| job.instance_id == principal.instance_id)
                .take(JOB_SAMPLE_LIMIT)
                .collect();
            for job in &owned {
                let status = match job.status.as_str() {
                    "queued" | "running" | "completed" | "failed" | "cancelled" | "interrupted" => {
                        job.status.as_str()
                    }
                    _ => "unknown",
                };
                *jobs.counts.entry(status.into()).or_default() += 1;
            }
            jobs.sampled = owned.len();
            jobs.readiness = Readiness::new(
                ReadinessState::Ready,
                "Bounded recent job sample for this principal only.",
                None,
            );
        }
    }
    ServiceReadiness {
        schema_version: 1,
        read_only: true,
        inference_probed: false,
        instance_id: principal.instance_id.clone(),
        application: application_report,
        jobs,
    }
}
