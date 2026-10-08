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
            application.list_remote_jobs(
                caller,
                JOB_SAMPLE_LIMIT,
                principal.allows(Capability::Enrich),
            ),
        )
        .await
        {
            // Defense in depth: adapter output must still belong to this
            // principal. Family visibility is applied before sampling and
            // aggregation, so hidden failures cannot affect readiness.
            let mut sampled = 0;
            for job in list
                .jobs
                .into_iter()
                .filter(|job| job.instance_id == principal.instance_id)
                .filter(|job| crate::uploads::job_visible(principal, job))
                .take(JOB_SAMPLE_LIMIT)
            {
                let status = match job.status.as_str() {
                    "queued" | "running" | "completed" | "failed" | "cancelled" | "interrupted" => {
                        job.status.as_str()
                    }
                    _ => "unknown",
                };
                *jobs.counts.entry(status.into()).or_default() += 1;
                sampled += 1;
            }
            jobs.sampled = sampled;
            jobs.readiness = Readiness::new(
                ReadinessState::Ready,
                "Bounded recent job sample for this principal only.",
                None,
            );
            if jobs.counts.contains_key("failed") || jobs.counts.contains_key("interrupted") {
                jobs.readiness = Readiness::new(ReadinessState::Partial, "The recent owned-job sample includes failed or interrupted work.", Some("Inspect your owned jobs, repair the reported cause, then explicitly resume the selected recoverable job."));
            } else if jobs.counts.contains_key("unknown") {
                jobs.readiness = Readiness::unknown(
                    "The recent owned-job sample includes an unrecognized state.",
                    "Inspect your owned jobs and check service/client versions with the owner.",
                );
            }
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
