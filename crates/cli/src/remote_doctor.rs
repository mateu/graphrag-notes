//! Provider-free remote diagnostics. Raw stateless MCP keeps HTTP auth errors distinct.
use crate::{
    cli::{Cli, DoctorFormat},
    output,
};
use anyhow::Result;
use graphrag_application::{
    ApplicationReadiness, OwnedJobReadiness, Readiness, ReadinessState, ServiceReadiness,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{io::Read, path::Path, time::Duration};

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    read_only: bool,
    inference_probed: bool,
    exit_code: i32,
    transport: Readiness,
    authorization: Readiness,
    application: ApplicationReadiness,
    jobs: OwnedJobReadiness,
    service: Option<ServiceReadiness>,
    client_refresh_evidence: Option<RefreshEvidence>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RefreshCounts {
    created: u64,
    changed: u64,
    unchanged: u64,
    failed: u64,
    missing: u64,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RefreshRetry {
    action: Option<String>,
    #[serde(default)]
    plan_sha256: Option<String>,
}
/// This is client-local evidence; a successful refresh does not establish current server freshness.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RefreshEvidence {
    schema_version: u32,
    endpoint_sha256: String,
    instance_id: String,
    collection_id: String,
    status: String,
    last_attempt_at: Option<String>,
    last_success_at: Option<String>,
    counts: RefreshCounts,
    pending_parts: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    retained_extraction_policy_parts: Option<u64>,
    error_code: Option<String>,
    retry: RefreshRetry,
}

fn load_refresh(path: &Path, server: &str, instance: &str) -> Option<RefreshEvidence> {
    let before = std::fs::symlink_metadata(path).ok()?;
    if !before.is_file() || before.file_type().is_symlink() || before.len() > 16_384 {
        return None;
    }
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options.open(path).ok()?;
    let metadata = file.metadata().ok()?;
    if !metadata.is_file() || metadata.len() > 16_384 {
        return None;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        // SAFETY: geteuid has no arguments or memory access.
        if metadata.mode() & 0o777 != 0o600
            || metadata.uid() != unsafe { libc::geteuid() }
            || metadata.dev() != before.dev()
            || metadata.ino() != before.ino()
        {
            return None;
        }
    }
    let mut bytes = Vec::new();
    file.take(16_385).read_to_end(&mut bytes).ok()?;
    if bytes.len() > 16_384 {
        return None;
    }
    let evidence: RefreshEvidence = serde_json::from_slice(&bytes).ok()?;
    let safe_label = |value: &str| {
        !value.is_empty()
            && value.len() <= 128
            && value
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
    };
    if evidence.schema_version != 1
        || evidence.instance_id != instance
        || evidence.endpoint_sha256 != format!("{:x}", Sha256::digest(server.as_bytes()))
        || (!safe_label(&evidence.collection_id) || evidence.collection_id.len() > 64)
        || !matches!(
            evidence.status.as_str(),
            "unknown" | "running" | "partial" | "failed" | "complete" | "paused"
        )
        || evidence
            .error_code
            .as_deref()
            .is_some_and(|value| !safe_label(value))
        || evidence
            .retry
            .action
            .as_deref()
            .is_some_and(|value| !matches!(value, "resume" | "refresh" | "reconcile"))
        || evidence.retry.plan_sha256.as_deref().is_some_and(|value| {
            value.len() != 64
                || !value
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        })
        || (evidence.retry.action.as_deref() == Some("reconcile")
            && evidence.retry.plan_sha256.is_none())
        || [&evidence.last_attempt_at, &evidence.last_success_at]
            .into_iter()
            .flatten()
            .any(|value| chrono::DateTime::parse_from_rfc3339(value).is_err())
    {
        return None;
    }
    Some(evidence)
}

async fn collect(cli: &Cli, server: &str) -> Report {
    let mut report = Report {
        schema_version: 1,
        read_only: true,
        inference_probed: false,
        exit_code: 2,
        transport: Readiness::new(
            ReadinessState::Unavailable,
            "The MCP endpoint could not be reached.",
            Some("Check the endpoint and encrypted tunnel or reverse proxy."),
        ),
        authorization: Readiness::unknown(
            "Authentication has not been established.",
            "Check the selected credential environment variable.",
        ),
        application: ApplicationReadiness::default(),
        jobs: OwnedJobReadiness {
            readiness: Readiness::unknown(
                "Authenticated owned-job evidence is unavailable.",
                "Establish status authorization before checking owned jobs.",
            ),
            sampled: 0,
            limit: 10,
            counts: Default::default(),
        },
        service: None,
        client_refresh_evidence: None,
    };
    let token = match std::env::var(&cli.credential_env) {
        Ok(token)
            if !token.is_empty() && !token.chars().any(|c| c.is_whitespace() || c.is_control()) =>
        {
            token
        }
        _ => {
            report.authorization = Readiness::new(ReadinessState::NotConfigured, "The selected credential is missing or invalid.", Some("Set the selected credential environment variable; never place its value in the URL."));
            report.transport.state = ReadinessState::Unknown;
            report.transport.summary =
                "No request was sent because a usable credential was absent.".into();
            return report;
        }
    };
    let Ok(http) = reqwest_mcp::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(8))
        .redirect(reqwest_mcp::redirect::Policy::none())
        .build()
    else {
        return report;
    };
    let response = http.post(server).bearer_auth(token).header("Accept", "application/json, text/event-stream").header("MCP-Protocol-Version", "2025-11-25")
        .json(&json!({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"service_status","arguments":{}}})).send().await;
    let Ok(mut response) = response else {
        return report;
    };
    report.transport = Readiness::new(ReadinessState::Ready, "The HTTP endpoint responded.", None);
    match response.status().as_u16() {
        401 | 403 => {
            report.authorization = Readiness::new(
                ReadinessState::Forbidden,
                "The endpoint denied authentication or authorization.",
                Some("Check credential selection and service permissions with the owner."),
            );
            return report;
        }
        200 => {}
        429 | 503 => {
            report.transport = Readiness::new(
                ReadinessState::Unavailable,
                "The service is busy or unavailable.",
                Some("Retry shortly; ask the owner to inspect service readiness if it persists."),
            );
            return report;
        }
        _ => {
            report.transport = Readiness::unknown(
                "The endpoint did not return a compatible status response.",
                "Check the /mcp endpoint and service version.",
            );
            return report;
        }
    }
    let mut bytes = Vec::new();
    while let Ok(Some(chunk)) = response.chunk().await {
        if chunk.len() > 65_536usize.saturating_sub(bytes.len()) {
            return report;
        }
        bytes.extend_from_slice(&chunk);
    }
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return report;
    };
    if value["jsonrpc"] != "2.0"
        || value["id"] != 1
        || value
            .pointer("/result/structuredContent/schema_version")
            .and_then(|version| version.as_u64())
            != Some(1)
    {
        report.authorization = Readiness::unknown(
            "A compatible authenticated status result was not returned.",
            "Check the MCP endpoint and update the service if status is unsupported.",
        );
        return report;
    }
    if value
        .pointer("/result/structuredContent/error/code")
        .and_then(|v| v.as_str())
        == Some("forbidden")
    {
        report.authorization = Readiness::new(
            ReadinessState::Forbidden,
            "The principal lacks read/status permission.",
            Some("Ask the owner for the required read permission."),
        );
        return report;
    }
    if value
        .pointer("/result/isError")
        .and_then(|error| error.as_bool())
        == Some(true)
        || !value
            .pointer("/result/structuredContent/error")
            .is_some_and(|error| error.is_null())
    {
        return report;
    }
    let Some(data) = value.pointer("/result/structuredContent/data") else {
        report.authorization = Readiness::unknown(
            "Status support or permissions could not be established.",
            "Check service version and read permissions.",
        );
        return report;
    };
    let Ok(service) = serde_json::from_value::<ServiceReadiness>(data.clone()) else {
        return report;
    };
    if service.schema_version != 1 || !service.read_only || service.inference_probed {
        return report;
    }
    report.authorization = Readiness::new(
        ReadinessState::Ready,
        "Authenticated read/status permission was accepted.",
        None,
    );
    report.application = service.application.clone();
    report.jobs = service.jobs.clone();
    report.service = Some(service);
    report.exit_code = 1; // Unknown freshness/backup/provider evidence remains a warning.
    if report.application.storage.state == ReadinessState::Unavailable {
        report.exit_code = 2;
    }
    report
}

pub(crate) async fn run(
    cli: &Cli,
    server: &str,
    format: DoctorFormat,
    freshness: Option<&Path>,
) -> Result<()> {
    let mut report = collect(cli, server).await;
    if let Some(path) = freshness {
        let evidence = report
            .service
            .as_ref()
            .and_then(|service| load_refresh(path, server, &service.instance_id));
        if let Some(evidence) = evidence {
            let retained_graph = evidence.retained_extraction_policy_parts.unwrap_or(0) > 0;
            let state = match evidence.status.as_str() {
                "failed" => ReadinessState::Unavailable,
                "partial" | "running" | "paused" => ReadinessState::Partial,
                _ if retained_graph => ReadinessState::Partial,
                _ => ReadinessState::Unknown,
            };
            report.application.sources = if retained_graph {
                Readiness::new(state, "Bound client evidence describes indexed originals/vectors separately from retained older graph policy; current source freshness is unknown.", Some("Inspect collection status, and have the owner explicitly reprocess the retained graph-policy parts when ready."))
            } else {
                Readiness::new(state, "Bound client collection refresh evidence is available; it does not prove the source is still current.", Some("Inspect client_refresh_evidence and run refresh/resume as appropriate."))
            };
            if state == ReadinessState::Unavailable {
                report.exit_code = 2;
            }
            report.client_refresh_evidence = Some(evidence);
        } else {
            report.application.sources = Readiness::unknown("Client refresh evidence is invalid, unavailable, or bound to another endpoint/principal.", "Regenerate the private status file from the collection refresh adapter.");
        }
    }
    match format {
        DoctorFormat::Json => println!("{}", serde_json::to_string_pretty(&report)?),
        DoctorFormat::Human => {
            println!("GraphRAG remote doctor (exit {})", report.exit_code);
            for (name, check) in [
                ("transport", &report.transport),
                ("authorization", &report.authorization),
                ("storage", &report.application.storage),
                ("embeddings", &report.application.embeddings),
                ("extraction", &report.application.extraction),
                ("sources", &report.application.sources),
                ("backup", &report.application.backup),
            ] {
                println!(
                    "[{:?}] {name}: {}",
                    check.state,
                    output::safe_text(&check.summary, false)
                );
                if let Some(action) = &check.next_action {
                    println!("  {}", output::safe_text(action, false));
                }
            }
            println!(
                "[{:?}] jobs: {} ({} recent jobs sampled, limit {})",
                report.jobs.readiness.state,
                output::safe_text(&report.jobs.readiness.summary, false),
                report.jobs.sampled,
                report.jobs.limit
            );
            if !report.jobs.counts.is_empty() {
                println!("  Recent owned job states: {:?}", report.jobs.counts);
            }
            if let Some(evidence) = &report.client_refresh_evidence {
                println!(
                    "Client collection {}: {}; failed {}, missing {}, pending parts {}",
                    evidence.collection_id,
                    evidence.status,
                    evidence.counts.failed,
                    evidence.counts.missing,
                    evidence.pending_parts
                );
                println!(
                    "  Last attempt: {}; last successful refresh: {}",
                    evidence.last_attempt_at.as_deref().unwrap_or("unknown"),
                    evidence.last_success_at.as_deref().unwrap_or("unknown")
                );
                if let Some(parts) = evidence.retained_extraction_policy_parts {
                    println!("  Retained older extraction-policy parts: {parts}; collection status describes indexed original/vector evidence separately. Graph policy needs explicit owner reprocessing.");
                }
                if let Some(plan) = &evidence.retry.plan_sha256 {
                    println!("  Pending cleanup retry: review the saved plan, then use the refresh adapter with --reconcile --yes --plan-sha256 {plan}.");
                }
            }
            println!("Read-only: no provider probes, inference, corpus mutation, backup, or second database owner.");
        }
    }
    Err(crate::app::DoctorExit(report.exit_code).into())
}
