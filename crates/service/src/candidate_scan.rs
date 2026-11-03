//! Separately authorized, bounded relationship candidate scans.
use crate::{
    tools::{application_failure, definition, failure, parse, success, ToolService},
    Capability, Principal,
};
use graphrag_application::{
    CallerIdentity, RelationshipCandidateScanRequest, RelationshipCandidateScanResponse,
};
use rmcp::model::{CallToolRequestParams, CallToolResult, Tool, ToolAnnotations};
use std::sync::Arc;

pub(crate) const TOOL: &str = "scan_relationship_candidates";

pub(crate) fn catalog() -> (Capability, Tool) {
    (
        Capability::Garden,
        definition::<RelationshipCandidateScanRequest, RelationshipCandidateScanResponse>(
            TOOL,
            "Nominate high-precision cross-source related_to candidates for human review. Same-source pairs are excluded; a pair needs similarity at or above the operator floor plus a shared specific (non-generic) entity or an explicit cross-reference. Required budgets: max_candidates, max_endpoints, deadline_ms; dry_run is required. The receipt echoes every resolved limit and policy value, why pairs were rejected and a resume cursor; zero candidates is a valid result. dry_run=false records pending proposals only through the normal review lifecycle (existing proposals and prior decisions are skipped, never refreshed); it never accepts, rejects or creates an accepted edge. Review results with list_proposals/get_proposal and decide_proposal.",
            false,
        )
        .with_annotations(
            ToolAnnotations::new()
                .read_only(false)
                .destructive(false)
                .idempotent(false)
                .open_world(false),
        ),
    )
}

pub(crate) async fn call(
    service: &ToolService,
    request: CallToolRequestParams,
    principal: Principal,
) -> CallToolResult {
    let input: RelationshipCandidateScanRequest = match parse(request.arguments) {
        Ok(value) => value,
        Err(error) => return error,
    };
    // One scan at a time: each examined endpoint is an exact similarity scan.
    let Ok(scan_permit) = service.garden_gate.clone().try_acquire_owned() else {
        return failure(
            "busy",
            "A candidate scan is already running; retry after it finishes.",
            true,
        );
    };
    let write = if input.dry_run {
        None
    } else {
        let Ok(permit) = service.write_gate.clone().try_acquire_owned() else {
            return failure(
                "busy",
                "Write capacity is busy; retry the candidate scan shortly.",
                true,
            );
        };
        let Some(guard) = service.writes.start() else {
            return failure(
                "service_unavailable",
                "The service is shutting down; retry the candidate scan later.",
                true,
            );
        };
        Some((permit, guard))
    };
    let application = Arc::clone(&service.application);
    // Detached so an HTTP disconnect cannot interrupt proposal persistence;
    // the scan's own deadline bounds it, and shutdown drains its write guard.
    match tokio::spawn(async move {
        let _held = (scan_permit, write);
        application
            .scan_relationship_candidates_remote(
                CallerIdentity {
                    instance_id: principal.instance_id,
                },
                input,
            )
            .await
    })
    .await
    {
        Ok(Ok(value)) => success(value),
        Ok(Err(error)) => application_failure(error),
        Err(_) => failure(
            "internal",
            "The candidate scan outcome is uncertain; list pending proposals before scanning again.",
            true,
        ),
    }
}
