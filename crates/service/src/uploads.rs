//! Upload and own-job tools: caller identity is always supplied by authentication.
use crate::{
    tools::{application_failure, definition, failure, parse, success},
    Capability, Principal,
};
use graphrag_application::{
    CallerIdentity, DeleteUploadedSourceRequest, RemoteApplicationOperations, RemoteJobList,
    RemoteJobStatus, UploadAdmission, UploadSourceRequest, UploadedSource,
};
use rmcp::model::{CallToolResult, Tool, ToolAnnotations};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Map, Value};

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct SourceInput {
    /// Canonical source:ID returned by upload_source; never a client/server path.
    #[schemars(length(min = 1, max = 512))]
    id: Option<String>,
    /// Look up this authenticated instance's existing opaque document key.
    #[schemars(length(min = 1, max = 256))]
    document_key: Option<String>,
}
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct JobInput {
    #[schemars(length(min = 1, max = 512))]
    id: String,
}
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct JobsInput {
    #[serde(default = "default_limit")]
    #[schemars(range(min = 1, max = 100))]
    limit: usize,
}
fn default_limit() -> usize {
    20
}

/// The application maps the database's non-decoding family classification into
/// `job_type`, so damaged saved enrichment plans remain hidden without decoding.
pub(crate) fn job_visible(principal: &Principal, job: &RemoteJobStatus) -> bool {
    job.job_type != "remote_enrichment" || principal.allows(Capability::Enrich)
}

pub(crate) fn catalog() -> Vec<(Capability, Tool)> {
    let mut tools = vec![
        (Capability::Upload, definition::<UploadSourceRequest, UploadAdmission>("upload_source", "Upload supplied UTF-8 Markdown (at most 65536 bytes) under an opaque document_key scoped to this authenticated instance. No path is read or URL fetched. Reuse identical payload/request_id after a lost response; processing belongs to the returned durable job.", false)),
        (Capability::Read, definition::<SourceInput, UploadedSource>("get_source", "Inspect an uploaded source's latest supplied content and provenance, with attempted and successful generation labels. Server URIs are metadata, never client file actions.", true)),
        (Capability::Delete, definition::<DeleteUploadedSourceRequest, graphrag_application::RemoteMutationResponse>("delete_uploaded_source", "Explicitly retire one ready uploaded source owned by this authenticated instance and registered collection_id, using its reviewed revision and confirmed=true. Generated chunks are removed; detached/manual notes and durable receipt history survive. Retry identical request_id/payload after an uncertain response.", false).with_annotations(ToolAnnotations::new().read_only(false).destructive(true).idempotent(true).open_world(false))),
        (Capability::Jobs, definition::<JobInput, RemoteJobStatus>("get_job", "Inspect this instance's server-owned uploaded Markdown job, safe checkpoint and result. HTTP disconnects do not cancel work.", true)),
        (Capability::Jobs, definition::<JobsInput, RemoteJobList>("list_jobs", "List bounded uploaded jobs owned by this authenticated instance.", true)),
        (Capability::Jobs, definition::<JobInput, RemoteJobStatus>("cancel_job", "Request explicit cancellation of this instance's job at a safe write boundary. Cancellation remains available while workers prepare or persist content.", false)),
        (Capability::Jobs, definition::<JobInput, RemoteJobStatus>("resume_job", "Resume this instance's interrupted/failed/cancelled supported upload job from its durable input and checkpoint. Running workers cannot be claimed twice; newer source generations cause a conflict.", false)),
    ];
    tools.extend(enrichment_catalog());
    tools
}

fn enrichment_catalog() -> Vec<(Capability, Tool)> {
    use graphrag_application::*;
    vec![
 (Capability::Enrich,definition::<PrepareSourceEnrichment,ReviewedEnrichmentReceipt>("prepare_source_enrichment","Read-only exact owned-source reviewed plan; no inference or writes. DEVELOPMENT ONLY.",true)),
 (Capability::Enrich,definition::<ExecuteSourceEnrichment,UploadAdmission>("execute_source_enrichment","Explicit confirmed exact-plan conversion queued under durable owner lease; no acceptance or scans. DEVELOPMENT ONLY.",false)),
 (Capability::Enrich,definition::<ExecuteSourceEnrichment,UploadAdmission>("rollback_source_enrichment","Separate confirmed exact-plan forward compensation, queued under durable owner lease. DEVELOPMENT ONLY.",false)),
 (Capability::Enrich,definition::<JobInput,ReviewedEnrichmentReceipt>("read_source_enrichment_plan","Read exact persisted owned enrichment plan and policy digests. DEVELOPMENT ONLY.",true)),
 (Capability::Enrich,definition::<JobInput,RemoteJobStatus>("get_source_enrichment_job","Read owned conversion outcome and canonical current source readback. DEVELOPMENT ONLY.",true)),
 (Capability::Enrich,definition::<JobInput,RemoteJobStatus>("cancel_source_enrichment_job","Request cancellation before atomic publication entry; committed conversion remains completed. DEVELOPMENT ONLY.",false)),
 (Capability::Enrich,definition::<JobInput,RemoteJobStatus>("retry_source_enrichment_job","Retry exact durable owned plan/checkpoints; changed source or policy fails closed. DEVELOPMENT ONLY.",false)),
 ]
}

pub(crate) async fn dispatch(
    application: &dyn RemoteApplicationOperations,
    principal: &Principal,
    name: &str,
    arguments: Option<Map<String, Value>>,
) -> CallToolResult {
    let caller = CallerIdentity {
        instance_id: principal.instance_id.clone(),
    };
    match name {
        "prepare_source_enrichment" => {
            let input = match parse(arguments) {
                Ok(x) => x,
                Err(e) => return e,
            };
            match application.prepare_source_enrichment(caller, input).await {
                Ok(x) => success(x),
                Err(e) => application_failure(e),
            }
        }
        "execute_source_enrichment" | "rollback_source_enrichment" => {
            let input = match parse(arguments) {
                Ok(x) => x,
                Err(e) => return e,
            };
            match application
                .execute_source_enrichment(caller, input, name == "rollback_source_enrichment")
                .await
            {
                Ok(x) => success(x),
                Err(e) => application_failure(e),
            }
        }
        "read_source_enrichment_plan"
        | "get_source_enrichment_job"
        | "cancel_source_enrichment_job"
        | "retry_source_enrichment_job" => {
            let input: JobInput = match parse(arguments) {
                Ok(x) => x,
                Err(e) => return e,
            };
            // Validate the job family and credential owner before any generic job control.
            let receipt = match application
                .read_source_enrichment_plan(caller.clone(), &input.id)
                .await
            {
                Ok(x) => x,
                Err(e) => return application_failure(e),
            };
            if name == "read_source_enrichment_plan" {
                return success(receipt);
            }
            let result = match name {
                "cancel_source_enrichment_job" => {
                    application
                        .cancel_remote_job(caller.clone(), &input.id)
                        .await
                }
                "retry_source_enrichment_job" => {
                    application
                        .resume_remote_job(caller.clone(), &input.id)
                        .await
                }
                _ => application.get_remote_job(caller.clone(), &input.id).await,
            };
            match result {
                Ok(mut job) => {
                    if let Some(result) = job.result.as_mut() {
                        result["reviewed"] = serde_json::to_value(receipt.clone()).unwrap();
                        match application
                            .get_owned_uploaded_source(
                                caller.clone(),
                                receipt.plan["source_id"].as_str().unwrap_or_default(),
                            )
                            .await
                        {
                            Ok(current) => {
                                result["current_source"] = serde_json::to_value(current).unwrap()
                            }
                            Err(e) => return application_failure(e),
                        }
                    }
                    success(job)
                }
                Err(e) => application_failure(e),
            }
        }

        "delete_uploaded_source" => {
            let request = match parse(arguments) {
                Ok(request) => request,
                Err(error) => return error,
            };
            match application.delete_uploaded_source(caller, request).await {
                Ok(result) => success(result),
                Err(error) => application_failure(error),
            }
        }
        "upload_source" => {
            let request = match parse(arguments) {
                Ok(request) => request,
                Err(error) => return error,
            };
            match application.upload_source(caller, request).await {
                Ok(result) => success(result),
                Err(error) => application_failure(error),
            }
        }
        "get_source" => {
            let input: SourceInput = match parse(arguments) {
                Ok(input) => input,
                Err(error) => return error,
            };
            let result = match (input.id, input.document_key) {
                (Some(id), None) => application.get_owned_uploaded_source(caller, &id).await,
                (None, Some(key)) => application.lookup_uploaded_source(caller, &key).await,
                _ => {
                    return failure(
                        "invalid_input",
                        "Supply exactly one of id or document_key",
                        false,
                    )
                }
            };
            match result {
                Ok(result) => success(result),
                Err(error) => application_failure(error),
            }
        }
        "list_jobs" => {
            let input: JobsInput = match parse(arguments) {
                Ok(input) => input,
                Err(error) => return error,
            };
            match application
                .list_remote_jobs(
                    caller.clone(),
                    input.limit,
                    principal.allows(Capability::Enrich),
                )
                .await
            {
                Ok(mut result) => {
                    result.jobs.retain(|job| job_visible(principal, job));
                    success(result)
                }
                Err(error) => application_failure(error),
            }
        }
        "get_job" | "cancel_job" | "resume_job" => {
            let input: JobInput = match parse(arguments) {
                Ok(input) => input,
                Err(error) => return error,
            };
            let job = match application.get_remote_job(caller.clone(), &input.id).await {
                Ok(job) => job,
                Err(error) => return application_failure(error),
            };
            if !job_visible(principal, &job) {
                return failure(
                    "forbidden",
                    "Enrichment control requires explicit enrich capability",
                    false,
                );
            }
            if name == "get_job" {
                return success(job);
            }
            let result = match name {
                "cancel_job" => application.cancel_remote_job(caller, &input.id).await,
                _ => application.resume_remote_job(caller, &input.id).await,
            };
            match result {
                Ok(result) => success(result),
                Err(error) => application_failure(error),
            }
        }
        _ => failure("invalid_input", "Unknown tool; refresh tools/list.", false),
    }
}
