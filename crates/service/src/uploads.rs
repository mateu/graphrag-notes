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

pub(crate) fn catalog() -> Vec<(Capability, Tool)> {
    vec![
        (Capability::Upload, definition::<UploadSourceRequest, UploadAdmission>("upload_source", "Upload supplied UTF-8 Markdown (at most 65536 bytes) under an opaque document_key scoped to this authenticated instance. No path is read or URL fetched. Reuse identical payload/request_id after a lost response; processing belongs to the returned durable job.", false)),
        (Capability::Read, definition::<SourceInput, UploadedSource>("get_source", "Inspect an uploaded source's latest supplied content and provenance, with attempted and successful generation labels. Server URIs are metadata, never client file actions.", true)),
        (Capability::Delete, definition::<DeleteUploadedSourceRequest, graphrag_application::RemoteMutationResponse>("delete_uploaded_source", "Explicitly retire one ready uploaded source owned by this authenticated instance and registered collection_id, using its reviewed revision and confirmed=true. Generated chunks are removed; detached/manual notes and durable receipt history survive. Retry identical request_id/payload after an uncertain response.", false).with_annotations(ToolAnnotations::new().read_only(false).destructive(true).idempotent(true).open_world(false))),
        (Capability::Jobs, definition::<JobInput, RemoteJobStatus>("get_job", "Inspect this instance's server-owned uploaded Markdown job, safe checkpoint and result. HTTP disconnects do not cancel work.", true)),
        (Capability::Jobs, definition::<JobsInput, RemoteJobList>("list_jobs", "List bounded uploaded jobs owned by this authenticated instance.", true)),
        (Capability::Jobs, definition::<JobInput, RemoteJobStatus>("cancel_job", "Request explicit cancellation of this instance's job at a safe write boundary. Cancellation remains available while workers prepare or persist content.", false)),
        (Capability::Jobs, definition::<JobInput, RemoteJobStatus>("resume_job", "Resume this instance's interrupted/failed/cancelled supported upload job from its durable input and checkpoint. Running workers cannot be claimed twice; newer source generations cause a conflict.", false)),
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
                (Some(id), None) => application.get_uploaded_source(&id).await,
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
            match application.list_remote_jobs(caller, input.limit).await {
                Ok(result) => success(result),
                Err(error) => application_failure(error),
            }
        }
        "get_job" | "cancel_job" | "resume_job" => {
            let input: JobInput = match parse(arguments) {
                Ok(input) => input,
                Err(error) => return error,
            };
            let result = match name {
                "get_job" => application.get_remote_job(caller, &input.id).await,
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
