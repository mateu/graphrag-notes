use crate::{server::Writes, Capability, Principal};
use graphrag_application::{
    ActionCancellation, ApplicationError, BuildContextRequest, CallerIdentity, CaptureProvenance,
    ContextResponse, GraphPolicy, RecordRef, RemoteApplicationOperations, RemoteCaptureRequest,
    RemoteCaptureResponse, Scope, SearchMode, SearchRequest,
};
use rmcp::{
    model::{
        CallToolRequestParams, CallToolResponse, CallToolResult, ErrorData, Implementation,
        ListToolsResult, PaginatedRequestParams, ServerCapabilities, ServerConfig, Tool,
        ToolAnnotations,
    },
    service::RequestContext,
    RoleServer, ServerHandler,
};
use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::Semaphore;

#[derive(Clone)]
pub(crate) struct ToolService {
    application: Arc<dyn RemoteApplicationOperations>,
    writes: Arc<Writes>,
    write_gate: Arc<Semaphore>,
}

impl ToolService {
    pub(crate) fn new(
        application: Arc<dyn RemoteApplicationOperations>,
        writes: Arc<Writes>,
        concurrency: usize,
    ) -> Self {
        Self {
            application,
            writes,
            write_gate: Arc::new(Semaphore::new(concurrency)),
        }
    }
}

#[derive(Debug, Serialize, JsonSchema)]
struct RemoteFailure {
    code: String,
    message: String,
    retryable: bool,
}

#[derive(Serialize, JsonSchema)]
struct Envelope<T> {
    schema_version: u32,
    data: Option<T>,
    error: Option<RemoteFailure>,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct SearchInput {
    #[schemars(length(min = 1, max = 1024))]
    query: String,
    #[serde(default = "hybrid")]
    mode: SearchMode,
    #[serde(default = "all")]
    scope: Scope,
    #[serde(default = "default_limit")]
    #[schemars(range(min = 1, max = 200))]
    limit: usize,
    #[serde(default = "auto")]
    graph: GraphPolicy,
    #[schemars(range(min = 0, max = 365000))]
    since_days: Option<u32>,
    #[schemars(length(max = 2048))]
    source_uri: Option<String>,
}
fn hybrid() -> SearchMode {
    SearchMode::Hybrid
}
fn all() -> Scope {
    Scope::All
}
fn auto() -> GraphPolicy {
    GraphPolicy::Auto
}
fn default_limit() -> usize {
    10
}
fn default_neighbors() -> usize {
    2
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct InspectInput {
    #[schemars(length(min = 1, max = 512))]
    id: String,
    #[schemars(length(max = 128))]
    revision: Option<String>,
    #[serde(default = "default_neighbors")]
    #[schemars(range(min = 0, max = 20))]
    neighbors: usize,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ContextInput {
    #[schemars(length(min = 1, max = 1024))]
    query: String,
    #[serde(default = "all")]
    scope: Scope,
    #[serde(default = "auto")]
    graph: GraphPolicy,
    #[schemars(range(min = 0, max = 365000))]
    since_days: Option<u32>,
    #[schemars(length(max = 2048))]
    source_uri: Option<String>,
    #[schemars(length(max = 512))]
    entity_filter: Option<String>,
    #[schemars(range(min = 0, max = 200))]
    max_chunks: Option<usize>,
    #[schemars(range(min = 0, max = 32768))]
    max_total_tokens: Option<usize>,
    #[schemars(range(min = 0, max = 8192))]
    max_chunk_tokens: Option<usize>,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct CaptureInput {
    /// Persist this identifier and reuse it for uncertain/retried writes. Maximum
    /// 128 Unicode characters and 256 UTF-8 bytes; byte limits are checked semantically.
    #[schemars(length(min = 1, max = 128))]
    request_id: String,
    /// Nonempty text, at most 65536 UTF-8 bytes (checked semantically).
    /// JSON Schema maxLength counts Unicode characters, rather than bytes.
    #[schemars(length(min = 1, max = 65536))]
    content: String,
    #[schemars(length(max = 512))]
    title: Option<String>,
    #[serde(default)]
    #[schemars(length(max = 32))]
    tags: Vec<String>,
    provenance: Option<CaptureProvenance>,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct SearchOutput {
    records: Vec<RecordOutput>,
}
#[derive(Serialize, Deserialize, JsonSchema)]
struct RecordOutput {
    id: String,
    revision: Option<String>,
    hit_type: String,
    title: Option<String>,
    content: String,
    provenance: Option<Value>,
    score: Option<f32>,
    warnings: Vec<String>,
}
#[derive(Serialize, Deserialize, JsonSchema)]
struct InspectionOutput {
    id: String,
    hit_type: String,
    title: Option<String>,
    content: String,
    revision: String,
    provenance: Value,
    conversations: Vec<ConversationOutput>,
    messages: Vec<MessageOutput>,
    messages_truncated: bool,
    warnings: Vec<String>,
}
#[derive(Serialize, Deserialize, JsonSchema)]
struct ConversationOutput {
    id: String,
    uuid: String,
    title: Option<String>,
    summary: Option<String>,
    source_uri: Option<String>,
    revision: String,
}
#[derive(Serialize, Deserialize, JsonSchema)]
struct MessageOutput {
    id: String,
    message_key: String,
    message_uuid: Option<String>,
    conversation_id: String,
    conversation_uuid: String,
    message_index: i64,
    role: String,
    content: String,
    revision: String,
}

fn definition<I: JsonSchema + 'static, O: JsonSchema + 'static>(
    name: &'static str,
    description: &'static str,
    read_only: bool,
) -> Tool {
    Tool::new(name, description, serde_json::Map::new())
        .with_input_schema::<I>()
        .with_output_schema::<Envelope<O>>()
        .with_annotations(
            ToolAnnotations::new()
                .read_only(read_only)
                .destructive(false)
                .idempotent(true)
                .open_world(false),
        )
}

fn catalog() -> Vec<(Capability, Tool)> {
    vec![
        (Capability::Read, definition::<SearchInput, SearchOutput>("search_notes", "Search shared notes and chat records. Use mode=keyword and graph=off for provider-free retrieval. Source URIs describe the server's corpus.", true)),
        (Capability::Read, definition::<InspectInput, InspectionOutput>("get_record", "Inspect an exact record ID with bounded chat context. Supply the search revision to reject stale selection. Server source paths are provenance, never client file actions.", true)),
        (Capability::Read, definition::<ContextInput, ContextResponse>("build_context", "Build bounded, cited context from shared notes and chats using server-owned providers and defaults.", true)),
        (Capability::Capture, definition::<CaptureInput, RemoteCaptureResponse>("capture_note", "Capture a new shared note. Reuse the same request_id and identical payload after interruption; authenticated instance identity is supplied by the server. Changed payloads under the same request_id are rejected.", false)),
    ]
}

fn principal(context: &RequestContext<RoleServer>) -> Result<&Principal, ErrorData> {
    context
        .extensions
        .get::<axum::http::request::Parts>()
        .and_then(|parts| parts.extensions.get::<Principal>())
        .ok_or_else(|| {
            ErrorData::invalid_request("Authenticated request identity is required.", None)
        })
}

fn parse<T: DeserializeOwned>(
    arguments: Option<serde_json::Map<String, Value>>,
) -> Result<T, CallToolResult> {
    serde_json::from_value(Value::Object(arguments.unwrap_or_default())).map_err(|_| {
        failure(
            "invalid_input",
            "Tool arguments do not match the advertised schema; check required fields and types.",
            false,
        )
    })
}

fn success<T: Serialize>(data: T) -> CallToolResult {
    match serde_json::to_value(Envelope {
        schema_version: 1,
        data: Some(data),
        error: None,
    }) {
        Ok(value) => {
            let result = CallToolResult::structured(value);
            match serde_json::to_vec(&result) {
                Ok(bytes) if bytes.len() <= 2 * 1024 * 1024 => result,
                Ok(_) => failure("response_too_large", "The result exceeds the service's 2 MiB response limit. Narrow the query, reduce limit/neighbors or context budgets, or ask the corpus owner to inspect the large record locally.", false),
                Err(_) => failure("internal", "Cannot encode the tool result; contact the service owner.", false),
            }
        }
        Err(_) => failure(
            "internal",
            "Cannot encode the tool result; contact the service owner.",
            false,
        ),
    }
}

fn failure(code: &str, message: &str, retryable: bool) -> CallToolResult {
    CallToolResult::structured_error(
        json!({"schema_version":1,"data":null,"error":{"code":code,"message":message,"retryable":retryable}}),
    )
}

fn application_failure(error: ApplicationError) -> CallToolResult {
    // Application/provider diagnostics may contain host paths or secrets. The
    // network boundary exposes stable categories and recovery guidance only.
    match error {
        ApplicationError::Validation(_) => failure("invalid_input", "The operation rejected these arguments; check the tool schema and request bounds.", false),
        ApplicationError::NotFound(_) => failure("not_found", "The selected record is unavailable. Search again to refresh its ID.", false),
        ApplicationError::RevisionConflict(_) => failure("revision_conflict", "The record or request payload changed. Refresh the record, or use a new capture request_id for a changed draft.", false),
        ApplicationError::ProviderUnavailable(_) => failure("provider_unavailable", "A server provider is unavailable. For retrieval, retry search_notes with mode=keyword and graph=off; keep capture drafts and their request_id for retry.", true),
        ApplicationError::Compatibility(_) => failure("compatibility", "The server's corpus and embedding configuration differ. Use search_notes with mode=keyword and graph=off, or contact the corpus owner.", false),
        ApplicationError::ServiceUnreachable(_) => failure("service_unreachable", "An upstream service is unreachable; retry later and retain capture drafts with their request_id.", true),
        ApplicationError::Cancelled => failure("cancelled", "The read operation was cancelled. Retry when ready.", true),
        ApplicationError::Internal(_) => failure("internal", "The operation failed internally; contact the service owner before changing or repeating a write.", false),
    }
}

fn string_bound(value: Option<&str>, max: usize) -> bool {
    value.is_none_or(|value| value.chars().count() <= max && !value.contains('\0'))
}
fn query_valid(query: &str) -> bool {
    !query.trim().is_empty() && string_bound(Some(query), 1024)
}
fn bounded(value: Option<usize>, max: usize) -> bool {
    value.is_none_or(|value| value <= max)
}

impl ServerHandler for ToolService {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("graphrag-notes", env!("CARGO_PKG_VERSION")))
            .with_instructions("This is a shared corpus. Read tools never execute local source paths. Capture requires a stable client-generated request_id; preserve it with the draft until an authoritative outcome is received.")
    }

    async fn list_tools(
        &self,
        request: Option<PaginatedRequestParams>,
        context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        let principal = principal(&context)?;
        if request.is_some_and(|request| request.cursor.is_some()) {
            return Err(ErrorData::invalid_params(
                "This compact tool catalog does not use cursors.",
                None,
            ));
        }
        Ok(ListToolsResult {
            tools: catalog()
                .into_iter()
                .filter(|(capability, _)| principal.allows(*capability))
                .map(|(_, tool)| tool)
                .collect(),
            ..Default::default()
        })
    }

    fn get_tool(&self, name: &str) -> Option<Tool> {
        catalog()
            .into_iter()
            .map(|(_, tool)| tool)
            .find(|tool| tool.name == name)
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        let principal = principal(&context)?.clone();
        let Some((required, _)) = catalog()
            .into_iter()
            .find(|(_, tool)| tool.name == request.name)
        else {
            return Err(ErrorData::invalid_params(
                "Unknown tool; refresh tools/list.",
                None,
            ));
        };
        if !principal.allows(required) {
            return Ok(failure(
                "forbidden",
                "This instance token does not grant the required capability.",
                false,
            )
            .into());
        }
        let result = match request.name.as_ref() {
            "search_notes" => {
                let input: SearchInput = match parse(request.arguments) {
                    Ok(input) => input,
                    Err(error) => return Ok(error.into()),
                };
                if !query_valid(&input.query)
                    || !(1..=200).contains(&input.limit)
                    || !string_bound(input.source_uri.as_deref(), 2048)
                {
                    return Ok(failure("invalid_input", "Query must contain 1–1024 characters, limit 1–200, and source_uri at most 2048 characters.", false).into());
                }
                let request = SearchRequest {
                    query: input.query,
                    mode: input.mode,
                    scope: input.scope,
                    limit: input.limit,
                    graph: input.graph,
                    since_days: input.since_days,
                    source_uri: input.source_uri,
                };
                let cancellation = ActionCancellation::new();
                let result = tokio::select! {
                    result = self.application.search(request, cancellation.clone()) => result,
                    _ = context.ct.cancelled() => { cancellation.cancel(); Err(ApplicationError::Cancelled) }
                };
                match result {
                    Ok(records) => match serde_json::to_value(records)
                        .and_then(serde_json::from_value::<Vec<RecordOutput>>)
                    {
                        Ok(records) => success(SearchOutput { records }),
                        Err(_) => failure(
                            "internal",
                            "Cannot encode search records; contact the service owner.",
                            false,
                        ),
                    },
                    Err(error) => application_failure(error),
                }
            }
            "get_record" => {
                let input: InspectInput = match parse(request.arguments) {
                    Ok(input) => input,
                    Err(error) => return Ok(error.into()),
                };
                if input.id.is_empty()
                    || !string_bound(Some(&input.id), 512)
                    || !string_bound(input.revision.as_deref(), 128)
                    || input.neighbors > 20
                {
                    return Ok(failure(
                        "invalid_input",
                        "Record ID or revision exceeds its bounds, or neighbors exceeds 20.",
                        false,
                    )
                    .into());
                }
                match self
                    .application
                    .inspect(
                        RecordRef {
                            id: input.id,
                            revision: input.revision,
                        },
                        input.neighbors,
                    )
                    .await
                {
                    Ok(record) => match serde_json::to_value(record)
                        .and_then(serde_json::from_value::<InspectionOutput>)
                    {
                        Ok(record) => success(record),
                        Err(_) => failure(
                            "internal",
                            "Cannot encode the record; contact the service owner.",
                            false,
                        ),
                    },
                    Err(error) => application_failure(error),
                }
            }
            "build_context" => {
                let input: ContextInput = match parse(request.arguments) {
                    Ok(input) => input,
                    Err(error) => return Ok(error.into()),
                };
                if !query_valid(&input.query)
                    || !string_bound(input.source_uri.as_deref(), 2048)
                    || !string_bound(input.entity_filter.as_deref(), 512)
                    || !bounded(input.max_chunks, 200)
                    || !bounded(input.max_total_tokens, 32768)
                    || !bounded(input.max_chunk_tokens, 8192)
                {
                    return Ok(failure(
                        "invalid_input",
                        "Context query, filter or token budgets exceed the advertised bounds.",
                        false,
                    )
                    .into());
                }
                let request = BuildContextRequest {
                    query: input.query,
                    scope: input.scope,
                    graph: input.graph,
                    since_days: input.since_days,
                    source_uri: input.source_uri,
                    entity_filter: input.entity_filter,
                    max_chunks: input.max_chunks,
                    max_total_tokens: input.max_total_tokens,
                    max_chunk_tokens: input.max_chunk_tokens,
                };
                let cancellation = ActionCancellation::new();
                let result = tokio::select! {
                    result = self.application.build_context(request, cancellation.clone()) => result,
                    _ = context.ct.cancelled() => { cancellation.cancel(); Err(ApplicationError::Cancelled) }
                };
                match result {
                    Ok(context) => success(context),
                    Err(error) => application_failure(error),
                }
            }
            "capture_note" => {
                let input: CaptureInput = match parse(request.arguments) {
                    Ok(input) => input,
                    Err(error) => return Ok(error.into()),
                };
                if input.request_id.is_empty()
                    || !string_bound(Some(&input.request_id), 128)
                    || input.content.trim().is_empty()
                    || input.content.len() > 65536
                    || input.content.contains('\0')
                    || !string_bound(input.title.as_deref(), 512)
                    || input.tags.len() > 32
                    || input.tags.iter().any(|tag| !string_bound(Some(tag), 64))
                {
                    return Ok(failure("invalid_input", "Capture needs a stable request_id (at most 128 characters), nonempty content (at most 65536 bytes), title at most 512 characters, and at most 32 tags of 64 characters.", false).into());
                }
                let Ok(permit) = self.write_gate.clone().try_acquire_owned() else {
                    return Ok(failure("busy", "Capture capacity is busy; retry the identical draft and request_id shortly.", true).into());
                };
                let application = Arc::clone(&self.application);
                let Some(guard) = self.writes.start() else {
                    return Ok(failure(
                        "service_unavailable",
                        "The service is shutting down; retain the draft and request_id for retry.",
                        true,
                    )
                    .into());
                };
                let request = RemoteCaptureRequest {
                    request_id: input.request_id,
                    content: input.content,
                    title: input.title,
                    tags: input.tags,
                    provenance: input.provenance,
                };
                let task = tokio::spawn(async move {
                    let (_permit, _guard) = (permit, guard);
                    application
                        .capture_remote(
                            CallerIdentity {
                                instance_id: principal.instance_id,
                            },
                            request,
                            ActionCancellation::new(),
                        )
                        .await
                });
                // Dropping this waiter on an HTTP disconnect does not cancel the
                // detached mutation; shutdown drains it before releasing storage.
                match task.await {
                    Ok(Ok(record)) => success(record),
                    Ok(Err(error)) => application_failure(error),
                    Err(_) => failure("internal", "Capture outcome is uncertain; retain the identical draft and request_id for a safe retry.", true),
                }
            }
            _ => unreachable!("catalog and dispatch agree"),
        };
        Ok(result.into())
    }
}
