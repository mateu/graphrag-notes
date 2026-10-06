use crate::{server::Writes, Capability, Principal};
use graphrag_application::{
    ActionCancellation, ApplicationError, BuildContextRequest, CallerIdentity, CaptureProvenance,
    ContextResponse, GraphPolicy, RecordRef, RemoteApplicationOperations, RemoteCaptureRequest,
    RemoteCaptureResponse, Scope, SearchMode, SearchRequest,
};
use rmcp::{
    model::{
        CacheScope, CallToolRequestParams, CallToolResponse, CallToolResult, ErrorData,
        Implementation, ListToolsResult, PaginatedRequestParams, ServerCapabilities, ServerConfig,
        Tool, ToolAnnotations,
    },
    service::RequestContext,
    RoleServer, ServerHandler,
};
use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{io, sync::Arc};
use tokio::sync::Semaphore;
use tracing::Instrument;

#[derive(Clone)]
pub(crate) struct ToolService {
    pub(crate) application: Arc<dyn RemoteApplicationOperations>,
    pub(crate) writes: Arc<Writes>,
    pub(crate) write_gate: Arc<Semaphore>,
    admission_gate: Arc<Semaphore>,
    cancellation_gate: Arc<Semaphore>,
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
            admission_gate: Arc::new(Semaphore::new(concurrency)),
            cancellation_gate: Arc::new(Semaphore::new(crate::server::CANCELLATION_CAPACITY)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::ser::SerializeSeq;
    use std::cell::Cell;

    struct LargeSequence<'a>(&'a Cell<usize>);

    impl Serialize for LargeSequence<'_> {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut sequence = serializer.serialize_seq(Some(1_000_000))?;
            for _ in 0..1_000_000 {
                self.0.set(self.0.get() + 1);
                sequence.serialize_element(&"x".repeat(1024))?;
            }
            sequence.end()
        }
    }

    #[test]
    fn oversized_output_stops_serialization_before_materializing_the_payload() {
        let visited = Cell::new(0);
        let result = success(LargeSequence(&visited));
        assert_eq!(
            result.structured_content.unwrap()["error"]["code"],
            "response_too_large"
        );
        assert!(
            visited.get() < 2100,
            "must stop near 2 MiB rather than serialize all million entries"
        );
    }

    #[test]
    fn response_limit_counts_encoded_escapes_and_accepts_exact_byte_boundary() {
        assert!(check_encoded_size(&"a".repeat(MAX_RESPONSE_BYTES - 2)).is_ok());
        assert!(matches!(
            check_encoded_size(&"a".repeat(MAX_RESPONSE_BYTES - 1)),
            Err(EncodingFailure::TooLarge)
        ));
        assert!(matches!(
            check_encoded_size(&"\0".repeat(MAX_RESPONSE_BYTES / 6 + 1)),
            Err(EncodingFailure::TooLarge)
        ));
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

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct StatusInput {}
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
    #[schemars(length(max = 32), inner(length(min = 1, max = 64)))]
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

pub(crate) fn definition<I: JsonSchema + 'static, O: JsonSchema + 'static>(
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
    let mut tools = vec![
        (Capability::Read, definition::<StatusInput, graphrag_application::ServiceReadiness>("service_status", "Bounded authenticated readiness through the owning service: storage, configured/cached providers, refresh/backup evidence and permitted own jobs. No inference or mutation. Unknown evidence is not healthy; no private host paths are exposed.", true)),
        (Capability::Read, definition::<SearchInput, SearchOutput>("search_notes", "Search shared notes and chat records. Use mode=keyword and graph=off for provider-free retrieval. Source URIs describe the server's corpus.", true)),
        (Capability::Read, definition::<InspectInput, InspectionOutput>("get_record", "Inspect an exact record ID with bounded chat context. Supply the search revision to reject stale selection. Server source paths are provenance, never client file actions.", true)),
        (Capability::Read, definition::<ContextInput, ContextResponse>("build_context", "Build bounded, cited context from shared notes and chats using server-owned providers and defaults.", true)),
        (Capability::Capture, definition::<CaptureInput, RemoteCaptureResponse>("capture_note", "Capture a new shared note. Reuse the same request_id and identical payload after interruption; authenticated instance identity is supplied by the server. Changed payloads under the same request_id are rejected.", false)),
    ];
    tools.extend(crate::mutations::catalog());
    tools.extend(crate::uploads::catalog());
    tools
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

pub(crate) fn parse<T: DeserializeOwned>(
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

const MAX_RESPONSE_BYTES: usize = 2 * 1024 * 1024;

enum EncodingFailure {
    TooLarge,
    Invalid,
}

/// Count encoded bytes without accumulating output, and stop serialization as
/// soon as the limit is crossed. This runs before creating JSON values/buffers.
#[derive(Default)]
struct ResponseSizeWriter {
    written: usize,
    exceeded: bool,
}

impl io::Write for ResponseSizeWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > MAX_RESPONSE_BYTES.saturating_sub(self.written) {
            self.exceeded = true;
            return Err(io::Error::other("response byte limit exceeded"));
        }
        self.written += bytes.len();
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn check_encoded_size<T: Serialize>(value: &T) -> Result<(), EncodingFailure> {
    let mut writer = ResponseSizeWriter::default();
    serde_json::to_writer(&mut writer, value).map_err(|_| {
        if writer.exceeded {
            EncodingFailure::TooLarge
        } else {
            EncodingFailure::Invalid
        }
    })
}

fn bounded_value<T: Serialize>(value: T) -> Result<Value, EncodingFailure> {
    check_encoded_size(&value)?;
    serde_json::to_value(value).map_err(|_| EncodingFailure::Invalid)
}

fn encoding_failure(error: EncodingFailure) -> CallToolResult {
    match error {
        EncodingFailure::TooLarge => failure("response_too_large", "The result exceeds the service's 2 MiB response limit. Narrow the query, reduce limit/neighbors or context budgets, or ask the corpus owner to inspect the large record locally.", false),
        EncodingFailure::Invalid => failure("internal", "Cannot encode the tool result; contact the service owner.", false),
    }
}

pub(crate) fn success<T: Serialize>(data: T) -> CallToolResult {
    match bounded_value(Envelope {
        schema_version: 1,
        data: Some(data),
        error: None,
    }) {
        Ok(value) => {
            let result = CallToolResult::structured(value);
            match check_encoded_size(&result) {
                Ok(()) => result,
                Err(error) => encoding_failure(error),
            }
        }
        Err(error) => encoding_failure(error),
    }
}

pub(crate) fn failure(code: &str, message: &str, retryable: bool) -> CallToolResult {
    CallToolResult::structured_error(
        json!({"schema_version":1,"data":null,"error":{"code":code,"message":message,"retryable":retryable}}),
    )
}

pub(crate) fn application_failure(error: ApplicationError) -> CallToolResult {
    // Application/provider diagnostics may contain host paths or secrets. The
    // network boundary exposes stable categories and recovery guidance only.
    match error {
        ApplicationError::Validation(_) => failure("invalid_input", "The operation rejected these arguments; check the tool schema and request bounds.", false),
        ApplicationError::NotFound(_) => failure("not_found", "The selected record is unavailable. Search again to refresh its ID.", false),
        ApplicationError::RevisionConflict(_) => failure("revision_conflict", "The record or request payload changed. Refresh the snapshot; retain the same request_id and payload for an uncertain outcome. A new intent requires a new request_id.", false),
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
                .filter(|(capability, tool)| {
                    crate::mutations::allows_catalog(principal, *capability, tool.name.as_ref())
                })
                .map(|(_, tool)| tool)
                .collect(),
            ..Default::default()
        }
        // Required by modern MCP discovery. The catalog depends on the bearer
        // principal, so it must never be cached across authorization contexts.
        .with_ttl_ms(0)
        .with_cache_scope(CacheScope::Private))
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
        let id = context.id.clone();
        let result = self.dispatch_tool(request, context).await?;
        // Serialize the exact SDK response model, including the echoed RPC ID.
        // The HTTP transport adds no further JSON body fields.
        let response = rmcp::model::JsonRpcResponse {
            jsonrpc: rmcp::model::JsonRpcVersion2_0,
            id,
            result: rmcp::model::ServerResult::from(result.clone()),
        };
        match check_encoded_size(&response) {
            Ok(()) => Ok(result),
            Err(EncodingFailure::TooLarge) => Ok(failure("response_too_large", "The complete response exceeds the service's 2 MiB output limit. Narrow the query or budgets, and use a shorter protocol request ID.", false).into()),
            Err(EncodingFailure::Invalid) => Ok(failure("internal", "Cannot encode the response; contact the service owner.", false).into()),
        }
    }
}

impl ToolService {
    async fn dispatch_tool(
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
        if !crate::mutations::allows_catalog(&principal, required, request.name.as_ref()) {
            return Ok(failure(
                "forbidden",
                "This instance token does not grant the required capability.",
                false,
            )
            .into());
        }
        let result = match request.name.as_ref() {
            "service_status" => {
                let _: StatusInput = match parse(request.arguments) {
                    Ok(input) => input,
                    Err(error) => return Ok(error.into()),
                };
                success(crate::status::report(self.application.as_ref(), &principal).await)
            }
            "search_notes" => {
                let input: SearchInput = match parse(request.arguments) {
                    Ok(input) => input,
                    Err(error) => return Ok(error.into()),
                };
                if !query_valid(&input.query)
                    || input
                        .since_days
                        .is_some_and(|days| days > graphrag_application::MAX_REMOTE_SINCE_DAYS)
                    || !(1..=200).contains(&input.limit)
                    || !string_bound(input.source_uri.as_deref(), 2048)
                {
                    return Ok(failure("invalid_input", "Query must contain 1–1024 characters, limit 1–200, since_days at most 365000, and source_uri at most 2048 characters.", false).into());
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
                // A bounded hash correlates private debug phases with a client's
                // RPC without logging an arbitrary client-supplied request ID.
                let encoded_id = serde_json::to_vec(&context.id).map_err(|_| {
                    ErrorData::internal_error("Cannot encode request identity.", None)
                })?;
                let rpc_id_sha256 = format!("{:x}", Sha256::digest(encoded_id));
                let span = tracing::debug_span!("mcp_search", rpc_id_sha256 = %rpc_id_sha256);
                let cancellation = ActionCancellation::new();
                let result = async {
                    let started = std::time::Instant::now();
                    let result = tokio::select! {
                        result = self.application.search(request, cancellation.clone()) => result,
                        _ = context.ct.cancelled() => { cancellation.cancel(); Err(ApplicationError::Cancelled) }
                    };
                    tracing::debug!(
                        phase = "application_search",
                        elapsed_ms = started.elapsed().as_secs_f64() * 1000.0,
                        success = result.is_ok(),
                        "Retrieval phase completed"
                    );
                    result
                }.instrument(span).await;
                match result {
                    Ok(records) => match bounded_value(records) {
                        Ok(value) => match serde_json::from_value::<Vec<RecordOutput>>(value) {
                            Ok(records) => success(SearchOutput { records }),
                            Err(_) => encoding_failure(EncodingFailure::Invalid),
                        },
                        Err(error) => encoding_failure(error),
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
                    Ok(record) => match bounded_value(record) {
                        Ok(value) => match serde_json::from_value::<InspectionOutput>(value) {
                            Ok(record) => success(record),
                            Err(_) => encoding_failure(EncodingFailure::Invalid),
                        },
                        Err(error) => encoding_failure(error),
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
                    || input
                        .since_days
                        .is_some_and(|days| days > graphrag_application::MAX_REMOTE_SINCE_DAYS)
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
            name if matches!(
                name,
                "upload_source" | "cancel_job" | "resume_job" | "delete_uploaded_source"
            ) =>
            {
                // These short durable mutations never share the provider/capture
                // semaphore. Explicit cancellation remains reachable while a
                // worker is preparing or inside a guarded atomic write.
                let gate = if name == "cancel_job" {
                    &self.cancellation_gate
                } else {
                    &self.admission_gate
                };
                let Ok(permit) = gate.clone().try_acquire_owned() else {
                    return Ok(failure("busy", "Job admission or control capacity is busy; retry the identical request shortly.", true).into());
                };
                let Some(guard) = self.writes.start() else {
                    return Ok(failure("service_unavailable", "The service is shutting down; preserve the original upload request ID and draft for retry.", true).into());
                };
                let application = Arc::clone(&self.application);
                let name = name.to_owned();
                let task = tokio::spawn(async move {
                    let (_permit, _guard) = (permit, guard);
                    crate::uploads::dispatch(
                        application.as_ref(),
                        &principal,
                        &name,
                        request.arguments,
                    )
                    .await
                });
                match task.await {
                    Ok(result) => result,
                    Err(_) => failure("internal", "The job control outcome is uncertain; inspect the existing job or retry the identical upload request ID.", true),
                }
            }
            name @ ("get_source" | "get_job" | "list_jobs") => {
                crate::uploads::dispatch(
                    self.application.as_ref(),
                    &principal,
                    name,
                    request.arguments,
                )
                .await
            }
            _ => crate::mutations::call(self, request, principal).await,
        };
        Ok(result.into())
    }
}
