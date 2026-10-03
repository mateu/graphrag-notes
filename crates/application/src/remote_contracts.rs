//! Version-one wire contracts; caller identity is supplied only by a trusted adapter.
use crate::{GraphPolicy, Scope};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Frozen durable payload version, independent of the local application interface.
/// Changes require explicit receipt replay compatibility.
pub const REMOTE_CAPTURE_PAYLOAD_VERSION: u32 = 1;

pub const MAX_REMOTE_CAPTURE_BYTES: usize = 64 * 1024;
pub const MAX_REMOTE_QUERY_CHARS: usize = 1024;
/// A bounded lookback avoids overflowing Chrono date arithmetic.
pub const MAX_REMOTE_SINCE_DAYS: u32 = 365_000;
pub const MAX_CONTEXT_TOKENS: usize = 32 * 1024;
pub const MAX_CONTEXT_CHUNK_TOKENS: usize = 8 * 1024;

#[derive(Debug, Clone)]
pub struct CallerIdentity {
    pub instance_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CaptureProvenance {
    pub uri: Option<String>,
    pub label: Option<String>,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RemoteCaptureRequest {
    pub request_id: String,
    pub content: String,
    pub title: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    pub provenance: Option<CaptureProvenance>,
}

/// Stable, vector-free authoritative capture outcome stored with its write.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct SavedRecord {
    pub id: String,
    pub revision: String,
    pub title: Option<String>,
    pub content: String,
    pub tags: Vec<String>,
    pub created_at: String,
    pub updated_at: String,
    pub provenance: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RemoteCaptureResponse {
    pub request_id: String,
    pub replayed: bool,
    pub record: SavedRecord,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BuildContextRequest {
    pub query: String,
    pub scope: Scope,
    pub graph: GraphPolicy,
    pub since_days: Option<u32>,
    pub source_uri: Option<String>,
    pub entity_filter: Option<String>,
    pub max_chunks: Option<usize>,
    pub max_total_tokens: Option<usize>,
    pub max_chunk_tokens: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ContextChunk {
    pub citation: usize,
    pub id: String,
    pub hit_type: String,
    pub title: Option<String>,
    pub snippet: String,
    pub created_at: Option<String>,
    pub source_uri: Option<String>,
    pub score: f32,
    pub conversation_uuid: Option<String>,
    pub message_index: Option<i64>,
    pub role: Option<String>,
    pub rendered_tokens: usize,
    pub approx_tokens: usize,
    pub truncated: bool,
    pub selected_span_start: Option<usize>,
    pub selected_span_end: Option<usize>,
    pub graph: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ContextResponse {
    pub query: String,
    pub scope: Scope,
    pub chunks: Vec<ContextChunk>,
    pub total_tokens: usize,
    pub diagnostics: serde_json::Value,
    pub rendered_context: String,
}
