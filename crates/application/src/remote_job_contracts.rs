//! Bounded uploaded content and server-owned job contracts.
use crate::CaptureProvenance;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub const MAX_REMOTE_UPLOAD_BYTES: usize = 65_536;
pub const MAX_REMOTE_UPLOAD_CHUNKS: usize = 200;
pub const MAX_REMOTE_JOB_LIST: usize = 100;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct UploadSourceRequest {
    /// 128 Unicode characters and 256 UTF-8 bytes; byte limits are checked semantically.
    #[schemars(length(min = 1, max = 128))]
    pub request_id: String,
    /// Opaque client document identity: 256 characters/512 UTF-8 bytes; never a server path or URL to fetch.
    #[schemars(length(min = 1, max = 256))]
    pub document_key: String,
    /// Supplied Markdown: at most 65536 UTF-8 bytes, independently of character count.
    #[schemars(length(min = 1, max = 65536))]
    pub content: String,
    #[schemars(length(max = 512))]
    pub title: Option<String>,
    pub provenance: Option<CaptureProvenance>,
    #[serde(default)]
    pub extract_entities: bool,
    /// Register matching metadata only; never replace a source generation.
    #[serde(default)]
    pub preserve_unchanged: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct UploadAdmission {
    pub request_id: String,
    pub job_id: String,
    pub source_id: String,
    pub source_uri: String,
    pub replayed: bool,
}

/// Explicit retirement of one uploaded source in a client-registered collection.
/// The authenticated owner and stored provenance independently constrain scope.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DeleteUploadedSourceRequest {
    #[schemars(length(min = 1, max = 128))]
    pub request_id: String,
    #[schemars(length(min = 1, max = 512))]
    pub id: String,
    #[schemars(length(min = 64, max = 64))]
    pub revision: String,
    #[schemars(length(min = 1, max = 64))]
    pub collection_id: String,
    pub confirmed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct RemoteJobStatus {
    pub id: String,
    pub job_type: String,
    pub instance_id: String,
    pub status: String,
    pub phase: String,
    pub cancellation_requested: bool,
    pub source_id: String,
    pub generation: Option<u64>,
    pub total: u64,
    pub completed: u64,
    pub failed: u64,
    pub checkpoint: Option<String>,
    pub result: Option<serde_json::Value>,
    pub error_code: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RemoteJobList {
    pub jobs: Vec<RemoteJobStatus>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct UploadedSource {
    pub id: String,
    pub uri: String,
    pub title: Option<String>,
    /// Latest supplied input, explicitly identified by its attempted generation.
    pub content: String,
    pub content_hash: Option<String>,
    pub generation: u64,
    pub successful_generation: u64,
    pub status: String,
    pub instance_id: String,
    pub document_key: String,
    pub provenance: serde_json::Value,
    pub extract_entities: bool,
    /// Opaque processing snapshot identity, without provider URLs/settings.
    pub processing_policy_sha256: String,
    pub processing_policy_current: bool,
    pub revision: String,
}

/// Internal worker authority, never an MCP input or credential.
#[derive(Debug, Clone)]
pub struct RemoteJobExecution {
    pub job_id: String,
    pub instance_id: String,
    pub service_epoch: String,
    pub worker_token: String,
}
