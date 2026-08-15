//! Versioned, portable logical backup format.
//!
//! This format deliberately describes application records rather than a
//! SurrealDB storage directory. It can therefore be verified before a fresh
//! database is touched and remains independent of RocksDB's binary format.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Magic string that identifies a GraphRAG Notes portable backup.
pub const PORTABLE_BACKUP_FORMAT: &str = "graphrag-notes-portable-backup";

/// The only portable backup format currently understood by this binary.
pub const PORTABLE_BACKUP_FORMAT_VERSION: u32 = 1;

/// One logical record written as a single JSONL line.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PortableRecord {
    /// Allow-listed application table that owns this record.
    pub table: String,
    /// A JSON-safe representation of the complete logical record, including
    /// its stable record ID and any references to other logical records.
    pub record: serde_json::Value,
}

/// Integrity information for the streaming JSONL payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PortablePayload {
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
    pub records: u64,
}

/// Identity required to safely interpret an exported vector payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PortableEmbeddingIdentity {
    pub provider: String,
    pub model: String,
    pub dimension: usize,
}

/// Human-readable and machine-stable description of a portable archive.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PortableBackupManifest {
    pub format: String,
    pub format_version: u32,
    pub created_at: DateTime<Utc>,
    /// The application schema used when the records were exported.
    pub schema_version: u32,
    /// Embeddings are intentionally excluded unless explicitly requested.
    pub includes_embeddings: bool,
    /// Present exactly when vectors are included, so restore can reject an
    /// incomplete or dimension-ambiguous vector archive before mutation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_identity: Option<PortableEmbeddingIdentity>,
    pub payload: PortablePayload,
    /// Counts are grouped by table for verification and operator visibility.
    pub record_counts: BTreeMap<String, u64>,
}

impl PortableBackupManifest {
    pub fn new(schema_version: u32, includes_embeddings: bool) -> Self {
        Self {
            format: PORTABLE_BACKUP_FORMAT.to_string(),
            format_version: PORTABLE_BACKUP_FORMAT_VERSION,
            created_at: Utc::now(),
            schema_version,
            includes_embeddings,
            embedding_identity: None,
            payload: PortablePayload {
                path: "records.jsonl".into(),
                sha256: String::new(),
                bytes: 0,
                records: 0,
            },
            record_counts: BTreeMap::new(),
        }
    }

    pub fn validate_format(&self) -> std::result::Result<(), String> {
        if self.format != PORTABLE_BACKUP_FORMAT {
            return Err(format!(
                "unsupported portable backup format {:?}",
                self.format
            ));
        }
        if self.format_version != PORTABLE_BACKUP_FORMAT_VERSION {
            return Err(format!(
                "unsupported portable backup format version {}; this binary supports {}",
                self.format_version, PORTABLE_BACKUP_FORMAT_VERSION
            ));
        }
        // v1 permits one payload with a plain filename. This prevents an
        // untrusted manifest from escaping its archive directory while also
        // allowing `export <path> --format jsonl` sidecar manifests.
        if self.payload.path.is_empty()
            || self.payload.path == "."
            || self.payload.path == ".."
            || self.payload.path.contains(['/', '\\'])
        {
            return Err("portable backup payload path is unsafe".into());
        }
        if self.payload.sha256.len() != 64
            || !self
                .payload
                .sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err("portable backup payload checksum is not SHA-256 hex".into());
        }
        if self.includes_embeddings != self.embedding_identity.is_some() {
            return Err(
                "portable backup must record embedding identity exactly when vectors are included"
                    .into(),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unknown_format_versions_and_unsafe_payloads() {
        let mut manifest = PortableBackupManifest::new(8, false);
        manifest.payload.sha256 = "0".repeat(64);
        assert!(manifest.validate_format().is_ok());

        manifest.format_version = 99;
        assert!(manifest.validate_format().is_err());
        manifest.format_version = PORTABLE_BACKUP_FORMAT_VERSION;
        manifest.payload.path = "../records.jsonl".into();
        assert!(manifest.validate_format().is_err());
    }
}
