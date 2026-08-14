//! Source types - where notes come from

use crate::{CoreError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Component, Path, PathBuf};
use surrealdb::types::RecordId;
use surrealdb_types::SurrealValue;

/// The type of source
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, SurrealValue)]
#[serde(rename_all = "snake_case")]
#[surreal(crate = "surrealdb_types")]
#[surreal(untagged, lowercase)]
pub enum SourceType {
    /// User-typed note
    Manual,
    /// Markdown file
    Markdown,
    /// Plain text file
    Text,
    /// URL/webpage
    Url,
    /// PDF document (future)
    Pdf,
    /// Voice memo (future)
    Voice,
    /// Chat export (e.g., Claude Desktop)
    ChatExport,
}

/// State of a source's most recent file ingestion attempt.
///
/// `failed` never invalidates a previous successful generation. Consumers must
/// use `successful_generation` rather than assuming `generation` is readable.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, SurrealValue)]
#[serde(rename_all = "snake_case")]
#[surreal(crate = "surrealdb_types")]
#[surreal(untagged, lowercase)]
pub enum SourceIngestionStatus {
    Pending,
    Ready,
    Failed,
}

impl Default for SourceIngestionStatus {
    fn default() -> Self {
        Self::Ready
    }
}

impl Default for SourceType {
    fn default() -> Self {
        Self::Manual
    }
}

/// A source of notes/content
#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct Source {
    /// Unique identifier
    pub id: Option<RecordId>,

    /// Type of source
    pub source_type: SourceType,

    /// Human-readable title
    pub title: Option<String>,

    /// URL or file path (if applicable)
    pub uri: Option<String>,

    /// Canonical identity for a non-manual source. File paths use a `file://`
    /// URI built from a lexical absolute path (or the canonical path when the
    /// file exists).
    #[serde(default)]
    pub normalized_uri: Option<String>,

    /// Raw content (for reference)
    pub content: Option<String>,

    /// SHA-256 of UTF-8 content after CRLF/CR line endings are normalized to LF.
    #[serde(default)]
    pub content_hash: Option<String>,

    /// Monotonic ingestion attempt number for this source.
    #[serde(default)]
    pub generation: u64,

    /// Most recent generation whose derived records are safe to search.
    #[serde(default)]
    pub successful_generation: u64,

    #[serde(default)]
    pub status: SourceIngestionStatus,

    #[serde(default)]
    pub last_error: Option<String>,

    /// Additional metadata
    #[serde(default)]
    pub metadata: serde_json::Value,

    /// When this source was added
    #[serde(skip_serializing)]
    pub created_at: DateTime<Utc>,

    #[serde(default = "Utc::now", skip_serializing)]
    pub updated_at: DateTime<Utc>,

    #[serde(default, skip_serializing)]
    pub last_ingested_at: Option<DateTime<Utc>>,
}

impl Source {
    /// Create a new manual source
    pub fn manual() -> Self {
        Self {
            id: None,
            source_type: SourceType::Manual,
            title: None,
            uri: None,
            normalized_uri: None,
            content: None,
            content_hash: None,
            generation: 0,
            successful_generation: 0,
            status: SourceIngestionStatus::Ready,
            last_error: None,
            metadata: serde_json::json!({}),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_ingested_at: None,
        }
    }

    /// Create a source from a file
    pub fn from_file(path: impl Into<String>, source_type: SourceType) -> Result<Self> {
        let path = path.into();
        let normalized_uri = normalize_file_uri(&path)?;
        let now = Utc::now();
        Ok(Self {
            id: None,
            source_type,
            title: Some(path.clone()),
            uri: Some(normalized_uri.clone()),
            normalized_uri: Some(normalized_uri),
            content: None,
            content_hash: None,
            generation: 0,
            successful_generation: 0,
            status: SourceIngestionStatus::Pending,
            last_error: None,
            metadata: serde_json::json!({}),
            created_at: now,
            updated_at: now,
            last_ingested_at: None,
        })
    }

    /// Builder: set title
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Builder: set content
    pub fn with_content(mut self, content: impl Into<String>) -> Self {
        self.content = Some(content.into());
        self
    }

    /// Builder: set metadata
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }

    /// Create a source from a chat export
    pub fn chat_export(title: impl Into<String>, uri: Option<String>) -> Self {
        Self {
            id: None,
            source_type: SourceType::ChatExport,
            title: Some(title.into()),
            uri,
            normalized_uri: None,
            content: None,
            content_hash: None,
            generation: 0,
            successful_generation: 0,
            status: SourceIngestionStatus::Ready,
            last_error: None,
            metadata: serde_json::json!({}),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_ingested_at: None,
        }
    }
}

/// Return an identity URI for a local file path without requiring the file to
/// exist. Existing paths are canonicalized (resolving symlinks); otherwise a
/// lexical absolute path with `.` and `..` removed is used. This keeps imports
/// stable across repeated invocations and test environments.
pub fn normalize_file_uri(path: impl AsRef<Path>) -> Result<String> {
    let path = path.as_ref();
    let normalized = std::fs::canonicalize(path).unwrap_or_else(|_| lexical_absolute_path(path));
    let display = normalized.to_str().ok_or_else(|| {
        CoreError::Validation("source path must be valid UTF-8 after canonicalization".into())
    })?;
    let display = if cfg!(windows) {
        normalize_windows_path(display)
    } else {
        // A verbatim Windows path can be supplied in tests or through a
        // cross-platform caller. Ordinary Unix backslashes are valid filename
        // characters and must remain part of the source identity.
        normalize_windows_verbatim_path(display)
    };
    Ok(file_uri_from_normalized_display(&display))
}

fn file_uri_from_normalized_display(display: &str) -> String {
    if display.starts_with("//") {
        // UNC paths encode their host in the URI authority. Keeping this form
        // also lets Windows reconstruct a UNC path during `sources reimport`.
        format!("file:{display}")
    } else if display.starts_with('/') {
        format!("file://{display}")
    } else {
        format!("file:///{display}")
    }
}

/// Convert Windows extended-length paths returned by `canonicalize` into the
/// ordinary drive/UNC forms used for stable file URIs. Kept platform-neutral
/// so the behavior can be regression-tested on every host.
pub fn normalize_windows_verbatim_path(path: &str) -> String {
    if let Some(unc) = path
        .strip_prefix(r"\\?\UNC\")
        .or_else(|| path.strip_prefix("//?/UNC/"))
    {
        return format!("//{}", unc.replace('\\', "/"));
    }
    if let Some(disk) = path
        .strip_prefix(r"\\?\")
        .or_else(|| path.strip_prefix("//?/"))
    {
        return disk.replace('\\', "/");
    }
    path.to_string()
}

fn normalize_windows_path(path: &str) -> String {
    let normalized = normalize_windows_verbatim_path(path);
    if normalized == path {
        path.replace('\\', "/")
    } else {
        normalized
    }
}

fn lexical_absolute_path(path: &Path) -> PathBuf {
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    };
    let mut result = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                result.pop();
            }
            other => result.push(other.as_os_str()),
        }
    }
    result
}

/// Hash UTF-8 source content deterministically. Line endings are normalized so
/// the same document has the same identity on Windows and Unix.
pub fn normalized_content_hash(content: &str) -> String {
    let normalized = content.replace("\r\n", "\n").replace('\r', "\n");
    format!("sha256:{:x}", Sha256::digest(normalized.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manual_source() {
        let source = Source::manual().with_title("My Notes");

        assert_eq!(source.source_type, SourceType::Manual);
        assert_eq!(source.title, Some("My Notes".into()));
    }

    #[test]
    fn test_file_source() {
        let source = Source::from_file("/path/to/file.md", SourceType::Markdown).unwrap();

        assert_eq!(source.source_type, SourceType::Markdown);
        assert_eq!(source.uri, Some("file:///path/to/file.md".into()));
        assert_eq!(source.normalized_uri, source.uri);
    }

    #[test]
    fn file_uri_is_lexically_stable_when_a_path_does_not_exist() {
        let uri = normalize_file_uri("./fixture/../fixture/notes.md").unwrap();
        assert!(uri.starts_with("file://"));
        assert!(uri.ends_with("/fixture/notes.md"));
    }

    #[test]
    fn content_hash_normalizes_line_endings() {
        assert_eq!(
            normalized_content_hash("one\r\ntwo\rthree\n"),
            normalized_content_hash("one\ntwo\nthree\n")
        );
    }

    #[test]
    fn windows_verbatim_paths_normalize_before_uri_encoding() {
        assert_eq!(
            normalize_windows_verbatim_path(r"\\?\C:\notes\alpha.md"),
            "C:/notes/alpha.md"
        );
        assert_eq!(
            normalize_windows_verbatim_path(r"\\?\UNC\server\share\alpha.md"),
            "//server/share/alpha.md"
        );
        assert_eq!(
            file_uri_from_normalized_display(&normalize_windows_verbatim_path(
                r"\\?\C:\notes\alpha.md"
            )),
            "file:///C:/notes/alpha.md"
        );
        assert_eq!(
            file_uri_from_normalized_display(&normalize_windows_verbatim_path(
                r"\\?\UNC\server\share\alpha.md"
            )),
            "file://server/share/alpha.md"
        );
    }

    #[cfg(unix)]
    #[test]
    fn unix_backslashes_remain_part_of_the_source_identity() {
        let uri = normalize_file_uri("notes\\with\\backslashes.md").unwrap();
        assert!(uri.ends_with("/notes\\with\\backslashes.md"));
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
    #[test]
    fn non_utf8_canonical_source_target_is_rejected() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let target = directory
            .path()
            .join(OsString::from_vec(b"non-utf8-\xff.md".to_vec()));
        std::fs::write(&target, "content").unwrap();
        let link = directory.path().join("utf8-link.md");
        symlink(&target, &link).unwrap();

        let error = normalize_file_uri(&link).unwrap_err();
        assert!(error
            .to_string()
            .contains("source path must be valid UTF-8 after canonicalization"));
    }
}
