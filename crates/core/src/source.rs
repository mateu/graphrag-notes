//! Source types - where notes come from

use chrono::{DateTime, Utc};
use surrealdb::RecordId;
use serde::{Deserialize, Serialize};

/// The type of source
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
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

impl Default for SourceType {
    fn default() -> Self {
        Self::Manual
    }
}

/// A source of notes/content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    /// Unique identifier
    pub id: Option<RecordId>,
    
    /// Type of source
    pub source_type: SourceType,
    
    /// Human-readable title
    pub title: Option<String>,
    
    /// URL or file path (if applicable)
    pub uri: Option<String>,
    
    /// Raw content (for reference)
    pub content: Option<String>,
    
    /// Additional metadata
    #[serde(default)]
    pub metadata: serde_json::Value,
    
    /// When this source was added
    #[serde(skip_serializing)]
    pub created_at: DateTime<Utc>,
}

impl Source {
    /// Create a new manual source
    pub fn manual() -> Self {
        Self {
            id: None,
            source_type: SourceType::Manual,
            title: None,
            uri: None,
            content: None,
            metadata: serde_json::Value::Null,
            created_at: Utc::now(),
        }
    }
    
    /// Create a source from a file
    pub fn from_file(path: impl Into<String>, source_type: SourceType) -> Self {
        let path = path.into();
        Self {
            id: None,
            source_type,
            title: Some(path.clone()),
            uri: Some(path),
            content: None,
            metadata: serde_json::Value::Null,
            created_at: Utc::now(),
        }
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
            content: None,
            metadata: serde_json::Value::Null,
            created_at: Utc::now(),
        }
    }
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
        let source = Source::from_file("/path/to/file.md", SourceType::Markdown);
        
        assert_eq!(source.source_type, SourceType::Markdown);
        assert_eq!(source.uri, Some("/path/to/file.md".into()));
    }
}
