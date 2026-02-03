//! Chat export types for importing conversations from Claude Desktop and other chat clients

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Role of the message sender
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    /// Human/user message
    Human,
    /// Assistant/AI response
    Assistant,
    /// System prompt (if exported)
    System,
}

/// A single message in a chat conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// The role of who sent this message
    #[serde(alias = "sender", alias = "role")]
    pub role: MessageRole,

    /// The message content
    #[serde(alias = "text")]
    pub content: String,

    /// When the message was sent (if available)
    #[serde(default, alias = "created_at", alias = "timestamp")]
    pub sent_at: Option<DateTime<Utc>>,

    /// Additional metadata
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// A complete chat conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatConversation {
    /// Unique identifier for this conversation (e.g., UUID)
    #[serde(default, alias = "id", alias = "conversation_id")]
    pub uuid: Option<String>,

    /// Human-readable title for the conversation
    #[serde(default, alias = "title", alias = "name")]
    pub name: Option<String>,

    /// When the conversation was created
    #[serde(default, alias = "created", alias = "created_at")]
    pub created_at: Option<DateTime<Utc>>,

    /// When the conversation was last updated
    #[serde(default, alias = "updated", alias = "updated_at")]
    pub updated_at: Option<DateTime<Utc>>,

    /// The messages in this conversation
    #[serde(default, alias = "messages", alias = "chat_messages")]
    pub messages: Vec<ChatMessage>,

    /// Additional metadata (model used, etc.)
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// A collection of chat conversations (full export)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatExport {
    /// Version of the export format
    #[serde(default)]
    pub version: Option<String>,

    /// The conversations in this export
    #[serde(alias = "conversations", alias = "chats")]
    pub conversations: Vec<ChatConversation>,

    /// When this export was created
    #[serde(default)]
    pub exported_at: Option<DateTime<Utc>>,

    /// Additional metadata about the export
    #[serde(default)]
    pub metadata: serde_json::Value,
}

impl ChatExport {
    /// Parse a chat export from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        // Try parsing as a full export first
        if let Ok(export) = serde_json::from_str::<ChatExport>(json) {
            return Ok(export);
        }

        // Try parsing as a single conversation
        if let Ok(conversation) = serde_json::from_str::<ChatConversation>(json) {
            return Ok(ChatExport {
                version: Some("1.0".into()),
                conversations: vec![conversation],
                exported_at: Some(Utc::now()),
                metadata: serde_json::Value::Null,
            });
        }

        // Try parsing as an array of conversations
        if let Ok(conversations) = serde_json::from_str::<Vec<ChatConversation>>(json) {
            return Ok(ChatExport {
                version: Some("1.0".into()),
                conversations,
                exported_at: Some(Utc::now()),
                metadata: serde_json::Value::Null,
            });
        }

        // Fall back to standard parsing (will give a proper error)
        serde_json::from_str(json)
    }

    /// Get total message count across all conversations
    pub fn total_messages(&self) -> usize {
        self.conversations.iter().map(|c| c.messages.len()).sum()
    }

    /// Get total conversation count
    pub fn conversation_count(&self) -> usize {
        self.conversations.len()
    }
}

impl ChatConversation {
    /// Get a title for this conversation (uses first message if no title set)
    pub fn display_title(&self) -> String {
        if let Some(ref name) = self.name {
            return name.clone();
        }

        // Try to generate from first human message
        if let Some(msg) = self.messages.iter().find(|m| m.role == MessageRole::Human) {
            let content = &msg.content;
            // Take first 50 characters of first line
            let first_line = content.lines().next().unwrap_or(content);
            if first_line.len() > 50 {
                format!("{}...", &first_line[..50])
            } else {
                first_line.to_string()
            }
        } else {
            "Untitled Conversation".to_string()
        }
    }

    /// Convert the conversation to a formatted markdown string
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        // Add title
        md.push_str(&format!("# {}\n\n", self.display_title()));

        // Add metadata if available
        if let Some(ref created) = self.created_at {
            md.push_str(&format!("*Created: {}*\n\n", created.format("%Y-%m-%d %H:%M")));
        }

        // Add messages
        for msg in &self.messages {
            let role_label = match msg.role {
                MessageRole::Human => "**Human**",
                MessageRole::Assistant => "**Assistant**",
                MessageRole::System => "**System**",
            };

            md.push_str(&format!("{}: {}\n\n", role_label, msg.content));
        }

        md
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_single_conversation() {
        let json = r#"{
            "name": "Test Chat",
            "messages": [
                {"role": "human", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        }"#;

        let export = ChatExport::from_json(json).unwrap();
        assert_eq!(export.conversations.len(), 1);
        assert_eq!(export.conversations[0].messages.len(), 2);
    }

    #[test]
    fn test_parse_conversation_array() {
        let json = r#"[
            {
                "name": "Chat 1",
                "messages": [{"role": "human", "content": "Hello"}]
            },
            {
                "name": "Chat 2",
                "messages": [{"role": "human", "content": "World"}]
            }
        ]"#;

        let export = ChatExport::from_json(json).unwrap();
        assert_eq!(export.conversations.len(), 2);
    }

    #[test]
    fn test_display_title() {
        let conv = ChatConversation {
            uuid: None,
            name: None,
            created_at: None,
            updated_at: None,
            messages: vec![ChatMessage {
                role: MessageRole::Human,
                content: "What is the meaning of life?".into(),
                sent_at: None,
                metadata: serde_json::Value::Null,
            }],
            metadata: serde_json::Value::Null,
        };

        assert_eq!(conv.display_title(), "What is the meaning of life?");
    }
}
