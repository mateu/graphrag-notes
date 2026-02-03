//! Chat export types for importing conversations from Claude Desktop
//!
//! This module handles parsing the official Claude Desktop export format.
//! The export consists of a `conversations.json` file containing an array
//! of conversation objects with their chat messages.
//!
//! Field mapping from Claude Desktop format to internal names:
//! - `chat_messages` → `messages`
//! - `sender` → `role`
//! - `text` → `content`

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Role of the message sender
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// Human/user message
    Human,
    /// Assistant/AI response
    Assistant,
    /// System prompt (if present)
    System,
}

/// A single message in a chat conversation
///
/// Claude Desktop exports messages with `sender` and `text` fields,
/// which are mapped to `role` and `content` for internal consistency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Unique identifier for this message
    #[serde(default)]
    pub uuid: Option<String>,

    /// The sender role: "human" or "assistant"
    /// Maps from Claude Desktop's `sender` field
    #[serde(alias = "sender")]
    pub role: MessageRole,

    /// The message text content
    /// Maps from Claude Desktop's `text` field
    #[serde(alias = "text")]
    pub content: String,

    /// When the message was created
    #[serde(default)]
    pub created_at: Option<DateTime<Utc>>,

    /// When the message was last updated
    #[serde(default)]
    pub updated_at: Option<DateTime<Utc>>,

    /// File attachments
    #[serde(default)]
    pub attachments: Vec<serde_json::Value>,

    /// Associated files
    #[serde(default)]
    pub files: Vec<serde_json::Value>,
}

/// Account information from Claude Desktop export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Account {
    pub uuid: String,
}

/// A complete chat conversation from Claude Desktop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatConversation {
    /// Unique identifier for this conversation
    pub uuid: String,

    /// Human-readable title for the conversation
    pub name: String,

    /// AI-generated summary of the conversation (may be empty)
    #[serde(default)]
    pub summary: String,

    /// When the conversation was created
    pub created_at: DateTime<Utc>,

    /// When the conversation was last updated
    pub updated_at: DateTime<Utc>,

    /// Account that owns this conversation
    #[serde(default)]
    pub account: Option<Account>,

    /// The messages in this conversation
    /// Maps from Claude Desktop's `chat_messages` field
    #[serde(default, alias = "chat_messages")]
    pub messages: Vec<ChatMessage>,
}

/// A collection of chat conversations (Claude Desktop export)
///
/// This represents the contents of a `conversations.json` export file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatExport {
    /// The conversations in this export
    pub conversations: Vec<ChatConversation>,

    /// When this export was created (added during parsing)
    #[serde(default)]
    pub exported_at: Option<DateTime<Utc>>,
}

impl ChatExport {
    /// Parse a Claude Desktop chat export from JSON
    ///
    /// Handles the standard format: an array of conversation objects.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        // Claude Desktop exports conversations.json as a direct array
        if let Ok(conversations) = serde_json::from_str::<Vec<ChatConversation>>(json) {
            return Ok(ChatExport {
                conversations,
                exported_at: Some(Utc::now()),
            });
        }

        // Also try parsing as a wrapped object (for flexibility)
        if let Ok(export) = serde_json::from_str::<ChatExport>(json) {
            return Ok(export);
        }

        // Try parsing as a single conversation
        if let Ok(conversation) = serde_json::from_str::<ChatConversation>(json) {
            return Ok(ChatExport {
                conversations: vec![conversation],
                exported_at: Some(Utc::now()),
            });
        }

        // Return error from array parsing (most likely format)
        serde_json::from_str::<Vec<ChatConversation>>(json).map(|conversations| ChatExport {
            conversations,
            exported_at: Some(Utc::now()),
        })
    }

    /// Get total message count across all conversations
    pub fn total_messages(&self) -> usize {
        self.conversations.iter().map(|c| c.messages.len()).sum()
    }

    /// Get total conversation count
    pub fn conversation_count(&self) -> usize {
        self.conversations.len()
    }

    /// Filter to only conversations that have messages
    pub fn with_messages_only(self) -> Self {
        ChatExport {
            conversations: self
                .conversations
                .into_iter()
                .filter(|c| !c.messages.is_empty())
                .collect(),
            exported_at: self.exported_at,
        }
    }
}

impl ChatConversation {
    /// Get a display title for this conversation
    pub fn display_title(&self) -> String {
        if !self.name.is_empty() {
            return self.name.clone();
        }

        // Fall back to first human message
        if let Some(msg) = self.messages.iter().find(|m| m.role == MessageRole::Human) {
            let first_line = msg.content.lines().next().unwrap_or(&msg.content);
            if first_line.len() > 50 {
                format!("{}...", &first_line[..50])
            } else {
                first_line.to_string()
            }
        } else {
            "Untitled Conversation".to_string()
        }
    }

    /// Check if conversation has a summary
    pub fn has_summary(&self) -> bool {
        !self.summary.is_empty()
    }

    /// Convert the conversation to a formatted markdown string
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        // Add title
        md.push_str(&format!("# {}\n\n", self.display_title()));

        // Add metadata
        md.push_str(&format!(
            "*Created: {}*\n\n",
            self.created_at.format("%Y-%m-%d %H:%M")
        ));

        // Add summary if present
        if self.has_summary() {
            md.push_str(&format!("## Summary\n\n{}\n\n", self.summary));
        }

        // Add messages
        md.push_str("## Conversation\n\n");
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
    fn test_parse_claude_desktop_format() {
        // This matches the actual Claude Desktop export format
        // Note: uses `sender`, `text`, `chat_messages` which get mapped to internal names
        let json = r#"[
            {
                "uuid": "d8d25da3-5645-43b9-87dd-abf0a3d703ae",
                "name": "Test Conversation",
                "summary": "",
                "created_at": "2026-01-22T15:56:10.335839Z",
                "updated_at": "2026-01-22T15:56:17.944144Z",
                "account": {"uuid": "3baa4d0e-cc94-422c-9489-4b93a828df36"},
                "chat_messages": [
                    {
                        "uuid": "019bfac3-7bb7-7288-87b8-7b9bbadac60c",
                        "text": "Hello, how are you?",
                        "sender": "human",
                        "created_at": "2026-01-26T14:44:33.978272Z",
                        "updated_at": "2026-01-26T14:44:33.978272Z",
                        "attachments": [],
                        "files": []
                    },
                    {
                        "uuid": "019bfac3-7bb7-7288-87b8-7b9c1fc5bc86",
                        "text": "I'm doing well, thank you!",
                        "sender": "assistant",
                        "created_at": "2026-01-26T14:44:44.257542Z",
                        "updated_at": "2026-01-26T14:44:44.257542Z",
                        "attachments": [],
                        "files": []
                    }
                ]
            }
        ]"#;

        let export = ChatExport::from_json(json).unwrap();
        assert_eq!(export.conversations.len(), 1);
        assert_eq!(export.conversations[0].messages.len(), 2);
        assert_eq!(export.conversations[0].name, "Test Conversation");

        // Verify field mapping worked
        assert_eq!(
            export.conversations[0].messages[0].content,
            "Hello, how are you?"
        );
        assert_eq!(
            export.conversations[0].messages[0].role,
            MessageRole::Human
        );
    }

    #[test]
    fn test_parse_empty_conversations() {
        let json = r#"[
            {
                "uuid": "test-uuid",
                "name": "Empty Chat",
                "summary": "",
                "created_at": "2026-01-22T15:56:10.335839Z",
                "updated_at": "2026-01-22T15:56:17.944144Z",
                "chat_messages": []
            }
        ]"#;

        let export = ChatExport::from_json(json).unwrap();
        assert_eq!(export.conversations.len(), 1);
        assert_eq!(export.conversations[0].messages.len(), 0);

        // Filter to only conversations with messages
        let filtered = export.with_messages_only();
        assert_eq!(filtered.conversations.len(), 0);
    }

    #[test]
    fn test_display_title() {
        let conv = ChatConversation {
            uuid: "test".into(),
            name: "My Chat Title".into(),
            summary: String::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            account: None,
            messages: vec![],
        };

        assert_eq!(conv.display_title(), "My Chat Title");
    }

    #[test]
    fn test_message_count() {
        let export = ChatExport {
            conversations: vec![
                ChatConversation {
                    uuid: "1".into(),
                    name: "Chat 1".into(),
                    summary: String::new(),
                    created_at: Utc::now(),
                    updated_at: Utc::now(),
                    account: None,
                    messages: vec![
                        ChatMessage {
                            uuid: Some("m1".into()),
                            content: "Hello".into(),
                            role: MessageRole::Human,
                            created_at: None,
                            updated_at: None,
                            attachments: vec![],
                            files: vec![],
                        },
                        ChatMessage {
                            uuid: Some("m2".into()),
                            content: "Hi".into(),
                            role: MessageRole::Assistant,
                            created_at: None,
                            updated_at: None,
                            attachments: vec![],
                            files: vec![],
                        },
                    ],
                },
                ChatConversation {
                    uuid: "2".into(),
                    name: "Chat 2".into(),
                    summary: String::new(),
                    created_at: Utc::now(),
                    updated_at: Utc::now(),
                    account: None,
                    messages: vec![ChatMessage {
                        uuid: Some("m3".into()),
                        content: "Test".into(),
                        role: MessageRole::Human,
                        created_at: None,
                        updated_at: None,
                        attachments: vec![],
                        files: vec![],
                    }],
                },
            ],
            exported_at: None,
        };

        assert_eq!(export.conversation_count(), 2);
        assert_eq!(export.total_messages(), 3);
    }

    #[test]
    fn test_field_aliases() {
        // Test that both internal names and Claude Desktop names work
        let internal_format = r#"{
            "uuid": "test",
            "name": "Test",
            "summary": "",
            "created_at": "2026-01-22T15:56:10.335839Z",
            "updated_at": "2026-01-22T15:56:17.944144Z",
            "messages": [
                {"role": "human", "content": "Hello"}
            ]
        }"#;

        let claude_format = r#"{
            "uuid": "test",
            "name": "Test",
            "summary": "",
            "created_at": "2026-01-22T15:56:10.335839Z",
            "updated_at": "2026-01-22T15:56:17.944144Z",
            "chat_messages": [
                {"sender": "human", "text": "Hello"}
            ]
        }"#;

        let internal = ChatExport::from_json(internal_format).unwrap();
        let claude = ChatExport::from_json(claude_format).unwrap();

        assert_eq!(internal.conversations[0].messages.len(), 1);
        assert_eq!(claude.conversations[0].messages.len(), 1);
        assert_eq!(
            internal.conversations[0].messages[0].content,
            claude.conversations[0].messages[0].content
        );
    }
}
