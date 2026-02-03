//! Chat export types for importing conversations from Claude Desktop
//!
//! This module handles parsing the official Claude Desktop export format.
//! The export consists of a `conversations.json` file containing an array
//! of conversation objects with their chat messages.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Role of the message sender in Claude Desktop format
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

/// A single message in a Claude Desktop chat conversation
///
/// Claude Desktop exports messages with both a `text` field (plain text)
/// and a `content` array (structured content blocks). We use `text` for
/// the message content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Unique identifier for this message
    #[serde(default)]
    pub uuid: Option<String>,

    /// The message text content
    pub text: String,

    /// The sender role: "human" or "assistant"
    pub sender: MessageRole,

    /// When the message was created
    #[serde(default)]
    pub created_at: Option<DateTime<Utc>>,

    /// When the message was last updated
    #[serde(default)]
    pub updated_at: Option<DateTime<Utc>>,

    /// Structured content blocks (preserved but not actively used)
    #[serde(default)]
    pub content: serde_json::Value,

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
    #[serde(default)]
    pub chat_messages: Vec<ChatMessage>,
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
        self.conversations
            .iter()
            .map(|c| c.chat_messages.len())
            .sum()
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
                .filter(|c| !c.chat_messages.is_empty())
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
        if let Some(msg) = self
            .chat_messages
            .iter()
            .find(|m| m.sender == MessageRole::Human)
        {
            let first_line = msg.text.lines().next().unwrap_or(&msg.text);
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
        for msg in &self.chat_messages {
            let role_label = match msg.sender {
                MessageRole::Human => "**Human**",
                MessageRole::Assistant => "**Assistant**",
                MessageRole::System => "**System**",
            };

            md.push_str(&format!("{}: {}\n\n", role_label, msg.text));
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
                        "content": [],
                        "attachments": [],
                        "files": []
                    },
                    {
                        "uuid": "019bfac3-7bb7-7288-87b8-7b9c1fc5bc86",
                        "text": "I'm doing well, thank you!",
                        "sender": "assistant",
                        "created_at": "2026-01-26T14:44:44.257542Z",
                        "updated_at": "2026-01-26T14:44:44.257542Z",
                        "content": [],
                        "attachments": [],
                        "files": []
                    }
                ]
            }
        ]"#;

        let export = ChatExport::from_json(json).unwrap();
        assert_eq!(export.conversations.len(), 1);
        assert_eq!(export.conversations[0].chat_messages.len(), 2);
        assert_eq!(export.conversations[0].name, "Test Conversation");
        assert_eq!(
            export.conversations[0].chat_messages[0].text,
            "Hello, how are you?"
        );
        assert_eq!(
            export.conversations[0].chat_messages[0].sender,
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
        assert_eq!(export.conversations[0].chat_messages.len(), 0);

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
            chat_messages: vec![],
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
                    chat_messages: vec![
                        ChatMessage {
                            uuid: Some("m1".into()),
                            text: "Hello".into(),
                            sender: MessageRole::Human,
                            created_at: None,
                            updated_at: None,
                            content: serde_json::Value::Null,
                            attachments: vec![],
                            files: vec![],
                        },
                        ChatMessage {
                            uuid: Some("m2".into()),
                            text: "Hi".into(),
                            sender: MessageRole::Assistant,
                            created_at: None,
                            updated_at: None,
                            content: serde_json::Value::Null,
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
                    chat_messages: vec![ChatMessage {
                        uuid: Some("m3".into()),
                        text: "Test".into(),
                        sender: MessageRole::Human,
                        created_at: None,
                        updated_at: None,
                        content: serde_json::Value::Null,
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
}
