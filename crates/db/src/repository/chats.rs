//! Conversation/message persistence and note-link ownership.
//!
//! Chat migration SQL and its deterministic link behavior remain verbatim.

use super::*;

impl Repository {
    pub async fn upsert_conversation(
        &self,
        conversation: &ChatConversation,
        source_uri: Option<String>,
        metadata: serde_json::Value,
        summary_embedding: Option<Vec<f32>>,
    ) -> Result<RecordId> {
        #[derive(Debug, Deserialize, SurrealValue)]
        struct ConversationIdRow {
            id: RecordId,
        }

        let account_uuid = conversation
            .account
            .as_ref()
            .map(|account| account.uuid.clone());
        let summary = if conversation.summary.is_empty() {
            None
        } else {
            Some(conversation.summary.clone())
        };

        let upserted: Option<ConversationIdRow> = self
            .db
            .query(
                r#"
                INSERT INTO conversation (
                    uuid, title, summary, source_uri, account_uuid, metadata, summary_embedding, created_at, updated_at, ingested_at
                )
                VALUES (
                    $uuid, $title, $summary, $source_uri, $account_uuid, $metadata, $summary_embedding, <datetime>$created_at, <datetime>$updated_at, time::now()
                )
                ON DUPLICATE KEY UPDATE
                    title = $title,
                    summary = $summary,
                    source_uri = $source_uri,
                    account_uuid = $account_uuid,
                    metadata = $metadata,
                    summary_embedding = $summary_embedding,
                    created_at = <datetime>$created_at,
                    updated_at = <datetime>$updated_at,
                    ingested_at = time::now()
            "#,
            )
            .bind(("uuid", conversation.uuid.clone()))
            .bind(("title", conversation.display_title()))
            .bind(("summary", summary))
            .bind(("source_uri", source_uri))
            .bind(("account_uuid", account_uuid))
            .bind(("metadata", metadata))
            .bind(("summary_embedding", summary_embedding))
            .bind(("created_at", conversation.created_at.to_rfc3339()))
            .bind(("updated_at", conversation.updated_at.to_rfc3339()))
            .await?
            .take(0)?;

        if let Some(row) = upserted {
            return Ok(row.id);
        }

        let fetched: Option<ConversationIdRow> = self
            .db
            .query("SELECT id FROM conversation WHERE uuid = $uuid LIMIT 1")
            .bind(("uuid", conversation.uuid.clone()))
            .await?
            .take(0)?;

        fetched
            .map(|row| row.id)
            .ok_or_else(|| DbError::CreateFailed("conversation".into()))
    }

    /// Upsert a message record from a chat export message.
    #[instrument(skip(self, message))]
    pub async fn upsert_message(
        &self,
        conversation_id: &RecordId,
        conversation_uuid: &str,
        index: usize,
        message: &ChatMessage,
        embedding: Option<Vec<f32>>,
    ) -> Result<RecordId> {
        #[derive(Debug, Deserialize, SurrealValue)]
        struct MessageIdRow {
            id: RecordId,
        }

        let message_uuid = message.uuid.clone();
        let message_key = message_uuid
            .clone()
            .unwrap_or_else(|| format!("{}:{}", conversation_uuid, index));

        let role = serde_json::to_string(&message.role)
            .unwrap_or_else(|_| "\"system\"".to_string())
            .trim_matches('"')
            .to_string();

        let content_blocks = message
            .content_blocks
            .as_array()
            .cloned()
            .unwrap_or_default();
        let attachments = message.attachments.clone();
        let files = message.files.clone();

        let upserted: Option<MessageIdRow> = self
            .db
            .query(
                r#"
                INSERT INTO message (
                    message_key, message_uuid, conversation_id, conversation_uuid, message_index, role,
                    content, embedding, content_blocks, attachments, files, has_files, created_at, updated_at, ingested_at
                )
                VALUES (
                    $message_key, $message_uuid, $conversation_id, $conversation_uuid, $message_index, $role,
                    $content, $embedding, $content_blocks, $attachments, $files, $has_files,
                    IF $created_at = NONE THEN NONE ELSE <datetime>$created_at END,
                    IF $updated_at = NONE THEN NONE ELSE <datetime>$updated_at END,
                    time::now()
                )
                ON DUPLICATE KEY UPDATE
                    message_uuid = $message_uuid,
                    conversation_id = $conversation_id,
                    conversation_uuid = $conversation_uuid,
                    message_index = $message_index,
                    role = $role,
                    content = $content,
                    embedding = $embedding,
                    content_blocks = $content_blocks,
                    attachments = $attachments,
                    files = $files,
                    has_files = $has_files,
                    created_at = IF $created_at = NONE THEN NONE ELSE <datetime>$created_at END,
                    updated_at = IF $updated_at = NONE THEN NONE ELSE <datetime>$updated_at END,
                    ingested_at = time::now()
            "#,
            )
            .bind(("message_key", message_key.clone()))
            .bind(("message_uuid", message_uuid))
            .bind(("conversation_id", conversation_id.clone()))
            .bind(("conversation_uuid", conversation_uuid.to_string()))
            .bind(("message_index", index as i64))
            .bind(("role", role))
            .bind(("content", message.content.clone()))
            .bind(("embedding", embedding))
            .bind(("content_blocks", content_blocks))
            .bind(("attachments", attachments))
            .bind(("files", files.clone()))
            .bind(("has_files", !files.is_empty()))
            .bind((
                "created_at",
                message.created_at.as_ref().map(|dt| dt.to_rfc3339()),
            ))
            .bind((
                "updated_at",
                message.updated_at.as_ref().map(|dt| dt.to_rfc3339()),
            ))
            .await?
            .take(0)?;

        if let Some(row) = upserted {
            return Ok(row.id);
        }

        let fetched: Option<MessageIdRow> = self
            .db
            .query("SELECT id FROM message WHERE message_key = $message_key LIMIT 1")
            .bind(("message_key", message_key))
            .await?
            .take(0)?;

        fetched
            .map(|row| row.id)
            .ok_or_else(|| DbError::CreateFailed("message".into()))
    }

    /// Link note provenance to a conversation.
    #[instrument(skip(self))]
    pub async fn link_note_to_conversation(
        &self,
        note_id: &RecordId,
        conversation_id: &RecordId,
    ) -> Result<bool> {
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        self.link_note_to_conversation_locked(note_id, conversation_id)
            .await
    }

    pub(crate) async fn link_note_to_conversation_locked(
        &self,
        note_id: &RecordId,
        conversation_id: &RecordId,
    ) -> Result<bool> {
        if !self.note_is_writable(note_id).await? {
            return Err(DbError::NotFound(
                "note endpoint".into(),
                "a conversation-provenance endpoint is hidden, failed, or no longer exists".into(),
            ));
        }
        #[derive(Deserialize, SurrealValue)]
        struct CountRow {
            count: Option<u64>,
        }

        let existing: Option<CountRow> = self
            .db
            .query(
                "SELECT count() FROM note_from_conversation WHERE in = $note_id AND out = $conversation_id GROUP ALL",
            )
            .bind(("note_id", note_id.clone()))
            .bind(("conversation_id", conversation_id.clone()))
            .await?
            .take(0)?;

        let count = existing.and_then(|row| row.count).unwrap_or(0);
        if count > 0 {
            return Ok(false);
        }

        self.db
            .query("CREATE note_from_conversation SET in = $note_id, out = $conversation_id")
            .bind(("note_id", note_id.clone()))
            .bind(("conversation_id", conversation_id.clone()))
            .await?;

        Ok(true)
    }

    /// Link note provenance to a message.
    #[instrument(skip(self))]
    pub async fn link_note_to_message(
        &self,
        note_id: &RecordId,
        message_id: &RecordId,
    ) -> Result<bool> {
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        self.link_note_to_message_locked(note_id, message_id).await
    }

    pub(crate) async fn link_note_to_message_locked(
        &self,
        note_id: &RecordId,
        message_id: &RecordId,
    ) -> Result<bool> {
        if !self.note_is_writable(note_id).await? {
            return Err(DbError::NotFound(
                "note endpoint".into(),
                "a message-provenance endpoint is hidden, failed, or no longer exists".into(),
            ));
        }
        #[derive(Deserialize, SurrealValue)]
        struct CountRow {
            count: Option<u64>,
        }

        let existing: Option<CountRow> = self
            .db
            .query(
                "SELECT count() FROM note_from_message WHERE in = $note_id AND out = $message_id GROUP ALL",
            )
            .bind(("note_id", note_id.clone()))
            .bind(("message_id", message_id.clone()))
            .await?
            .take(0)?;

        let count = existing.and_then(|row| row.count).unwrap_or(0);
        if count > 0 {
            return Ok(false);
        }

        self.db
            .query("CREATE note_from_message SET in = $note_id, out = $message_id")
            .bind(("note_id", note_id.clone()))
            .bind(("message_id", message_id.clone()))
            .await?;

        Ok(true)
    }

    /// Check whether a conversation already has any linked notes.
    #[instrument(skip(self))]
    pub async fn conversation_has_note_links(&self, conversation_id: &RecordId) -> Result<bool> {
        #[derive(Deserialize, SurrealValue)]
        struct CountRow {
            count: Option<u64>,
        }

        let existing: Option<CountRow> = self
            .db
            .query(
                "SELECT count() FROM note_from_conversation WHERE out = $conversation_id GROUP ALL",
            )
            .bind(("conversation_id", conversation_id.clone()))
            .await?
            .take(0)?;

        Ok(existing.and_then(|row| row.count).unwrap_or(0) > 0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct NoteEdgeRow {
    pub id: RecordId,
    pub edge_type: String,
    pub in_id: RecordId,
    pub out_id: RecordId,
    #[serde(default)]
    pub proposal_id: Option<RecordId>,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub provenance: Option<String>,
    #[serde(default)]
    pub is_manual: bool,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Local lexical entity match used as a graph-retrieval seed. `metadata` is
/// retained only to make alias evidence inspectable by higher layers.
#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct GraphEntityMatch {
    pub id: RecordId,
    pub name: String,
    pub canonical_name: String,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// A visible note directly mentioned by one query-matched entity. Retaining
/// both IDs prevents query-wide entity labels from being attached to unrelated
/// seeds when several matched entities are present.
#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct GraphEntityNoteSeed {
    pub note_id: RecordId,
    pub entity_id: RecordId,
}

/// Data required to persist an edge proposal. The repository canonicalizes
/// symmetric endpoint order before deriving the proposal's stable key.
#[derive(Debug, Clone)]
pub struct EdgeProposalDraft {
    pub from_id: RecordId,
    pub to_id: RecordId,
    pub edge_type: EdgeType,
    pub confidence: f32,
    pub reason: String,
    pub generator: String,
    pub generator_version: Option<String>,
    pub model: Option<String>,
}

#[derive(Debug, Deserialize, SurrealValue)]
pub(crate) struct ProposedEdgeRow {
    id: Option<RecordId>,
    dedupe_key: String,
    from_id: RecordId,
    to_id: RecordId,
    edge_type: String,
    confidence: f32,
    reason: String,
    generator: String,
    #[serde(default)]
    generator_version: Option<String>,
    #[serde(default)]
    model: Option<String>,
    status: String,
    created_at: chrono::DateTime<chrono::Utc>,
    updated_at: chrono::DateTime<chrono::Utc>,
    #[serde(default)]
    reviewed_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(default)]
    reviewer: Option<String>,
    #[serde(default)]
    action_reason: Option<String>,
    #[serde(default)]
    acceptance_is_manual: Option<bool>,
    #[serde(default)]
    resulting_edge_id: Option<RecordId>,
    #[serde(default)]
    superseded_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(default)]
    supersession_reason: Option<String>,
}

impl ProposedEdgeRow {
    pub(crate) fn into_domain(self) -> Result<ProposedEdge> {
        let edge_type = match self.edge_type.as_str() {
            "supports" => EdgeType::Supports,
            "contradicts" => EdgeType::Contradicts,
            "derived_from" => EdgeType::DerivedFrom,
            "related_to" => EdgeType::RelatedTo,
            other => {
                return Err(DbError::QueryFailed(format!(
                    "unknown proposed edge type {other:?}"
                )))
            }
        };
        let status = match self.status.as_str() {
            "pending" => ProposedEdgeStatus::Pending,
            "accepting" => ProposedEdgeStatus::Accepting,
            "accepted" => ProposedEdgeStatus::Accepted,
            "rejected" => ProposedEdgeStatus::Rejected,
            "superseded" => ProposedEdgeStatus::Superseded,
            other => {
                return Err(DbError::QueryFailed(format!(
                    "unknown proposed edge status {other:?}"
                )))
            }
        };
        Ok(ProposedEdge {
            id: self.id,
            dedupe_key: self.dedupe_key,
            from_id: self.from_id,
            to_id: self.to_id,
            edge_type,
            confidence: self.confidence,
            reason: self.reason,
            generator: self.generator,
            generator_version: self.generator_version,
            model: self.model,
            status,
            created_at: self.created_at,
            updated_at: self.updated_at,
            reviewed_at: self.reviewed_at,
            reviewer: self.reviewer,
            action_reason: self.action_reason,
            acceptance_is_manual: self.acceptance_is_manual,
            resulting_edge_id: self.resulting_edge_id,
            superseded_at: self.superseded_at,
            supersession_reason: self.supersession_reason,
        })
    }
}
