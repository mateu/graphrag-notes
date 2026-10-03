//! Provider-free, bounded inspection of the canonical IDs emitted by search.

use super::*;
use sha2::{Digest, Sha256};

/// Maximum number of messages to retrieve on either side of a selected message.
pub const MAX_INSPECTION_NEIGHBORS: usize = 20;
const MAX_LINKED_RECORDS: usize = 2 * MAX_INSPECTION_NEIGHBORS + 1;

/// Persisted provenance. A URI describes the server's source; clients must not
/// assume that it names a file on their own computer.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct InspectionProvenance {
    pub source_id: Option<String>,
    pub source_uri: Option<String>,
    pub source_type: Option<String>,
    pub source_generation: Option<u64>,
    pub heading_path: Vec<String>,
    pub start_line: Option<u64>,
    pub end_line: Option<u64>,
    pub conversation_id: Option<String>,
    pub conversation_uuid: Option<String>,
    /// Original export identity, or the conversation/index key for UUID-less messages.
    #[serde(default)]
    pub message_key: Option<String>,
    #[serde(default)]
    pub message_uuid: Option<String>,
    pub message_index: Option<i64>,
    pub role: Option<String>,
    /// Authenticated originating instance for a shared-service capture.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instance_id: Option<String>,
    /// Opaque caller provenance, nested separately from trusted instance identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InspectedConversation {
    pub id: String,
    pub uuid: String,
    pub title: Option<String>,
    pub summary: Option<String>,
    pub source_uri: Option<String>,
    pub revision: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InspectedMessage {
    pub id: String,
    #[serde(default)]
    pub message_key: String,
    pub message_uuid: Option<String>,
    pub conversation_id: String,
    pub conversation_uuid: String,
    /// Zero-based index from the imported conversation.
    pub message_index: i64,
    pub role: String,
    pub content: String,
    pub revision: String,
}

/// Exact record content and bounded chat context, without vector embeddings.
/// `revision` identifies the selected record, never the requested context size.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RecordInspection {
    pub id: String,
    pub hit_type: String,
    pub title: Option<String>,
    pub content: String,
    pub revision: String,
    pub provenance: InspectionProvenance,
    pub conversations: Vec<InspectedConversation>,
    /// Selected message and its neighbors, the first messages of a conversation,
    /// or the direct message provenance of a derived note, in index order.
    pub messages: Vec<InspectedMessage>,
    pub messages_truncated: bool,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
struct ConversationRow {
    id: RecordId,
    uuid: String,
    title: Option<String>,
    summary: Option<String>,
    source_uri: Option<String>,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    ingested_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
struct MessageRow {
    id: RecordId,
    message_key: String,
    message_uuid: Option<String>,
    conversation_id: RecordId,
    conversation_uuid: String,
    message_index: i64,
    role: String,
    content: String,
    created_at: Option<DateTime<Utc>>,
    updated_at: Option<DateTime<Utc>>,
    ingested_at: DateTime<Utc>,
}

const CONVERSATION_FIELDS: &str =
    "id, uuid, title, summary, source_uri, created_at, updated_at, ingested_at";
const MESSAGE_FIELDS: &str = "id, message_key, message_uuid, conversation_id, conversation_uuid, message_index, role, content, created_at, updated_at, ingested_at";

pub(super) fn fingerprint(value: &impl Serialize) -> Result<String> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| DbError::QueryFailed(format!("record revision: {error}")))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn ambiguous_id(id: &str) -> DbError {
    DbError::QueryFailed(format!(
        "inspection requires an unambiguous record ID; {id} exists with both string and native keys. Reimport these records with distinct IDs."
    ))
}

impl ConversationRow {
    fn inspection(&self) -> Result<InspectedConversation> {
        Ok(InspectedConversation {
            id: record_id_to_string(&self.id),
            uuid: self.uuid.clone(),
            title: self.title.clone(),
            summary: self.summary.clone(),
            source_uri: self.source_uri.clone(),
            revision: fingerprint(self)?,
        })
    }
}

impl MessageRow {
    fn inspection(&self, conversation: Option<&InspectedConversation>) -> Result<InspectedMessage> {
        Ok(InspectedMessage {
            id: record_id_to_string(&self.id),
            message_key: self.message_key.clone(),
            message_uuid: self.message_uuid.clone(),
            conversation_id: record_id_to_string(&self.conversation_id),
            conversation_uuid: self.conversation_uuid.clone(),
            message_index: self.message_index,
            role: self.role.clone(),
            content: self.content.clone(),
            revision: fingerprint(&(self, conversation))?,
        })
    }
}

impl Repository {
    /// Resolve an exact canonical search ID without an embedding or extraction
    /// provider. Raw note keys and display ordinals are deliberately rejected.
    /// Missing, staged, and superseded notes all return `NotFound`.
    pub async fn inspect_record(&self, id: &str, neighbors: usize) -> Result<RecordInspection> {
        let record_id = parse_record_id(id, None)?;
        let (table, key) = id.trim().split_once(':').expect("parsed canonical id");
        let literal_id = RecordId::new(table, key);
        let neighbors = neighbors.min(MAX_INSPECTION_NEIGHBORS);
        match table {
            "note" => self.inspect_note(id.trim(), &record_id, neighbors).await,
            "message" => {
                let literal = self.inspection_message(&literal_id).await?;
                let native = if record_id != literal_id {
                    self.inspection_message(&record_id).await?
                } else {
                    None
                };
                if literal.is_some() && native.is_some() {
                    return Err(ambiguous_id(id));
                }
                let row = literal
                    .or(native)
                    .ok_or_else(|| DbError::NotFound("message".into(), id.trim().into()))?;
                self.inspect_message(row, neighbors).await
            }
            "conversation" => {
                let literal = self.inspection_conversation(&literal_id).await?;
                let native = if record_id != literal_id {
                    self.inspection_conversation(&record_id).await?
                } else {
                    None
                };
                if literal.is_some() && native.is_some() {
                    return Err(ambiguous_id(id));
                }
                let row = literal
                    .or(native)
                    .ok_or_else(|| DbError::NotFound("conversation".into(), id.trim().into()))?;
                self.inspect_conversation(row, neighbors).await
            }
            _ => Err(DbError::QueryFailed(format!(
                "inspection requires a note:, message:, or conversation: record id, got {id:?}"
            ))),
        }
    }

    async fn inspection_conversation(&self, id: &RecordId) -> Result<Option<ConversationRow>> {
        Ok(self
            .db
            .query(format!(
                "SELECT {CONVERSATION_FIELDS} FROM conversation WHERE id = $id LIMIT 1"
            ))
            .bind(("id", id.clone()))
            .await?
            .take(0)?)
    }

    async fn inspection_message(&self, id: &RecordId) -> Result<Option<MessageRow>> {
        Ok(self
            .db
            .query(format!(
                "SELECT {MESSAGE_FIELDS} FROM message WHERE id = $id LIMIT 1"
            ))
            .bind(("id", id.clone()))
            .await?
            .take(0)?)
    }

    async fn inspect_note(
        &self,
        id: &str,
        typed_id: &RecordId,
        _neighbors: usize,
    ) -> Result<RecordInspection> {
        // Most application IDs are string keys, including UUID-shaped keys.
        // Preserve the exact visible-note getter, then support native numeric
        // and UUID keys imported through portable database snapshots.
        let literal_id = normalize_note_id(id);
        if *typed_id != literal_id {
            let literal = self.get_note(id).await?;
            let native: Option<Note> = self.db.select(typed_id.clone()).await?;
            if literal.is_some() && native.is_some() {
                return Err(ambiguous_id(id));
            }
            // A hidden string record must never redirect to a different key.
            if literal.is_some() && self.get_visible_note(id).await?.is_none() {
                return Err(DbError::NotFound("visible note".into(), id.into()));
            }
        }
        let note = match self.get_visible_note(id).await? {
            Some(note) => note,
            None => self
                .db
                .query(format!(
                    "SELECT * FROM note WHERE id = $id AND {VISIBLE_NOTE_CONDITION} LIMIT 1"
                ))
                .bind(("id", typed_id.clone()))
                .await?
                .take::<Option<Note>>(0)?
                .ok_or_else(|| DbError::NotFound("visible note".into(), id.into()))?,
        };
        let note_id = note
            .id
            .clone()
            .ok_or_else(|| DbError::NotFound("note".into(), id.into()))?;
        let mut provenance = InspectionProvenance {
            source_id: note.source_id.as_ref().map(record_id_to_string),
            source_generation: note.source_generation,
            heading_path: note.chunk_heading_path.clone(),
            start_line: note.source_start_line,
            end_line: note.source_end_line,
            ..Default::default()
        };
        let mut warnings = Vec::new();
        let source: Option<Source> = match &note.source_id {
            Some(source_id) => self.db.select(source_id.clone()).await?,
            None => None,
        };
        if let Some(source) = &source {
            if note
                .source_generation
                .is_some_and(|generation| generation != source.successful_generation)
            {
                return Err(DbError::NotFound("visible note".into(), id.into()));
            }
            provenance.source_uri = source.uri.clone().or_else(|| source.normalized_uri.clone());
            provenance.source_type = Some(
                serde_json::to_value(&source.source_type)
                    .map_err(|error| DbError::QueryFailed(error.to_string()))?
                    .as_str()
                    .unwrap_or("unknown")
                    .to_string(),
            );
            let remote_origin = match (source.source_type.clone(), source.uri.as_deref()) {
                (SourceType::Manual, Some(uri)) if uri.starts_with("mcp://capture/") => {
                    source.metadata.get("remote_capture")
                }
                (SourceType::Markdown, Some(uri)) if uri.starts_with("mcp://upload/") => {
                    source.metadata.get("remote_upload")
                }
                _ => None,
            };
            if let Some(origin) = remote_origin {
                provenance.instance_id = origin
                    .get("instance_id")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string);
                provenance.source = origin.get("source").cloned();
            }
            if source.status == SourceIngestionStatus::Failed {
                warnings.push(
                    "The latest source refresh failed; this is the last successful generation."
                        .into(),
                );
            }
        } else if note.source_id.is_some() {
            if note.source_generation.is_some() {
                return Err(DbError::NotFound("visible note".into(), id.into()));
            }
            warnings.push(
                "The source record is missing; stored note content is still available.".into(),
            );
        }

        #[derive(Deserialize, SurrealValue)]
        struct LinkRow {
            out: RecordId,
        }
        let mut response = self
            .db
            .query(
                "SELECT out FROM note_from_conversation WHERE in = $id ORDER BY out LIMIT $limit; \
             SELECT out FROM note_from_message WHERE in = $id ORDER BY out LIMIT $limit;",
            )
            .bind(("id", note_id))
            .bind(("limit", (MAX_LINKED_RECORDS + 1) as i64))
            .await?;
        let mut conversation_links: Vec<LinkRow> = response.take(0)?;
        let mut message_links: Vec<LinkRow> = response.take(1)?;
        let messages_truncated = message_links.len() > MAX_LINKED_RECORDS;
        if messages_truncated || conversation_links.len() > MAX_LINKED_RECORDS {
            warnings.push("Chat provenance is truncated to 41 linked records per kind.".into());
        }
        conversation_links.truncate(MAX_LINKED_RECORDS);
        message_links.truncate(MAX_LINKED_RECORDS);
        let mut conversations = Vec::new();
        let mut messages = Vec::new();
        for link in conversation_links {
            match self.inspection_conversation(&link.out).await? {
                Some(row) => conversations.push(row.inspection()?),
                None => warnings.push(format!(
                    "Linked conversation {} is missing.",
                    record_id_to_string(&link.out)
                )),
            }
        }
        for link in message_links {
            match self.inspection_message(&link.out).await? {
                Some(row) => {
                    if !conversations.iter().any(|conversation| {
                        conversation.id == record_id_to_string(&row.conversation_id)
                    }) {
                        match self.inspection_conversation(&row.conversation_id).await? {
                            Some(conversation) => conversations.push(conversation.inspection()?),
                            None => warnings.push(format!(
                                "Conversation {} for a linked message is missing.",
                                record_id_to_string(&row.conversation_id)
                            )),
                        }
                    }
                    let conversation_id = record_id_to_string(&row.conversation_id);
                    let conversation = conversations
                        .iter()
                        .find(|conversation| conversation.id == conversation_id);
                    messages.push(row.inspection(conversation)?);
                }
                None => warnings.push(format!(
                    "Linked message {} is missing.",
                    record_id_to_string(&link.out)
                )),
            }
        }
        conversations.sort_by(|a, b| a.id.cmp(&b.id));
        messages.sort_by(|a, b| {
            (&a.conversation_id, a.message_index, &a.id).cmp(&(
                &b.conversation_id,
                b.message_index,
                &b.id,
            ))
        });
        let primary_conversation = if let Some(message) = messages.first() {
            provenance.conversation_id = Some(message.conversation_id.clone());
            provenance.conversation_uuid = Some(message.conversation_uuid.clone());
            provenance.message_key = Some(message.message_key.clone());
            provenance.message_uuid = message.message_uuid.clone();
            provenance.message_index = Some(message.message_index);
            provenance.role = Some(message.role.clone());
            conversations
                .iter()
                .find(|conversation| conversation.id == message.conversation_id)
        } else {
            conversations.first()
        };
        if let Some(conversation) = primary_conversation {
            provenance.conversation_id = Some(conversation.id.clone());
            provenance.conversation_uuid = Some(conversation.uuid.clone());
            if provenance.source_uri.is_none() {
                provenance.source_uri = conversation.source_uri.clone();
                provenance.source_type = Some("chat_export".into());
            }
        }
        if provenance.source_type.is_none() && provenance.conversation_id.is_some() {
            provenance.source_type = Some("chat_export".into());
        }
        let revision = fingerprint(&(
            &note,
            note.created_at,
            note.updated_at,
            &provenance,
            &conversations,
            &messages,
        ))?;
        Ok(RecordInspection {
            id: record_id_to_string(note.id.as_ref().expect("checked note id")),
            hit_type: "note".into(),
            title: note.title,
            content: note.content,
            revision,
            provenance,
            conversations,
            messages,
            messages_truncated,
            warnings,
        })
    }

    async fn inspect_message(&self, row: MessageRow, neighbors: usize) -> Result<RecordInspection> {
        let conversation = self.inspection_conversation(&row.conversation_id).await?;
        let mut warnings = Vec::new();
        if conversation.is_none() {
            warnings.push(
                "The conversation record is missing; stored message content is still available."
                    .into(),
            );
        }
        let mut response = self
            .db
            .query(format!(
                "SELECT {MESSAGE_FIELDS} FROM message WHERE conversation_id = $conversation \
             AND (message_index < $index OR (message_index = $index AND id < $id)) \
             ORDER BY message_index DESC, id DESC LIMIT $limit; \
             SELECT {MESSAGE_FIELDS} FROM message WHERE conversation_id = $conversation \
             AND (message_index > $index OR (message_index = $index AND id > $id)) \
             ORDER BY message_index, id LIMIT $limit;"
            ))
            .bind(("conversation", row.conversation_id.clone()))
            .bind(("index", row.message_index))
            .bind(("id", row.id.clone()))
            .bind(("limit", (neighbors + 1) as i64))
            .await?;
        let mut before: Vec<MessageRow> = response.take(0)?;
        let mut after: Vec<MessageRow> = response.take(1)?;
        let messages_truncated = before.len() > neighbors || after.len() > neighbors;
        before.truncate(neighbors);
        after.truncate(neighbors);
        before.reverse();
        let conversations = conversation
            .iter()
            .map(ConversationRow::inspection)
            .collect::<Result<Vec<_>>>()?;
        let context = conversations.first();
        let mut messages = before
            .iter()
            .map(|message| message.inspection(context))
            .collect::<Result<Vec<_>>>()?;
        let selected = row.inspection(context)?;
        let revision = selected.revision.clone();
        messages.push(selected);
        messages.extend(
            after
                .iter()
                .map(|message| message.inspection(context))
                .collect::<Result<Vec<_>>>()?,
        );
        let provenance = InspectionProvenance {
            source_uri: conversation
                .as_ref()
                .and_then(|conversation| conversation.source_uri.clone()),
            source_type: Some("chat_export".into()),
            conversation_id: Some(record_id_to_string(&row.conversation_id)),
            conversation_uuid: Some(row.conversation_uuid.clone()),
            message_key: Some(row.message_key.clone()),
            message_uuid: row.message_uuid.clone(),
            message_index: Some(row.message_index),
            role: Some(row.role.clone()),
            ..Default::default()
        };
        let title = conversation
            .as_ref()
            .and_then(|conversation| conversation.title.as_ref())
            .map(|title| format!("{title} — {} message #{}", row.role, row.message_index + 1));
        Ok(RecordInspection {
            id: record_id_to_string(&row.id),
            hit_type: "message".into(),
            title,
            content: row.content,
            revision,
            provenance,
            conversations,
            messages,
            messages_truncated,
            warnings,
        })
    }

    async fn inspect_conversation(
        &self,
        row: ConversationRow,
        neighbors: usize,
    ) -> Result<RecordInspection> {
        let limit = 2 * neighbors + 1;
        let mut messages: Vec<MessageRow> = self
            .db
            .query(format!(
                "SELECT {MESSAGE_FIELDS} FROM message WHERE conversation_id = $conversation \
             ORDER BY message_index, id LIMIT $limit"
            ))
            .bind(("conversation", row.id.clone()))
            .bind(("limit", (limit + 1) as i64))
            .await?
            .take(0)?;
        let messages_truncated = messages.len() > limit;
        messages.truncate(limit);
        let provenance = InspectionProvenance {
            source_uri: row.source_uri.clone(),
            source_type: Some("chat_export".into()),
            conversation_id: Some(record_id_to_string(&row.id)),
            conversation_uuid: Some(row.uuid.clone()),
            ..Default::default()
        };
        let revision = fingerprint(&row)?;
        let conversations = vec![row.inspection()?];
        let messages = messages
            .iter()
            .map(|message| message.inspection(conversations.first()))
            .collect::<Result<Vec<_>>>()?;
        Ok(RecordInspection {
            id: record_id_to_string(&row.id),
            hit_type: "conversation-summary".into(),
            title: row.title,
            content: row.summary.unwrap_or_default(),
            revision,
            provenance,
            conversations,
            messages,
            messages_truncated,
            warnings: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use graphrag_core::MessageRole;

    async fn memory() -> Repository {
        Repository::new(crate::init_memory().await.unwrap())
    }

    async fn chat(repo: &Repository, indexes: &[usize]) -> (RecordId, Vec<RecordId>) {
        let now = Utc::now();
        let conversation = ChatConversation {
            uuid: "inspection-conversation".into(),
            name: "Launch café 日本語".into(),
            summary: "The launch needs review.".into(),
            created_at: now,
            updated_at: now,
            account: None,
            messages: Vec::new(),
        };
        let conversation_id = repo
            .upsert_conversation(
                &conversation,
                Some("file:///exports/chat #1.json".into()),
                serde_json::json!({}),
                None,
            )
            .await
            .unwrap();
        let mut ids = Vec::new();
        for index in indexes {
            let message = ChatMessage {
                uuid: None,
                role: if index % 2 == 0 {
                    MessageRole::Human
                } else {
                    MessageRole::Assistant
                },
                content: format!("Message {index} — café 日本語"),
                content_blocks: serde_json::json!([]),
                created_at: None,
                updated_at: None,
                attachments: Vec::new(),
                files: Vec::new(),
            };
            ids.push(
                repo.upsert_message(&conversation_id, &conversation.uuid, *index, &message, None)
                    .await
                    .unwrap(),
            );
        }
        (conversation_id, ids)
    }

    #[tokio::test]
    async fn manual_note_is_exact_provider_free_and_revision_tracks_edits() {
        let repo = memory().await;
        let note = repo
            .create_note(Note::new("Exact café 日本語\nsecond line"))
            .await
            .unwrap();
        let id = record_id_to_string(note.id.as_ref().unwrap());
        let inspected = repo.inspect_record(&id, 0).await.unwrap();
        assert_eq!(inspected.id, id);
        assert_eq!(inspected.hit_type, "note");
        assert_eq!(inspected.content, note.content);
        assert!(inspected.provenance.source_id.is_none());
        assert!(inspected.warnings.is_empty());
        assert_eq!(inspected.revision.len(), 64);
        assert_eq!(
            inspected.revision,
            repo.inspect_record(&id, 20).await.unwrap().revision
        );
        assert!(serde_json::to_value(&inspected)
            .unwrap()
            .get("embedding")
            .is_none());
        let mut changed = note;
        changed.content = "Changed content".into();
        changed.updated_at = Utc::now();
        repo.update_note(&id, changed).await.unwrap();
        assert_ne!(
            inspected.revision,
            repo.inspect_record(&id, 0).await.unwrap().revision
        );
    }

    #[tokio::test]
    async fn canonical_id_validation_and_absent_record_are_distinct() {
        let repo = memory().await;
        for invalid in ["raw-key", "note:", ":key", "entity:key", "source:key"] {
            assert!(
                matches!(
                    repo.inspect_record(invalid, 2).await,
                    Err(DbError::QueryFailed(_))
                ),
                "{invalid}"
            );
        }
        for absent in ["note:absent", "message:absent", "conversation:absent"] {
            assert!(
                matches!(
                    repo.inspect_record(absent, 2).await,
                    Err(DbError::NotFound(_, _))
                ),
                "{absent}"
            );
        }
    }

    #[tokio::test]
    async fn generated_note_provenance_respects_source_promotion_and_failed_refresh() {
        let repo = memory().await;
        let uri = "file:///notes/café 日本語 #100%.md";
        let mut source = repo
            .begin_file_import(
                SourceType::Markdown,
                "café.md".into(),
                uri.into(),
                "## Launch\nThe launch needs review.".into(),
                "first-hash".into(),
                false,
            )
            .await
            .unwrap()
            .source;
        let mut note = Note::new("The launch needs review.");
        note.source_id = source.id.clone();
        note.source_generation = Some(source.generation);
        note.chunk_heading_path = vec!["Project".into(), "Launch café".into()];
        note.source_start_line = Some(2);
        note.source_end_line = Some(3);
        let note = repo.create_note(note).await.unwrap();
        let id = record_id_to_string(note.id.as_ref().unwrap());
        assert!(matches!(
            repo.inspect_record(&id, 0).await,
            Err(DbError::NotFound(_, _))
        ));
        repo.complete_file_import(&mut source).await.unwrap();
        let initial = repo.inspect_record(&id, 0).await.unwrap();
        assert_eq!(initial.provenance.source_uri.as_deref(), Some(uri));
        assert_eq!(initial.provenance.source_type.as_deref(), Some("markdown"));
        assert_eq!(initial.provenance.heading_path, ["Project", "Launch café"]);
        assert_eq!(initial.provenance.start_line, Some(2));
        assert_eq!(initial.provenance.end_line, Some(3));
        assert_eq!(initial.provenance.source_generation, Some(1));

        let mut refreshed = repo
            .begin_file_import(
                SourceType::Markdown,
                "café.md".into(),
                uri.into(),
                "Updated source".into(),
                "second-hash".into(),
                true,
            )
            .await
            .unwrap()
            .source;
        let mut staged = Note::new("Updated source");
        staged.source_id = refreshed.id.clone();
        staged.source_generation = Some(refreshed.generation);
        let staged = repo.create_note(staged).await.unwrap();
        let staged_id = record_id_to_string(staged.id.as_ref().unwrap());
        assert!(matches!(
            repo.inspect_record(&staged_id, 0).await,
            Err(DbError::NotFound(_, _))
        ));
        repo.fail_file_import(&mut refreshed, "offline provider")
            .await
            .unwrap();
        let retained = repo.inspect_record(&id, 0).await.unwrap();
        assert_eq!(retained.content, initial.content);
        assert!(retained
            .warnings
            .iter()
            .any(|warning| warning.contains("last successful")));
        assert_eq!(retained.revision, initial.revision);

        let mut recovered = repo
            .begin_file_import(
                SourceType::Markdown,
                "café.md".into(),
                uri.into(),
                "Recovered source".into(),
                "third-hash".into(),
                true,
            )
            .await
            .unwrap()
            .source;
        let mut replacement = Note::new("Recovered source");
        replacement.source_id = recovered.id.clone();
        replacement.source_generation = Some(recovered.generation);
        let replacement = repo.create_note(replacement).await.unwrap();
        repo.complete_file_import(&mut recovered).await.unwrap();
        assert!(matches!(
            repo.inspect_record(&id, 0).await,
            Err(DbError::NotFound(_, _))
        ));
        let replacement_id = record_id_to_string(replacement.id.as_ref().unwrap());
        assert_eq!(
            repo.inspect_record(&replacement_id, 0)
                .await
                .unwrap()
                .content,
            "Recovered source"
        );
    }

    #[tokio::test]
    async fn legacy_note_with_deleted_source_returns_content_and_warning() {
        let repo = memory().await;
        let mut note = Note::new("Stored legacy content");
        note.source_id = Some(RecordId::new("source", "deleted"));
        let note = repo.create_note(note).await.unwrap();
        let inspected = repo
            .inspect_record(&record_id_to_string(note.id.as_ref().unwrap()), 0)
            .await
            .unwrap();
        assert_eq!(inspected.content, "Stored legacy content");
        assert_eq!(
            inspected.provenance.source_id.as_deref(),
            Some("source:deleted")
        );
        assert!(inspected.provenance.source_uri.is_none());
        assert!(inspected
            .warnings
            .iter()
            .any(|warning| warning.contains("source record is missing")));
    }

    #[tokio::test]
    async fn message_neighbors_are_actual_ordered_rows_even_with_index_gaps() {
        let repo = memory().await;
        let (conversation, ids) = chat(&repo, &[0, 4, 10, 20, 99]).await;
        let id = record_id_to_string(&ids[2]);
        let inspected = repo.inspect_record(&id, 1).await.unwrap();
        assert_eq!(
            inspected
                .messages
                .iter()
                .map(|message| message.message_index)
                .collect::<Vec<_>>(),
            [4, 10, 20]
        );
        assert!(inspected.messages_truncated);
        assert_eq!(inspected.content, "Message 10 — café 日本語");
        assert_eq!(
            inspected.provenance.conversation_id,
            Some(record_id_to_string(&conversation))
        );
        assert_eq!(inspected.provenance.message_index, Some(10));
        assert_eq!(inspected.provenance.role.as_deref(), Some("human"));
        assert_eq!(inspected.provenance.message_uuid, None);
        assert_eq!(
            inspected.provenance.message_key.as_deref(),
            Some("inspection-conversation:10")
        );
        assert_eq!(
            inspected.messages[1].message_key,
            "inspection-conversation:10"
        );
        assert_eq!(inspected.messages[1].message_uuid, None);
        assert!(inspected.title.unwrap().contains("Launch café 日本語"));
        let only_focus = repo.inspect_record(&id, 0).await.unwrap();
        assert_eq!(only_focus.messages.len(), 1);
        assert_eq!(only_focus.messages[0].id, id);
        assert_eq!(only_focus.revision, inspected.revision);
        let complete = repo.inspect_record(&id, usize::MAX).await.unwrap();
        assert_eq!(complete.messages.len(), 5);
        assert!(!complete.messages_truncated);
        assert_eq!(complete.revision, inspected.revision);
        for message in &complete.messages {
            for neighbors in [0, 2] {
                assert_eq!(
                    message.revision,
                    repo.inspect_record(&message.id, neighbors)
                        .await
                        .unwrap()
                        .revision,
                    "neighbor revision must be usable for a direct follow-up"
                );
            }
        }
        for conversation in &complete.conversations {
            assert_eq!(
                conversation.revision,
                repo.inspect_record(&conversation.id, 0)
                    .await
                    .unwrap()
                    .revision
            );
        }
    }

    #[tokio::test]
    async fn exported_message_uuid_survives_direct_context_and_derived_inspection() {
        let repo = memory().await;
        let (conversation, _) = chat(&repo, &[0]).await;
        let message = ChatMessage {
            uuid: Some("original-export-message-uuid".into()),
            role: MessageRole::Assistant,
            content: "Original exported reply".into(),
            content_blocks: serde_json::json!([]),
            created_at: None,
            updated_at: None,
            attachments: Vec::new(),
            files: Vec::new(),
        };
        let message_id = repo
            .upsert_message(&conversation, "inspection-conversation", 1, &message, None)
            .await
            .unwrap();
        let canonical_id = record_id_to_string(&message_id);
        let inspected = repo.inspect_record(&canonical_id, 1).await.unwrap();
        assert_eq!(inspected.id, canonical_id);
        assert_ne!(canonical_id, "message:original-export-message-uuid");
        assert_eq!(inspected.provenance.message_uuid, message.uuid);
        assert_eq!(
            inspected.provenance.message_key.as_deref(),
            message.uuid.as_deref()
        );
        assert_eq!(inspected.messages[0].message_uuid, None);
        assert_eq!(
            inspected.messages[0].message_key,
            "inspection-conversation:0"
        );
        assert_eq!(inspected.messages[1].message_uuid, message.uuid);
        assert_eq!(
            inspected.messages[1].message_key,
            "original-export-message-uuid"
        );

        let note = repo
            .create_note(Note::new("Derived original reply"))
            .await
            .unwrap();
        repo.link_note_to_message(note.id.as_ref().unwrap(), &message_id)
            .await
            .unwrap();
        let derived = repo
            .inspect_record(&record_id_to_string(note.id.as_ref().unwrap()), 0)
            .await
            .unwrap();
        assert_eq!(derived.provenance.message_uuid, message.uuid);
        assert_eq!(
            derived.provenance.message_key.as_deref(),
            message.uuid.as_deref()
        );
        assert_eq!(derived.messages[0].message_uuid, message.uuid);
        assert_eq!(
            derived.messages[0].message_key,
            "original-export-message-uuid"
        );
    }

    #[tokio::test]
    async fn reused_uuidless_message_id_changes_revision_when_content_moves() {
        let repo = memory().await;
        let (conversation, ids) = chat(&repo, &[0]).await;
        let id = record_id_to_string(&ids[0]);
        let before = repo.inspect_record(&id, 2).await.unwrap();
        let changed = ChatMessage {
            uuid: None,
            role: MessageRole::Assistant,
            content: "Different message after inserted turn".into(),
            content_blocks: serde_json::json!([]),
            created_at: None,
            updated_at: None,
            attachments: Vec::new(),
            files: Vec::new(),
        };
        let reused = repo
            .upsert_message(&conversation, "inspection-conversation", 0, &changed, None)
            .await
            .unwrap();
        assert_eq!(reused, ids[0]);
        let after = repo.inspect_record(&id, 2).await.unwrap();
        assert_ne!(before.revision, after.revision);
        assert_eq!(after.provenance.role.as_deref(), Some("assistant"));
        assert_eq!(after.content, changed.content);
    }

    #[tokio::test]
    async fn summary_inspection_is_bounded_and_supports_absent_summary() {
        let repo = memory().await;
        let (conversation, _) = chat(&repo, &(0..45).collect::<Vec<_>>()).await;
        let id = record_id_to_string(&conversation);
        let first = repo.inspect_record(&id, 1).await.unwrap();
        assert_eq!(first.hit_type, "conversation-summary");
        assert_eq!(first.content, "The launch needs review.");
        assert_eq!(first.messages.len(), 3);
        assert!(first.messages_truncated);
        let capped = repo.inspect_record(&id, usize::MAX).await.unwrap();
        assert_eq!(capped.messages.len(), 41);
        assert_eq!(capped.revision, first.revision);
        repo.db
            .query("UPDATE $id SET summary = NONE")
            .bind(("id", conversation))
            .await
            .unwrap()
            .check()
            .unwrap();
        let empty = repo.inspect_record(&id, 0).await.unwrap();
        assert_eq!(empty.content, "");
        assert_ne!(empty.revision, first.revision);
        assert_eq!(empty.messages.len(), 1);
    }

    #[tokio::test]
    async fn derived_note_resolves_message_and_conversation_links_without_source() {
        let repo = memory().await;
        let (conversation, messages) = chat(&repo, &[2]).await;
        let note = repo
            .create_note(Note::new("Derived summary note"))
            .await
            .unwrap();
        let note_id = note.id.as_ref().unwrap();
        repo.link_note_to_message(note_id, &messages[0])
            .await
            .unwrap();
        let inspected = repo
            .inspect_record(&record_id_to_string(note_id), 1)
            .await
            .unwrap();
        assert_eq!(inspected.messages.len(), 1);
        assert_eq!(inspected.messages[0].id, record_id_to_string(&messages[0]));
        assert_eq!(inspected.conversations.len(), 1);
        assert_eq!(
            inspected.conversations[0].id,
            record_id_to_string(&conversation)
        );
        assert_eq!(
            inspected.provenance.source_type.as_deref(),
            Some("chat_export")
        );
        assert_eq!(
            inspected.provenance.source_uri.as_deref(),
            Some("file:///exports/chat #1.json")
        );
        assert_eq!(inspected.provenance.message_index, Some(2));
        assert_eq!(
            inspected.messages[0].revision,
            repo.inspect_record(&inspected.messages[0].id, 0)
                .await
                .unwrap()
                .revision
        );
        assert_eq!(
            inspected.conversations[0].revision,
            repo.inspect_record(&inspected.conversations[0].id, 0)
                .await
                .unwrap()
                .revision
        );
        assert_eq!(
            inspected.revision,
            repo.inspect_record(&record_id_to_string(note_id), 0)
                .await
                .unwrap()
                .revision
        );
        repo.link_note_to_conversation(note_id, &conversation)
            .await
            .unwrap();
        assert_eq!(
            repo.inspect_record(&record_id_to_string(note_id), 0)
                .await
                .unwrap()
                .conversations
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn orphan_message_is_still_readable_and_deleted_record_is_not() {
        let repo = memory().await;
        let (conversation, messages) = chat(&repo, &[0]).await;
        let note = repo
            .create_note(Note::new("Derived orphan summary"))
            .await
            .unwrap();
        repo.link_note_to_message(note.id.as_ref().unwrap(), &messages[0])
            .await
            .unwrap();
        let conversation_id = record_id_to_string(&conversation);
        repo.db
            .query("DELETE $id")
            .bind(("id", conversation))
            .await
            .unwrap()
            .check()
            .unwrap();
        let id = record_id_to_string(&messages[0]);
        let inspected = repo.inspect_record(&id, 0).await.unwrap();
        assert_eq!(inspected.content, "Message 0 — café 日本語");
        assert!(inspected
            .warnings
            .iter()
            .any(|warning| warning.contains("conversation record is missing")));
        let derived = repo
            .inspect_record(&record_id_to_string(note.id.as_ref().unwrap()), 0)
            .await
            .unwrap();
        assert_eq!(derived.provenance.conversation_id, Some(conversation_id));
        assert_eq!(
            derived.provenance.conversation_uuid.as_deref(),
            Some("inspection-conversation")
        );
        assert_eq!(derived.provenance.message_index, Some(0));
        assert_eq!(
            derived.provenance.source_type.as_deref(),
            Some("chat_export")
        );
        assert_eq!(derived.messages[0].revision, inspected.revision);
        repo.db
            .query("DELETE $id")
            .bind(("id", messages[0].clone()))
            .await
            .unwrap()
            .check()
            .unwrap();
        assert!(matches!(
            repo.inspect_record(&id, 0).await,
            Err(DbError::NotFound(_, _))
        ));
    }

    #[tokio::test]
    async fn uuid_shaped_string_note_key_is_not_misparsed_as_native_uuid() {
        let repo = memory().await;
        let note = repo
            .create_note_and_replace_entities(Note::new("UUID string identity"), Vec::new())
            .await
            .unwrap();
        let id = record_id_to_string(note.id.as_ref().unwrap());
        assert_eq!(
            repo.inspect_record(&id, 0).await.unwrap().content,
            "UUID string identity"
        );
    }

    #[tokio::test]
    async fn native_ids_are_supported_but_colliding_string_keys_are_rejected() {
        let repo = memory().await;
        let native = RecordId::new("note", 7i64);
        let literal = RecordId::new("note", "7");
        assert_ne!(native, literal);
        assert_eq!(record_id_to_string(&native), record_id_to_string(&literal));
        repo.db.query(
            "CREATE $id SET content = 'Native numeric note', tags = [], created_at = time::now(), updated_at = time::now()"
        ).bind(("id", native.clone())).await.unwrap().check().unwrap();
        assert_eq!(
            repo.inspect_record("note:7", 0).await.unwrap().content,
            "Native numeric note"
        );

        let source = repo
            .begin_file_import(
                SourceType::Markdown,
                "ambiguous.md".into(),
                "file:///notes/ambiguous.md".into(),
                "Pending import".into(),
                "pending-hash".into(),
                false,
            )
            .await
            .unwrap()
            .source;
        repo.db.query(
            "CREATE $id SET content = 'Hidden string note', source_id = $source, source_generation = 1, tags = [], created_at = time::now(), updated_at = time::now()"
        ).bind(("id", literal.clone())).bind(("source", source.id.unwrap()))
            .await.unwrap().check().unwrap();
        assert!(repo.get_visible_note("note:7").await.unwrap().is_none());
        assert!(
            matches!(repo.inspect_record("note:7", 0).await, Err(DbError::QueryFailed(message)) if message.contains("both string and native"))
        );
        repo.db
            .query("UPDATE $id SET source_generation = NONE")
            .bind(("id", literal))
            .await
            .unwrap()
            .check()
            .unwrap();
        assert!(repo.get_visible_note("note:7").await.unwrap().is_some());
        assert!(
            matches!(repo.inspect_record("note:7", 0).await, Err(DbError::QueryFailed(message)) if message.contains("both string and native"))
        );
    }

    #[cfg(feature = "rocksdb")]
    #[tokio::test]
    async fn inspection_and_revision_survive_persistent_reopen() {
        const FIXTURE_ROOT: &str = "GRAPHRAG_INSPECTION_REOPEN_FIXTURE_ROOT";
        const FIXTURE_PHASE: &str = "GRAPHRAG_INSPECTION_REOPEN_FIXTURE_PHASE";
        if let Some(directory) = std::env::var_os(FIXTURE_ROOT) {
            let directory = std::path::PathBuf::from(directory);
            let repo = Repository::new(
                crate::init_persistent(directory.join("database"))
                    .await
                    .unwrap(),
            );
            let snapshot_path = directory.join("inspection.json");
            if std::env::var(FIXTURE_PHASE).unwrap() == "seed" {
                let (conversation, messages) = chat(&repo, &[0, 1, 2]).await;
                let note = repo
                    .create_note(Note::new("Persistent derived note"))
                    .await
                    .unwrap();
                repo.link_note_to_conversation(note.id.as_ref().unwrap(), &conversation)
                    .await
                    .unwrap();
                repo.link_note_to_message(note.id.as_ref().unwrap(), &messages[1])
                    .await
                    .unwrap();
                let before = repo
                    .inspect_record(&record_id_to_string(note.id.as_ref().unwrap()), 2)
                    .await
                    .unwrap();
                std::fs::write(
                    snapshot_path,
                    serde_json::to_vec(&(before, record_id_to_string(&messages[1]))).unwrap(),
                )
                .unwrap();
            } else {
                let (before, message_id): (RecordInspection, String) =
                    serde_json::from_slice(&std::fs::read(snapshot_path).unwrap()).unwrap();
                let after = repo.inspect_record(&before.id, 0).await.unwrap();
                assert_eq!(before, after);
                assert_eq!(
                    repo.inspect_record(&message_id, 1)
                        .await
                        .unwrap()
                        .messages
                        .len(),
                    3
                );
            }
            return;
        }
        // SurrealDB's process-wide datastore cache may retain RocksDB locks
        // after dropping a client/runtime. Process exit establishes a real
        // durable close; sleeping in the same process cannot guarantee one.
        let directory = tempfile::tempdir().unwrap();
        for phase in ["seed", "inspect"] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "repository::inspection::tests::inspection_and_revision_survive_persistent_reopen",
                    "--nocapture",
                ])
                .env(FIXTURE_ROOT, directory.path())
                .env(FIXTURE_PHASE, phase)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{phase} process failed:\n{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
}
