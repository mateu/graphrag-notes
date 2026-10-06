//! Atomic shared-service capture and successful-request replay.
//!
//! The transport authenticates the instance and fingerprints the original
//! request before provider work. This module never accepts bearer credentials
//! or interprets the opaque caller provenance as a filesystem path.

use super::*;
use sha2::{Digest, Sha256};

#[cfg(test)]
#[path = "remote_capture_tests.rs"]
mod tests;

#[derive(Debug, Clone)]
pub struct RemoteCaptureInput {
    pub authenticated_instance_id: String,
    pub request_id: String,
    /// SHA-256 of the versioned original request, calculated by the trusted adapter.
    pub payload_fingerprint: String,
    pub content: String,
    pub title: Option<String>,
    pub tags: Vec<String>,
    pub source_provenance: serde_json::Value,
}

/// Exact original wire snapshot, without vectors or credentials. A later edit,
/// deletion or restore never silently turns replay into a new capture.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RemoteCaptureReceipt {
    pub result: serde_json::Value,
    pub replayed: bool,
}

#[derive(Debug, Deserialize, SurrealValue)]
struct ReceiptRow {
    payload_fingerprint: String,
    payload: serde_json::Value,
    result: serde_json::Value,
}

fn validate_identity(instance: &str, request: &str, fingerprint: &str) -> Result<()> {
    for (label, value) in [("instance ID", instance), ("request ID", request)] {
        if value.is_empty()
            || value.len() > 256
            || value.trim() != value
            || value.chars().any(char::is_control)
        {
            return Err(DbError::InvalidRemoteRequest(format!("{label} must be a nonempty bounded identity without controls or surrounding whitespace")));
        }
    }
    if fingerprint.len() != 64
        || !fingerprint
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(DbError::InvalidRemoteRequest(
            "payload fingerprint must be lowercase SHA-256 hex".into(),
        ));
    }
    Ok(())
}

fn conflict(instance: &str, request: &str) -> DbError {
    DbError::RemoteRequestConflict {
        instance_id: instance.into(),
        request_id: request.into(),
    }
}

impl Repository {
    /// Stable authenticated request identity, shared with preparation before
    /// inference. Opaque caller provenance never participates in this key.
    pub fn remote_capture_note_id(instance: &str, request: &str) -> Result<RecordId> {
        let bytes = serde_json::to_vec(&(instance, request))
            .map_err(|error| DbError::InvalidRemoteRequest(error.to_string()))?;
        Ok(RecordId::new(
            "note",
            format!("{:x}", Sha256::digest(bytes)),
        ))
    }

    async fn remote_capture_row(
        &self,
        instance: &str,
        request: &str,
    ) -> Result<Option<ReceiptRow>> {
        Ok(self
            .db
            .query(
                "SELECT payload_fingerprint, payload, result FROM remote_capture_receipt \
             WHERE instance_id = $instance AND request_id = $request LIMIT 1",
            )
            .bind(("instance", instance.to_string()))
            .bind(("request", request.to_string()))
            .await?
            .take(0)?)
    }

    /// Call before inference so a replay succeeds even with providers offline.
    pub async fn find_remote_capture_receipt(
        &self,
        instance: &str,
        request: &str,
        fingerprint: &str,
    ) -> Result<Option<RemoteCaptureReceipt>> {
        validate_identity(instance, request, fingerprint)?;
        match self.remote_capture_row(instance, request).await? {
            Some(row) if row.payload_fingerprint == fingerprint => Ok(Some(RemoteCaptureReceipt {
                result: row.result,
                replayed: true,
            })),
            Some(_) => Err(conflict(instance, request)),
            None => Ok(None),
        }
    }

    pub async fn get_remote_capture_receipt(
        &self,
        instance: &str,
        request: &str,
        fingerprint: &str,
    ) -> Result<Option<RemoteCaptureReceipt>> {
        self.find_remote_capture_receipt(instance, request, fingerprint)
            .await
    }

    /// Create source provenance, note, entity mentions and the successful
    /// receipt in one transaction. The lifecycle lock serializes competing
    /// captures in the server's shared connection; deterministic IDs and the
    /// unique request index also prohibit duplicate committed records.
    pub async fn capture_remote_note(
        &self,
        input: RemoteCaptureInput,
        embedding: Vec<f32>,
        entities: Vec<Entity>,
    ) -> Result<RemoteCaptureReceipt> {
        validate_identity(
            &input.authenticated_instance_id,
            &input.request_id,
            &input.payload_fingerprint,
        )?;
        if input.content.trim().is_empty() || embedding.iter().any(|value| !value.is_finite()) {
            return Err(DbError::InvalidRemoteRequest(
                "capture content must be nonempty and vectors finite".into(),
            ));
        }
        let payload = serde_json::json!({"content": input.content, "title": input.title, "tags": input.tags, "source": input.source_provenance});
        let _guard = self.proposal_acceptance_lock.lock().await;
        if let Some(row) = self
            .remote_capture_row(&input.authenticated_instance_id, &input.request_id)
            .await?
        {
            if row.payload_fingerprint != input.payload_fingerprint || row.payload != payload {
                return Err(conflict(
                    &input.authenticated_instance_id,
                    &input.request_id,
                ));
            }
            return Ok(RemoteCaptureReceipt {
                result: row.result,
                replayed: true,
            });
        }
        let key_bytes = serde_json::to_vec(&(&input.authenticated_instance_id, &input.request_id))
            .map_err(|error| DbError::InvalidRemoteRequest(error.to_string()))?;
        let key = format!("{:x}", Sha256::digest(key_bytes));
        let note_id =
            Self::remote_capture_note_id(&input.authenticated_instance_id, &input.request_id)?;
        let source_id = RecordId::new("source", key.clone());
        let receipt_id = RecordId::new("remote_capture_receipt", key.clone());
        let uri = format!("mcp://capture/{key}");
        let metadata = serde_json::json!({"remote_capture": {"instance_id": input.authenticated_instance_id, "request_id": input.request_id, "source": input.source_provenance}});
        let mut note = Note::new(input.content.clone())
            .with_embedding(embedding)
            .with_tags(input.tags.clone())
            .with_source(source_id.clone());
        note.id = Some(note_id.clone());
        note.title = input.title.clone();
        note.search_content = Some(input.content.clone());
        let provenance = InspectionProvenance {
            source_id: Some(record_id_to_string(&source_id)),
            source_uri: Some(uri.clone()),
            source_type: Some("manual".into()),
            instance_id: Some(input.authenticated_instance_id.clone()),
            source: Some(input.source_provenance.clone()),
            ..Default::default()
        };
        let revision = super::inspection::fingerprint(&(
            &note,
            note.created_at,
            note.updated_at,
            &provenance,
            Vec::<InspectedConversation>::new(),
            Vec::<InspectedMessage>::new(),
        ))?;
        let result = serde_json::json!({
            "id": record_id_to_string(&note_id), "revision": revision,
            "title": note.title, "content": note.content, "tags": note.tags,
            "created_at": note.created_at.to_rfc3339(), "updated_at": note.updated_at.to_rfc3339(),
            "provenance": provenance,
        });
        let entity_names = entities
            .iter()
            .map(Entity::effective_identity_key)
            .collect::<Vec<_>>();
        let entities_sql = super::notes::replacement_entities_transaction();
        // Validate the full stored note before recording its response. A schema
        // normalization or default which changes that snapshot aborts all writes
        // rather than committing a response with an incorrect revision.
        let snapshot_guard = super::notes::editor_snapshot_guard(false);
        let mut response = self.db.query(format!(
            "BEGIN TRANSACTION; \
             CREATE $source SET source_type = 'manual', title = $title, uri = $uri, normalized_uri = $uri, \
                generation = 0, successful_generation = 0, status = 'ready', metadata = $metadata, \
                created_at = <datetime>$created, updated_at = <datetime>$updated; \
             {entities_sql}\
             CREATE $note SET note_type = 'raw', title = $title, content = $content, embedding = $embedding, \
                source_id = $source, tags = $tags, search_content = $content, \
                created_at = <datetime>$created, updated_at = <datetime>$updated; \
             FOR $entity_id IN $entity_ids {{ CREATE mentions SET in = $note, out = $entity_id; }}; \
             {snapshot_guard}\
             IF array::len((SELECT VALUE id FROM source WHERE id = $source AND source_type = 'manual' \
                AND uri = $uri AND normalized_uri = $uri AND metadata = $metadata)) != 1 \
                {{ THROW 'remote capture source snapshot changed'; }}; \
             CREATE $receipt SET instance_id = $instance, request_id = $request, payload_fingerprint = $fingerprint, \
                payload = $payload, result = $result, note_id = $note, source_id = $source, \
                created_at = <datetime>$created, updated_at = <datetime>$updated; \
             COMMIT TRANSACTION;"
        ))
            .bind(("source", source_id)).bind(("note", note_id.clone())).bind(("receipt", receipt_id))
            .bind(("title", input.title)).bind(("uri", uri)).bind(("metadata", metadata))
            .bind(("content", input.content)).bind(("embedding", (!note.embedding.is_empty()).then_some(note.embedding.clone())))
            .bind(("tags", input.tags)).bind(("created", note.created_at.to_rfc3339())).bind(("updated", note.updated_at.to_rfc3339()))
            .bind(("replacement_entities", entities)).bind(("replacement_entity_names", entity_names))
            .bind(("editor_source", note_id)).bind(("editor_expected", note))
            .bind(("instance", input.authenticated_instance_id.clone())).bind(("request", input.request_id.clone()))
            .bind(("fingerprint", input.payload_fingerprint.clone())).bind(("payload", payload.clone())).bind(("result", result))
            .await?;
        let errors = response.take_errors();
        if !errors.is_empty() {
            // A competing transaction may have won the unique request claim.
            // This is failure recovery only, never a post-commit result read.
            if let Some(row) = self
                .remote_capture_row(&input.authenticated_instance_id, &input.request_id)
                .await?
            {
                if row.payload_fingerprint != input.payload_fingerprint || row.payload != payload {
                    return Err(conflict(
                        &input.authenticated_instance_id,
                        &input.request_id,
                    ));
                }
                return Ok(RemoteCaptureReceipt {
                    result: row.result,
                    replayed: true,
                });
            }
            return Err(DbError::QueryFailed(format!(
                "atomic remote capture failed: {errors:?}"
            )));
        }
        let result_index = response
            .num_statements()
            .checked_sub(2)
            .ok_or_else(|| DbError::CreateFailed("remote capture receipt result".into()))?;
        let stored: Option<ReceiptRow> = response.take(result_index)?;
        stored
            .map(|row| RemoteCaptureReceipt {
                result: row.result,
                replayed: false,
            })
            .ok_or_else(|| DbError::CreateFailed("remote capture receipt".into()))
    }
}
