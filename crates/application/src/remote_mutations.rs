//! Manual-note editing and governance share the embedded repository and agents.
//! Provider preparation never holds the mutation gate. Final public revisions,
//! full SQL snapshots, lifecycle effects and the receipt commit together.
use crate::inference::{CancellableEmbedder, CancellableExtractor};
use crate::*;
use graphrag_agents::LibrarianAgent;
use graphrag_core::{record_id_to_string, Note, ProposedEdgeStatus};
use graphrag_db::{
    MutationNoteSnapshot, RemoteCaptureReceipt, RemoteMutationEffect, RemoteMutationInput,
};
use sha2::{Digest, Sha256};
use std::sync::Arc;

fn identity(caller: &CallerIdentity, request: &str) -> ApplicationResult<()> {
    for value in [&caller.instance_id, request] {
        if value.is_empty()
            || value.len() > 256
            || value.chars().count() > 128
            || value.trim() != value
            || value.chars().any(char::is_control)
        {
            return Err(ApplicationError::Validation("Request/instance identities must be nonempty, at most 128 characters/256 bytes, without controls or surrounding whitespace.".into()));
        }
    }
    Ok(())
}
/// Canonical object-key order also makes replay stable if another dependency
/// enables serde_json's preserve_order feature in a future build.
fn canonical(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(object) => {
            let sorted = object
                .iter()
                .map(|(key, value)| (key.clone(), canonical(value)))
                .collect::<std::collections::BTreeMap<_, _>>();
            serde_json::Value::Object(sorted.into_iter().collect())
        }
        serde_json::Value::Array(array) => {
            serde_json::Value::Array(array.iter().map(canonical).collect())
        }
        value => value.clone(),
    }
}
pub fn remote_mutation_fingerprint(
    operation: &str,
    payload: &serde_json::Value,
) -> ApplicationResult<String> {
    let bytes = serde_json::to_vec(&(
        REMOTE_MUTATION_PAYLOAD_VERSION,
        operation,
        canonical(payload),
    ))
    .map_err(|e| ApplicationError::Internal(e.to_string()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn input<T: serde::Serialize>(
    caller: &CallerIdentity,
    request_id: &str,
    operation: &str,
    request: &T,
) -> ApplicationResult<RemoteMutationInput> {
    identity(caller, request_id)?;
    let payload =
        serde_json::to_value(request).map_err(|e| ApplicationError::Internal(e.to_string()))?;
    let fingerprint = remote_mutation_fingerprint(operation, &payload)?;
    Ok(RemoteMutationInput {
        instance_id: caller.instance_id.clone(),
        request_id: request_id.into(),
        operation: operation.into(),
        payload_fingerprint: fingerprint,
        payload,
        result: serde_json::json!({}),
    })
}
fn response(
    request_id: String,
    receipt: RemoteCaptureReceipt,
) -> ApplicationResult<RemoteMutationResponse> {
    Ok(RemoteMutationResponse {
        request_id,
        replayed: receipt.replayed,
        outcome: serde_json::from_value(receipt.result).map_err(|e| {
            ApplicationError::Internal(format!("Unsupported mutation receipt: {e}"))
        })?,
    })
}
fn revision(actual: &str, expected: &str) -> ApplicationResult<()> {
    if actual != expected {
        return Err(ApplicationError::RevisionConflict(
            "Revision mismatch. Refresh the snapshot before applying the retained draft/decision."
                .into(),
        ));
    }
    Ok(())
}
fn manual(snapshot: &MutationNoteSnapshot) -> ApplicationResult<()> {
    if !snapshot.manual {
        return Err(ApplicationError::Validation("Only manual notes may be edited or deleted remotely. Imported file/chat notes remain source-owned; change their original source instead.".into()));
    }
    Ok(())
}
fn validate_reference(id: &str, expected: &str, table: &str) -> ApplicationResult<()> {
    if id.len() > 512 || id.trim() != id || id.chars().any(char::is_control) {
        return Err(ApplicationError::Validation(
            "Canonical record IDs must be at most 512 bytes without controls/whitespace.".into(),
        ));
    }
    graphrag_db::parse_record_id(id, Some(table))?;
    if id.split_once(':').map(|(prefix, _)| prefix) != Some(table) {
        return Err(ApplicationError::Validation(format!(
            "Expected a canonical {table}: record ID."
        )));
    }
    if expected.len() != 64
        || !expected
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(ApplicationError::Validation(
            "A revision from the current snapshot is required.".into(),
        ));
    }
    Ok(())
}
fn outcome(
    input: &mut RemoteMutationInput,
    id: &str,
    previous_revision: &str,
    status: &str,
    edge: Option<String>,
    cascade: Option<serde_json::Value>,
) -> ApplicationResult<()> {
    input.result = serde_json::to_value(RemoteMutationOutcome {
        id: id.into(),
        operation: input.operation.clone(),
        previous_revision: previous_revision.into(),
        actor: format!("mcp:{}", input.instance_id),
        status: status.into(),
        resulting_edge_id: edge,
        cascade,
    })
    .map_err(|e| ApplicationError::Internal(e.to_string()))?;
    Ok(())
}

impl EmbeddedApplication {
    pub(crate) async fn remote_note_snapshot_impl(
        &self,
        reference: RecordRef,
    ) -> ApplicationResult<RemoteNoteSnapshot> {
        validate_record_id(&reference.id)?;
        let guard = self.repo.mutation_guard().await;
        let snapshot = self
            .repo
            .mutation_note_snapshot(&guard, &reference.id)
            .await?;
        if let Some(expected) = reference.revision {
            revision(&snapshot.inspection.revision, &expected)?;
        }
        Ok(RemoteNoteSnapshot {
            id: snapshot.inspection.id,
            revision: snapshot.inspection.revision,
            title: snapshot.note.title,
            content: snapshot.note.content,
            tags: snapshot.note.tags,
            editable: snapshot.manual,
            blocked_reason: (!snapshot.manual).then(|| {
                "Imported file/chat notes are source-owned; change the original source.".into()
            }),
        })
    }

    pub(crate) async fn remote_edit_impl(
        &self,
        caller: CallerIdentity,
        request: RemoteEditRequest,
        cancellation: ActionCancellation,
    ) -> ApplicationResult<RemoteMutationResponse> {
        validate_reference(&request.id, &request.revision, "note")?;
        if request.patch.clear_title && request.patch.title.is_some() {
            return Err(ApplicationError::Validation(
                "Specify title or clear_title, not both.".into(),
            ));
        }
        if request.patch.content.is_none()
            && request.patch.title.is_none()
            && !request.patch.clear_title
            && request.patch.tags.is_none()
        {
            return Err(ApplicationError::Validation(
                "Specify at least one note patch field.".into(),
            ));
        }
        if request
            .patch
            .content
            .as_ref()
            .is_some_and(|v| v.trim().is_empty() || v.len() > 65536 || v.contains('\0'))
            || request
                .patch
                .title
                .as_ref()
                .is_some_and(|v| v.chars().count() > 512 || v.contains('\0'))
            || request.patch.tags.as_ref().is_some_and(|tags| {
                tags.len() > 32
                    || tags
                        .iter()
                        .any(|t| t.trim().is_empty() || t.chars().count() > 64 || t.contains('\0'))
            })
        {
            return Err(ApplicationError::Validation("Patch content must be nonempty and at most 65536 bytes, title at most 512 characters, and tags at most 32 nonempty values of 64 characters.".into()));
        }
        let mut input = input(&caller, &request.request_id, "edit", &request)?;
        if let Some(receipt) = self.repo.find_remote_mutation_receipt(&input).await? {
            return response(request.request_id, receipt);
        }
        let original = {
            let guard = self.repo.mutation_guard().await;
            let original = self
                .repo
                .mutation_note_snapshot(&guard, &request.id)
                .await?;
            revision(&original.inspection.revision, &request.revision)?;
            manual(&original)?;
            original
        };
        let content_changed = request
            .patch
            .content
            .as_ref()
            .is_some_and(|v| v != &original.note.content);
        let mut replacement = original.note.clone();
        if let Some(content) = &request.patch.content {
            replacement.content = content.clone();
        }
        if request.patch.clear_title {
            replacement.title = None;
        } else if let Some(title) = &request.patch.title {
            replacement.title = Some(title.clone());
        }
        if let Some(tags) = &request.patch.tags {
            replacement.tags = tags.clone();
        }
        let entities = if content_changed {
            self.require_providers(!self.runtime.skip_entity_extraction, &cancellation)
                .await?;
            let librarian = LibrarianAgent::new(
                self.repo.clone(),
                Arc::new(CancellableEmbedder(
                    self.embedder.clone(),
                    cancellation.clone(),
                )),
                Arc::new(CancellableExtractor(
                    self.extractor.clone(),
                    cancellation.clone(),
                )),
            )
            .with_runtime_config(self.runtime.clone())
            .with_cancellation_flag(cancellation.flag());
            let (prepared, entities) = librarian
                .prepare_manual_capture(
                    replacement.content.clone(),
                    replacement.title.clone(),
                    replacement.tags.clone(),
                )
                .await?;
            replacement.embedding = prepared.embedding;
            Some(entities)
        } else {
            None
        };
        if cancellation.is_cancelled() {
            return Err(ApplicationError::Cancelled);
        }
        // Reacquire after inference. A source/chat/proposal/background writer
        // which changed any public revision field invalidates this draft.
        let guard = self.repo.mutation_guard().await;
        if let Some(receipt) = self.repo.find_remote_mutation_receipt(&input).await? {
            return response(request.request_id, receipt);
        }
        let current = self
            .repo
            .mutation_note_snapshot(&guard, &request.id)
            .await?;
        revision(&current.inspection.revision, &request.revision)?;
        manual(&current)?;
        replacement.updated_at = graphrag_core::Note::new("").updated_at;
        outcome(
            &mut input,
            &request.id,
            &request.revision,
            "edited",
            None,
            None,
        )?;
        let receipt = self
            .repo
            .apply_remote_mutation(
                &guard,
                input,
                RemoteMutationEffect::Edit {
                    expected: Box::new(current.note),
                    replacement: Box::new(replacement),
                    entities,
                },
            )
            .await?;
        response(request.request_id, receipt)
    }

    pub(crate) async fn remote_delete_impl(
        &self,
        caller: CallerIdentity,
        request: RemoteDeleteRequest,
    ) -> ApplicationResult<RemoteMutationResponse> {
        validate_reference(&request.id, &request.revision, "note")?;
        if !request.confirmed {
            return Err(ApplicationError::Validation(
                "Deletion requires explicit confirmed=true after reviewing the snapshot.".into(),
            ));
        }
        let mut input = input(&caller, &request.request_id, "delete", &request)?;
        if let Some(receipt) = self.repo.find_remote_mutation_receipt(&input).await? {
            return response(request.request_id, receipt);
        }
        let guard = self.repo.mutation_guard().await;
        if let Some(receipt) = self.repo.find_remote_mutation_receipt(&input).await? {
            return response(request.request_id, receipt);
        }
        let snapshot = self
            .repo
            .mutation_note_snapshot(&guard, &request.id)
            .await?;
        revision(&snapshot.inspection.revision, &request.revision)?;
        manual(&snapshot)?;
        let cascade = serde_json::to_value(
            self.repo
                .mutation_delete_preview(&guard, &snapshot.note)
                .await?,
        )
        .map_err(|e| ApplicationError::Internal(e.to_string()))?;
        outcome(
            &mut input,
            &request.id,
            &request.revision,
            "deleted",
            None,
            Some(cascade),
        )?;
        let receipt = self
            .repo
            .apply_remote_mutation(
                &guard,
                input,
                RemoteMutationEffect::Delete {
                    expected: Box::new(snapshot.note),
                },
            )
            .await?;
        response(request.request_id, receipt)
    }

    pub(crate) async fn remote_decision_impl(
        &self,
        caller: CallerIdentity,
        request: RemoteDecisionRequest,
    ) -> ApplicationResult<RemoteMutationResponse> {
        validate_reference(&request.id, &request.revision, "proposed_edge")?;
        if !request.confirmed {
            return Err(ApplicationError::Validation(
                "Proposal decisions require explicit confirmed=true after reviewing the card."
                    .into(),
            ));
        }
        if request
            .reason
            .as_ref()
            .is_some_and(|v| v.chars().count() > 2048 || v.contains('\0'))
        {
            return Err(ApplicationError::Validation(
                "Decision reason must contain at most 2048 characters.".into(),
            ));
        }
        let operation = match request.action {
            ProposalAction::Accept => "accept",
            ProposalAction::Reject => "reject",
            ProposalAction::Undo => "undo",
        };
        let mut input = input(&caller, &request.request_id, operation, &request)?;
        if let Some(receipt) = self.repo.find_remote_mutation_receipt(&input).await? {
            return response(request.request_id, receipt);
        }
        let guard = self.repo.mutation_guard().await;
        if let Some(receipt) = self.repo.find_remote_mutation_receipt(&input).await? {
            return response(request.request_id, receipt);
        }
        let proposal = self
            .repo
            .mutation_proposal_snapshot(&guard, &request.id)
            .await?;
        let id = proposal
            .id
            .clone()
            .ok_or_else(|| ApplicationError::Internal("Proposal has no ID.".into()))?;
        let card = proposal_card(&self.repo, proposal.clone()).await?;
        revision(&card.revision, &request.revision)?;
        let status = match request.action {
            ProposalAction::Accept
                if proposal.status == ProposedEdgeStatus::Pending && card.accept_allowed =>
            {
                "accepted"
            }
            ProposalAction::Reject if proposal.status == ProposedEdgeStatus::Pending => "rejected",
            ProposalAction::Undo if proposal.status == ProposedEdgeStatus::Accepted => "superseded",
            _ => {
                return Err(ApplicationError::RevisionConflict(
                    "Proposal action is unavailable in its current lifecycle. Refresh the card."
                        .into(),
                ))
            }
        };
        let mut endpoints: Vec<Note> = Vec::new();
        for endpoint in [&proposal.from_id, &proposal.to_id] {
            match self
                .repo
                .mutation_note_snapshot(&guard, &record_id_to_string(endpoint))
                .await
            {
                Ok(snapshot) => endpoints.push(snapshot.note),
                Err(graphrag_db::DbError::NotFound(..))
                    if !matches!(request.action, ProposalAction::Accept) => {}
                Err(error) => return Err(error.into()),
            }
        }
        let edge = if matches!(request.action, ProposalAction::Accept) {
            let key = format!(
                "{:x}",
                Sha256::digest(
                    serde_json::to_vec(&id)
                        .map_err(|e| ApplicationError::Internal(e.to_string()))?
                )
            );
            Some(format!("{}:remote_{key}", proposal.edge_type))
        } else {
            None
        };
        outcome(
            &mut input,
            &request.id,
            &request.revision,
            status,
            edge,
            None,
        )?;
        let receipt = self
            .repo
            .apply_remote_mutation(
                &guard,
                input,
                RemoteMutationEffect::Decision {
                    expected: Box::new(proposal),
                    endpoints,
                    action: operation.into(),
                    reason: request.reason,
                },
            )
            .await?;
        response(request.request_id, receipt)
    }
}
