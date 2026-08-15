//! Source lifecycle, staging, promotion, and generated-note ownership.
//!
//! Promotion and deletion retain their original transaction and proposal
//! lifecycle ordering; no source-visible semantics are changed here.

use super::graph::{
    canonicalize_note_edge, edge_dedupe_key, persisted_note_edge_type, SourceDeleteCount,
};
use super::*;

impl Repository {
    pub async fn copy_note_dependents_to_successors(
        &self,
        successors: &[(RecordId, RecordId, bool)],
    ) -> Result<()> {
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        self.copy_note_dependents_to_successors_locked(successors)
            .await
    }

    #[allow(clippy::mutable_key_type)] // Typed `RecordId` maps are copied directly into edge writes.
    async fn copy_note_dependents_to_successors_locked(
        &self,
        successors: &[(RecordId, RecordId, bool)],
    ) -> Result<()> {
        if successors.is_empty() {
            return Ok(());
        }
        let exact_content_successors = successors
            .iter()
            .filter_map(|(old_id, _, exact_content)| exact_content.then_some(old_id.clone()))
            .collect::<HashSet<_>>();
        let successors = successors
            .iter()
            .map(|(old_id, new_id, _)| (old_id.clone(), new_id.clone()))
            .collect::<HashMap<_, _>>();

        // Entity mentions describe extracted source text, so only carry them
        // across when that displayed text is exactly unchanged. A changed
        // successor remains mention-free and is therefore eligible for a new
        // entity-extraction pass. Chat provenance identifies origin rather
        // than extracted content, so it follows every safely reconciled chunk.
        for (old_id, new_id) in &successors {
            if exact_content_successors.contains(old_id) {
                let entity_ids: Vec<RecordId> = self
                    .db
                    .query("SELECT VALUE out FROM mentions WHERE in = $note_id")
                    .bind(("note_id", old_id.clone()))
                    .await?
                    .take(0)?;
                for entity_id in entity_ids {
                    self.link_note_to_entity_locked(new_id, &entity_id).await?;
                }
            }

            let conversation_ids: Vec<RecordId> = self
                .db
                .query("SELECT VALUE out FROM note_from_conversation WHERE in = $note_id")
                .bind(("note_id", old_id.clone()))
                .await?
                .take(0)?;
            for conversation_id in conversation_ids {
                self.link_note_to_conversation_locked(new_id, &conversation_id)
                    .await?;
            }

            let message_ids: Vec<RecordId> = self
                .db
                .query("SELECT VALUE out FROM note_from_message WHERE in = $note_id")
                .bind(("note_id", old_id.clone()))
                .await?
                .take(0)?;
            for message_id in message_ids {
                self.link_note_to_message_locked(new_id, &message_id)
                    .await?;
            }
        }

        // Snapshot each edge once before writing successors. This handles an
        // edge whose two endpoints are both reconciled chunks and preserves
        // manual as well as generated graph relationships.
        #[allow(clippy::mutable_key_type)] // Edge ids remain typed for successor copying.
        let mut seen_edges = HashSet::new();
        let mut edges = Vec::new();
        for old_id in successors.keys() {
            for edge in self.get_note_edges(&record_id_to_string(old_id)).await? {
                if seen_edges.insert(edge.id.clone()) {
                    edges.push(edge);
                }
            }
        }
        for edge in edges {
            let from_id = successors.get(&edge.in_id).cloned().unwrap_or(edge.in_id);
            let to_id = successors.get(&edge.out_id).cloned().unwrap_or(edge.out_id);
            if from_id == to_id {
                // A many-to-one reconciliation must not manufacture an
                // invalid self-edge. Current Markdown keys are one-to-one,
                // but retaining this guard makes the copy routine safe for
                // future reconciliation strategies.
                continue;
            }
            self.create_audited_edge(
                &from_id,
                &to_id,
                persisted_note_edge_type(&edge.edge_type)?,
                edge.confidence,
                edge.reason.as_deref(),
                edge.provenance
                    .as_deref()
                    .unwrap_or("source-reconciliation"),
                edge.proposal_id.as_ref(),
                edge.is_manual,
            )
            .await?;
        }
        Ok(())
    }

    // ==========================================
    // SOURCE OPERATIONS
    // ==========================================

    /// Create a source and return its database-assigned id.
    #[instrument(skip(self, source))]
    pub async fn create_source(&self, source: Source) -> Result<Source> {
        let created: Option<Source> = self
            .db
            .query("CREATE source CONTENT $source RETURN AFTER")
            .bind(("source", source_content_value(&source)?))
            .await?
            .take(0)?;
        created.ok_or_else(|| DbError::CreateFailed("source".into()))
    }

    /// List sources in stable identity order for human and JSON CLI output.
    #[instrument(skip(self))]
    pub async fn list_sources(&self) -> Result<Vec<Source>> {
        Ok(self
            .db
            .query("SELECT * FROM source ORDER BY normalized_uri, created_at, id")
            .await?
            .take(0)?)
    }

    /// Resolve a source by record id or by its normalized/legacy URI.
    #[instrument(skip(self))]
    pub async fn get_source(&self, id_or_uri: &str) -> Result<Option<Source>> {
        if let Some(raw_id) = id_or_uri.strip_prefix("source:") {
            let source: Option<Source> = self.db.select(("source", raw_id)).await?;
            return Ok(source);
        }

        let source: Option<Source> = self
            .db
            .query("SELECT * FROM source WHERE normalized_uri = $key OR uri = $key LIMIT 1")
            .bind(("key", id_or_uri.to_string()))
            .await?
            .take(0)?;
        Ok(source)
    }

    /// Begin a staged import. The old successful generation remains untouched
    /// until `complete_file_import` succeeds, making failed refreshes safe.
    #[instrument(skip(self, content))]
    pub async fn begin_file_import(
        &self,
        source_type: SourceType,
        title: String,
        normalized_uri: String,
        content: String,
        content_hash: String,
        force: bool,
    ) -> Result<SourceImportPlan> {
        if let Some(mut existing) = self.get_source(&normalized_uri).await? {
            if existing.source_type == SourceType::Manual {
                return Err(DbError::QueryFailed(format!(
                    "refusing to replace manual source at {normalized_uri}"
                )));
            }
            if !force
                && existing.status == SourceIngestionStatus::Ready
                && existing.content_hash.as_deref() == Some(content_hash.as_str())
            {
                // A process can stop after promotion and before old-generation
                // cleanup. An otherwise unchanged retry is the natural
                // recovery path; finish that deferred cleanup before reporting
                // a no-op so stale records cannot accumulate indefinitely.
                let cleanup = self.cleanup_non_successful_generations(&existing).await?;
                return Ok(SourceImportPlan {
                    source: existing,
                    action: SourceImportAction::Unchanged,
                    cleanup,
                });
            }

            existing.generation = existing.generation.saturating_add(1).max(1);
            existing.source_type = source_type;
            existing.title = Some(title);
            existing.uri = Some(normalized_uri.clone());
            existing.normalized_uri = Some(normalized_uri);
            existing.content = Some(content);
            existing.content_hash = Some(content_hash);
            existing.status = SourceIngestionStatus::Pending;
            existing.last_error = None;
            existing.updated_at = chrono::Utc::now();
            self.replace_source(&existing).await?;
            return Ok(SourceImportPlan {
                source: existing,
                action: SourceImportAction::Updated,
                cleanup: SourceDeleteSummary::default(),
            });
        }

        let now = chrono::Utc::now();
        let source = Source {
            id: None,
            source_type,
            title: Some(title),
            uri: Some(normalized_uri.clone()),
            normalized_uri: Some(normalized_uri),
            content: Some(content),
            content_hash: Some(content_hash),
            generation: 1,
            successful_generation: 0,
            status: SourceIngestionStatus::Pending,
            last_error: None,
            metadata: serde_json::json!({}),
            created_at: now,
            updated_at: now,
            last_ingested_at: None,
        };
        Ok(SourceImportPlan {
            source: self.create_source(source).await?,
            action: SourceImportAction::Created,
            cleanup: SourceDeleteSummary::default(),
        })
    }

    /// Promote an import before removing superseded records. If cleanup is
    /// interrupted, the new generation is already searchable and the old
    /// generation remains recoverable (but hidden) for a later cleanup.
    /// Manual and legacy notes have no generation and survive.
    #[instrument(skip(self, source))]
    pub async fn complete_file_import(&self, source: &mut Source) -> Result<SourceDeleteSummary> {
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        self.complete_file_import_locked(source).await
    }

    /// Atomically carry reconciled dependents into a staged source generation,
    /// promote it, then retire the superseded generation. A graph mutation
    /// therefore either lands before the dependent snapshot and is copied, or
    /// observes the completed transition instead of being silently discarded
    /// during old-generation cleanup.
    #[instrument(skip(self, source, successors))]
    pub async fn reconcile_file_import(
        &self,
        source: &mut Source,
        successors: &[(RecordId, RecordId, bool)],
    ) -> Result<SourceDeleteSummary> {
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        self.copy_note_dependents_to_successors_locked(successors)
            .await?;
        self.complete_file_import_locked(source).await
    }

    async fn complete_file_import_locked(
        &self,
        source: &mut Source,
    ) -> Result<SourceDeleteSummary> {
        // Promotion, proposal retargeting, and old-generation cleanup are one
        // lifecycle transition. In particular, acceptance/undo must not run
        // after the old endpoints become hidden but before an accepted
        // proposal is retargeted to its staged replacement edge.
        let source_id = source
            .id
            .as_ref()
            .ok_or_else(|| DbError::CreateFailed("source id".into()))?
            .clone();
        let summary = self
            .source_delete_summary(&source_id, Some(source.generation), true)
            .await?;
        self.promote_file_import_locked(source).await?;
        // Proposal-backed edges are staged copy-on-write with their new note
        // endpoints. Once promotion makes those endpoints authoritative,
        // retarget the accepted proposal before deleting the old generation so
        // its audit row and undo path follow the replacement edge.
        self.retarget_reconciled_proposals(&source_id, source.generation)
            .await?;
        // Do this only after durable promotion. A failure here can leave old
        // records behind, but cannot leave the corpus with no visible complete
        // generation; visibility selects `successful_generation`.
        self.delete_source_notes_locked(&source_id, Some(source.generation), true)
            .await?;
        Ok(summary)
    }

    async fn retarget_reconciled_proposals(
        &self,
        source_id: &RecordId,
        generation: u64,
    ) -> Result<()> {
        let note_ids = self
            .source_owned_note_ids(source_id, Some(generation), false)
            .await?;
        #[allow(clippy::mutable_key_type)] // Edge ids remain typed for proposal retargeting.
        let mut seen_edges = HashSet::new();
        for note_id in note_ids {
            for edge in self.get_note_edges(&record_id_to_string(&note_id)).await? {
                let Some(proposal_id) = edge.proposal_id.as_ref() else {
                    continue;
                };
                if !seen_edges.insert(edge.id.clone()) {
                    continue;
                }
                let edge_type = persisted_note_edge_type(&edge.edge_type)?;
                let mut from_id = edge.in_id.clone();
                let mut to_id = edge.out_id.clone();
                canonicalize_note_edge(&mut from_id, &mut to_id, &edge_type);
                let dedupe_key = edge_dedupe_key(&from_id, &to_id, &edge_type);
                #[derive(Deserialize, SurrealValue)]
                struct UpdatedRow {
                    id: RecordId,
                }
                let updated: Option<UpdatedRow> = self
                    .db
                    .query(
                        "UPDATE $proposal SET in = $from, out = $to, dedupe_key = $dedupe_key, resulting_edge_id = $edge, updated_at = time::now() WHERE status = 'accepted' RETURN AFTER",
                    )
                    .bind(("proposal", proposal_id.clone()))
                    .bind(("from", from_id))
                    .bind(("to", to_id))
                    .bind(("dedupe_key", dedupe_key))
                    .bind(("edge", edge.id.clone()))
                    .await?
                    .take(0)?;
                if updated.is_none() {
                    let proposal = self.get_edge_proposal(proposal_id).await?.ok_or_else(|| {
                        DbError::NotFound(
                            "reconciled proposal".into(),
                            record_id_to_string(proposal_id),
                        )
                    })?;
                    if matches!(
                        proposal.status,
                        ProposedEdgeStatus::Rejected | ProposedEdgeStatus::Superseded
                    ) {
                        // A crash can leave a staged copy of an accepted edge
                        // after the original edge was undone. Its proposal is
                        // terminal, so preserve that user decision by dropping
                        // the stale copied edge rather than making recovery
                        // fail forever trying to retarget it.
                        self.db
                            .query("DELETE $edge")
                            .bind(("edge", edge.id.clone()))
                            .await?
                            .check()?;
                        continue;
                    }
                    return Err(DbError::QueryFailed(format!(
                        "reconciled proposal {} is no longer accepted",
                        record_id_to_string(proposal_id)
                    )));
                }
            }
        }
        Ok(())
    }

    #[cfg(test)]
    async fn promote_file_import(&self, source: &mut Source) -> Result<()> {
        // Promotion changes which source generation is visible. Keep that
        // transition and retirement of proposals for the newly hidden notes
        // atomic with respect to proposal acceptance in this repository.
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        self.promote_file_import_locked(source).await
    }

    async fn promote_file_import_locked(&self, source: &mut Source) -> Result<()> {
        let source_id = source
            .id
            .as_ref()
            .ok_or_else(|| DbError::CreateFailed("source id".into()))?
            .clone();
        // Do not update the caller's in-memory source until the visibility
        // transition is durable. Callers use this state to distinguish a
        // pre-promotion failure (safe to discard staged notes) from a later
        // cleanup failure (new generation must remain intact for recovery).
        let mut promoted = source.clone();
        promoted.successful_generation = promoted.generation;
        promoted.status = SourceIngestionStatus::Ready;
        promoted.last_error = None;
        promoted.updated_at = chrono::Utc::now();
        promoted.last_ingested_at = Some(promoted.updated_at);
        self.replace_source(&promoted).await?;
        *source = promoted;
        // Promotion makes older source generations invisible even if their
        // destructive cleanup is interrupted. Retire their pending proposals
        // at the same durable boundary so batch acceptance cannot create an
        // edge for a hidden endpoint in that window.
        let superseded_notes = self
            .source_owned_note_ids(&source_id, Some(source.generation), true)
            .await?;
        self.supersede_pending_proposals_for_notes(&superseded_notes)
            .await
    }

    async fn cleanup_non_successful_generations(
        &self,
        source: &Source,
    ) -> Result<SourceDeleteSummary> {
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        let source_id = source
            .id
            .as_ref()
            .ok_or_else(|| DbError::CreateFailed("source id".into()))?;
        // A process can stop after promotion but before proposal retargeting
        // or old-generation deletion. Retry retargeting before cleanup so an
        // accepted proposal's audit and undo path follows its visible staged
        // replacement instead of being retired with the old generation.
        self.retarget_reconciled_proposals(source_id, source.successful_generation)
            .await?;
        self.delete_source_notes_locked(source_id, Some(source.successful_generation), true)
            .await
    }

    /// Remove partially-created notes for a failed generation and retain the
    /// last successful generation. The source remains resumable via reimport.
    #[instrument(skip(self, source, error))]
    pub async fn fail_file_import(&self, source: &mut Source, error: impl ToString) -> Result<()> {
        let source_id = source
            .id
            .as_ref()
            .ok_or_else(|| DbError::CreateFailed("source id".into()))?;
        self.delete_source_notes(source_id, Some(source.generation), false)
            .await?;
        source.status = SourceIngestionStatus::Failed;
        source.last_error = Some(error.to_string());
        source.updated_at = chrono::Utc::now();
        self.replace_source(source).await
    }

    /// Count every record that source deletion would mutate. This is used for
    /// dry-run output and intentionally excludes manual/legacy notes.
    #[instrument(skip(self, source))]
    pub async fn preview_source_delete(&self, source: &Source) -> Result<SourceDeleteSummary> {
        let source_id = source
            .id
            .as_ref()
            .ok_or_else(|| DbError::NotFound("source".into(), "missing id".into()))?;
        self.source_delete_summary(source_id, None, false).await
    }

    /// Delete a source and the records it owns. Edges/provenance/mentions are
    /// removed before notes, so no dangling graph records remain. Shared entity
    /// records are deliberately retained: without per-source entity ownership,
    /// deleting an unmentioned entity could erase a user-authored entity.
    #[instrument(skip(self, source))]
    pub async fn delete_source(&self, source: &Source) -> Result<SourceDeleteSummary> {
        let source_id = source
            .id
            .as_ref()
            .ok_or_else(|| DbError::NotFound("source".into(), "missing id".into()))?;
        let summary = self.delete_source_notes(source_id, None, false).await?;
        let _: Option<Source> = self.db.delete(source_id.clone()).await?;
        Ok(summary)
    }

    async fn replace_source(&self, source: &Source) -> Result<()> {
        let id = source
            .id
            .as_ref()
            .ok_or_else(|| DbError::CreateFailed("source id".into()))?;
        let content = source_content_value(source)?;
        self.db
            .query("UPDATE $id MERGE $source")
            .bind(("id", id.clone()))
            .bind(("source", content))
            .await?
            .check()?;
        self.db
            .query(
                "UPDATE $id SET updated_at = <datetime>$updated_at, \
                 last_error = $last_error, \
                 last_ingested_at = IF $last_ingested_at = NONE THEN NONE ELSE <datetime>$last_ingested_at END",
            )
            .bind(("id", id.clone()))
            .bind(("updated_at", source.updated_at.to_rfc3339()))
            .bind(("last_error", source.last_error.clone()))
            .bind((
                "last_ingested_at",
                source.last_ingested_at.map(|time| time.to_rfc3339()),
            ))
            .await?
            .check()?;
        Ok(())
    }

    async fn delete_source_notes(
        &self,
        source_id: &RecordId,
        generation: Option<u64>,
        older_than_generation: bool,
    ) -> Result<SourceDeleteSummary> {
        // Source cleanup shares the same endpoint/acceptance critical section
        // as single-note deletion.
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        self.delete_source_notes_locked(source_id, generation, older_than_generation)
            .await
    }

    async fn delete_source_notes_locked(
        &self,
        source_id: &RecordId,
        generation: Option<u64>,
        older_than_generation: bool,
    ) -> Result<SourceDeleteSummary> {
        let summary = self
            .source_delete_summary(source_id, generation, older_than_generation)
            .await?;
        let notes = self
            .source_owned_note_ids(source_id, generation, older_than_generation)
            .await?;
        self.supersede_proposals_for_removed_notes(&notes).await?;
        self.delete_notes_and_dependents(&notes).await?;
        Ok(summary)
    }

    /// Delete note rows and every relationship/provenance record owned by or
    /// incident on them. Proposal retirement is deliberately separate so
    /// callers can choose the lifecycle transition before this physical
    /// cascade runs.
    pub(crate) async fn delete_notes_and_dependents(&self, notes: &[RecordId]) -> Result<()> {
        for note_id in notes {
            self.db
                .query(
                    "DELETE supports WHERE in = $note OR out = $note; \
                     DELETE contradicts WHERE in = $note OR out = $note; \
                     DELETE derived_from WHERE in = $note OR out = $note; \
                     DELETE related_to WHERE in = $note OR out = $note; \
                     DELETE mentions WHERE in = $note; \
                     DELETE note_from_conversation WHERE in = $note; \
                     DELETE note_from_message WHERE in = $note; \
                     DELETE $note;",
                )
                .bind(("note", note_id.clone()))
                .await?
                .check()?;
        }
        Ok(())
    }

    /// Retire every mutable proposal whose source-owned endpoint is being
    /// removed. Accepted proposals lose their resulting edge reference along
    /// with the edge itself, matching [`Self::undo_edge`] semantics.
    pub(crate) async fn supersede_proposals_for_removed_notes(
        &self,
        notes: &[RecordId],
    ) -> Result<()> {
        for note_id in notes {
            self.db
                .query(
                    "UPDATE proposed_edge SET status = 'superseded', superseded_at = time::now(), supersession_reason = 'proposal endpoint removed by source lifecycle', resulting_edge_id = NONE, updated_at = time::now() WHERE (status = 'pending' OR status = 'accepting' OR status = 'accepted') AND (in = $note OR out = $note)",
                )
                .bind(("note", note_id.clone()))
                .await?
                .check()?;
        }
        Ok(())
    }

    /// Retire pending suggestions whose source-owned endpoint is no longer
    /// usable. Promotion uses this narrower transition because old-generation
    /// accepted edges remain intact until deferred destructive cleanup runs.
    async fn supersede_pending_proposals_for_notes(&self, notes: &[RecordId]) -> Result<()> {
        for note_id in notes {
            self.db
                .query(
                    "UPDATE proposed_edge SET status = 'superseded', superseded_at = time::now(), supersession_reason = 'proposal endpoint removed by source lifecycle', updated_at = time::now() WHERE (status = 'pending' OR status = 'accepting') AND (in = $note OR out = $note)",
                )
                .bind(("note", note_id.clone()))
                .await?
                .check()?;
        }
        Ok(())
    }

    async fn source_owned_note_ids(
        &self,
        source_id: &RecordId,
        generation: Option<u64>,
        older_than_generation: bool,
    ) -> Result<Vec<RecordId>> {
        let condition = match (generation, older_than_generation) {
            (Some(_), true) => "source_generation IS NOT NONE AND source_generation != $generation",
            (Some(_), false) => "source_generation = $generation",
            (None, _) => "source_generation IS NOT NONE",
        };
        let query =
            format!("SELECT VALUE id FROM note WHERE source_id = $source_id AND {condition}");
        let mut request = self.db.query(query).bind(("source_id", source_id.clone()));
        if let Some(generation) = generation {
            request = request.bind(("generation", generation as i64));
        }
        Ok(request.await?.take(0)?)
    }

    async fn source_delete_summary(
        &self,
        source_id: &RecordId,
        generation: Option<u64>,
        older_than_generation: bool,
    ) -> Result<SourceDeleteSummary> {
        let notes = self
            .source_owned_note_ids(source_id, generation, older_than_generation)
            .await?;
        self.delete_summary_for_notes(&notes).await
    }

    /// Count the relationships and provenance records that physical note
    /// cleanup removes. Shared source/note accounting keeps dry-run previews
    /// exact and makes confirmed single-note deletion report the same shape.
    pub(crate) async fn delete_summary_for_notes(
        &self,
        notes: &[RecordId],
    ) -> Result<SourceDeleteSummary> {
        let mut summary = SourceDeleteSummary {
            notes: notes.len() as u64,
            note_edges: self.count_note_edges_for_notes(notes).await?,
            proposals: self.count_mutable_proposals_for_notes(notes).await?,
            ..Default::default()
        };
        for note_id in notes {
            let counts: Vec<SourceDeleteCount> = self
                .db
                .query(
                    "RETURN [\
                       { kind: 'mentions', count: (SELECT count() FROM mentions WHERE in = $note GROUP ALL)[0].count },\
                       { kind: 'conversation_provenance', count: (SELECT count() FROM note_from_conversation WHERE in = $note GROUP ALL)[0].count },\
                       { kind: 'message_provenance', count: (SELECT count() FROM note_from_message WHERE in = $note GROUP ALL)[0].count }\
                     ];",
                )
                .bind(("note", note_id.clone()))
                .await?
                .take(0)?;
            for count in counts {
                match count.kind.as_str() {
                    "mentions" => summary.mentions += count.count,
                    "conversation_provenance" => {
                        summary.note_conversation_provenance += count.count
                    }
                    "message_provenance" => summary.note_message_provenance += count.count,
                    _ => {}
                }
            }
        }
        Ok(summary)
    }

    /// Count each edge row once across all owned notes. An internal edge is
    /// reachable from two endpoints but is deleted exactly once, so summing
    /// per-note counts would make dry-run output inaccurate.
    async fn count_note_edges_for_notes(&self, notes: &[RecordId]) -> Result<u64> {
        let mut total = 0_u64;
        for table in ["supports", "contradicts", "derived_from", "related_to"] {
            #[derive(Deserialize, SurrealValue)]
            struct CountRow {
                #[serde(default)]
                count: Option<u64>,
            }

            let query = format!(
                "SELECT count() FROM {table} WHERE in IN $notes OR out IN $notes GROUP ALL"
            );
            let row: Option<CountRow> = self
                .db
                .query(query)
                .bind(("notes", notes.to_vec()))
                .await?
                .take(0)?;
            total += row.and_then(|row| row.count).unwrap_or(0);
        }
        Ok(total)
    }

    /// Count proposal records that source cleanup will transition. Proposal
    /// rows are not deleted: they retain an auditable terminal decision.
    async fn count_mutable_proposals_for_notes(&self, notes: &[RecordId]) -> Result<u64> {
        if notes.is_empty() {
            return Ok(0);
        }
        #[derive(Deserialize, SurrealValue)]
        struct CountRow {
            #[serde(default)]
            count: Option<u64>,
        }
        let row: Option<CountRow> = self
            .db
            .query(
                "SELECT count() FROM proposed_edge WHERE (status = 'pending' OR status = 'accepting' OR status = 'accepted') AND (in IN $notes OR out IN $notes) GROUP ALL",
            )
            .bind(("notes", notes.to_vec()))
            .await?
            .take(0)?;
        Ok(row.and_then(|row| row.count).unwrap_or(0))
    }
}
