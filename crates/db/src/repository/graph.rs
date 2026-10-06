//! Entity, accepted-edge, and proposal lifecycle ownership.
//!
//! Acceptance, undo, and graph traversal retain their original lifecycle
//! locking and deterministic query ordering.

use super::chats::{
    EdgeProposalDraft, GraphEntityMatch, GraphEntityNoteSeed, NoteEdgeRow, ProposedEdgeRow,
};
use super::*;
use surrealdb_types::ToSql;

/// Fetch the stored endpoint before inspecting fields. Plain `endpoint.id`
/// can read an object-key component, falsely admitting a dangling compound
/// record ID. `endpoint.*` fetches the document, preserving the former visible
/// note-table membership check without selecting the whole note corpus.
pub(super) fn graph_endpoint_visible_sql(endpoint: &str) -> String {
    format!(
        "(record::tb({endpoint}) = 'note' AND type::is_range(record::id({endpoint})) = false AND {endpoint}.*.id IS NOT NONE AND ({endpoint}.*.source_id IS NONE OR {endpoint}.*.source_generation IS NONE OR {endpoint}.*.source_generation = {endpoint}.*.source_id.successful_generation))"
    )
}

pub(super) fn graph_endpoint_eligible_sql(endpoint: &str) -> String {
    format!(
        "({} AND ($since = NONE OR {endpoint}.*.created_at >= <datetime>$since) AND ($source_uri = NONE OR {endpoint}.*.source_id.uri = $source_uri))",
        graph_endpoint_visible_sql(endpoint)
    )
}

/// Every source/table retains its own ordered limit; only the database
/// request is shared. Evaluate eligibility on referenced endpoint documents
/// so even an empty edge table never scans the full note corpus.
pub(super) fn graph_edge_batch_sql(
    note_count: usize,
    tables: &[&str],
    allow_outbound: bool,
    allow_inbound: bool,
) -> String {
    let mut sql = String::new();
    let in_visible = graph_endpoint_visible_sql("in");
    let out_visible = graph_endpoint_visible_sql("out");
    let in_eligible = graph_endpoint_eligible_sql("in");
    let out_eligible = graph_endpoint_eligible_sql("out");
    for table in tables {
        for index in 0..note_count {
            let direction = match (allow_outbound, allow_inbound) {
                (true, true) => format!(
                    "((in = $note_{index} AND {in_visible} AND {out_eligible} AND out NOT IN $visited_{index}) OR (out = $note_{index} AND {out_visible} AND {in_eligible} AND in NOT IN $visited_{index}))"
                ),
                (true, false) => format!(
                    "in = $note_{index} AND {in_visible} AND {out_eligible} AND out NOT IN $visited_{index}"
                ),
                (false, true) => format!(
                    "out = $note_{index} AND {out_visible} AND {in_eligible} AND in NOT IN $visited_{index}"
                ),
                (false, false) => "false".into(),
            };
            sql.push_str(&format!(
                "SELECT id, '{table}' AS edge_type, in AS in_id, out AS out_id, proposal_id, confidence, reason, provenance, is_manual, created_at, IF confidence = NONE THEN 1.0 ELSE confidence END AS graph_confidence \
                 FROM {table} WHERE {direction} AND (confidence = NONE OR confidence >= $min_confidence) \
                 ORDER BY graph_confidence DESC, id ASC LIMIT $limit;"
            ));
        }
    }
    sql
}

pub(super) fn graph_mention_batch_sql(entity_count: usize) -> String {
    let mut sql = String::new();
    let eligible = graph_endpoint_eligible_sql("in");
    for index in 0..entity_count {
        sql.push_str(&format!(
            "SELECT in AS note_id, out AS entity_id FROM mentions WHERE out = $entity_{index} AND {eligible} ORDER BY in ASC LIMIT $limit;"
        ));
    }
    sql
}

impl Repository {
    #[instrument(skip(self))]
    pub async fn create_edge(
        &self,
        from_id: &surrealdb::types::RecordId,
        to_id: &surrealdb::types::RecordId,
        edge_type: EdgeType,
        confidence: Option<f32>,
    ) -> Result<()> {
        // A source reconciliation snapshots old-generation dependents before
        // atomically promoting and retiring that generation. Serialize manual
        // graph writes with that transition so a write cannot be accepted in
        // the snapshot/cleanup window and then silently removed by cleanup.
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        validate_note_edge(from_id, to_id, &edge_type)?;
        if !self.note_is_writable(from_id).await? || !self.note_is_writable(to_id).await? {
            return Err(DbError::NotFound(
                "note endpoint".into(),
                "a graph edge endpoint is hidden, failed, or no longer exists".into(),
            ));
        }
        self.create_audited_edge(
            from_id,
            to_id,
            edge_type,
            confidence,
            None,
            "manual_api",
            None,
            true,
        )
        .await?;
        Ok(())
    }

    /// Persist a similarity-derived Gardener proposal. Similarity may only
    /// produce `related_to`, never a logical support/contradiction assertion.
    #[instrument(skip(self))]
    pub async fn upsert_gardener_proposal(
        &self,
        from_id: &RecordId,
        to_id: &RecordId,
        confidence: f32,
        reason: String,
        generator_version: Option<String>,
        model: Option<String>,
    ) -> Result<ProposedEdge> {
        self.upsert_edge_proposal(EdgeProposalDraft {
            from_id: from_id.clone(),
            to_id: to_id.clone(),
            edge_type: EdgeType::RelatedTo,
            confidence,
            reason,
            generator: "gardener-similarity".into(),
            generator_version,
            model,
        })
        .await
    }

    /// Create or update a proposal identified by its stable canonical key.
    /// Terminal proposals are returned unchanged: a repeated scan must not
    /// silently resurrect a user decision or create an equivalent duplicate.
    #[instrument(skip(self, draft))]
    pub async fn upsert_edge_proposal(&self, mut draft: EdgeProposalDraft) -> Result<ProposedEdge> {
        let _lifecycle_guard = self.proposal_acceptance_lock.lock().await;
        validate_note_edge(&draft.from_id, &draft.to_id, &draft.edge_type)?;
        canonicalize_note_edge(&mut draft.from_id, &mut draft.to_id, &draft.edge_type);
        draft.confidence = draft.confidence.clamp(0.0, 1.0);
        let dedupe_key = edge_dedupe_key(&draft.from_id, &draft.to_id, &draft.edge_type);

        if let Some(existing) = self.find_proposal_by_dedupe_key(&dedupe_key).await? {
            if existing.status == ProposedEdgeStatus::Pending {
                #[derive(Deserialize, SurrealValue)]
                struct UpdatedRow {
                    id: RecordId,
                }
                let id = existing.id.clone().expect("stored proposal has id");
                let updated: Option<UpdatedRow> = self
                    .db
                    .query(
                        "UPDATE $id SET confidence = $confidence, reason = $reason, generator = $generator, generator_version = $generator_version, model = $model, updated_at = time::now() WHERE status = 'pending' RETURN AFTER",
                    )
                    .bind(("id", id.clone()))
                    .bind(("confidence", draft.confidence))
                    .bind(("reason", draft.reason))
                    .bind(("generator", draft.generator))
                    .bind(("generator_version", draft.generator_version))
                    .bind(("model", draft.model))
                    .await?
                    .take(0)?;
                return self
                    .get_edge_proposal(&updated.map(|row| row.id).unwrap_or(id))
                    .await?
                    .ok_or_else(|| DbError::QueryFailed("updated proposal disappeared".into()));
            }
            return Ok(existing);
        }

        #[derive(Deserialize, SurrealValue)]
        struct IdRow {
            id: RecordId,
        }
        let insert = self
            .db
            .query(
                "INSERT INTO proposed_edge (dedupe_key, in, out, edge_type, confidence, reason, generator, generator_version, model, status, created_at, updated_at) VALUES ($dedupe_key, $from, $to, $edge_type, $confidence, $reason, $generator, $generator_version, $model, 'pending', time::now(), time::now()) RETURN id",
            )
            .bind(("dedupe_key", dedupe_key.clone()))
            .bind(("from", draft.from_id))
            .bind(("to", draft.to_id))
            .bind(("edge_type", draft.edge_type.to_string()))
            .bind(("confidence", draft.confidence))
            .bind(("reason", draft.reason))
            .bind(("generator", draft.generator))
            .bind(("generator_version", draft.generator_version))
            .bind(("model", draft.model))
            .await;
        let created_result: Result<Vec<IdRow>> = match insert {
            Ok(mut response) => response.take(0).map_err(Into::into),
            Err(error) => Err(error.into()),
        };
        let created = match created_result {
            Ok(created) => created,
            Err(error) => {
                // A concurrent scan can pass the lookup above at the same
                // time. The unique dedupe index elects one winner; reload it
                // instead of surfacing a spurious duplicate-key failure.
                if let Some(existing) = self.find_proposal_by_dedupe_key(&dedupe_key).await? {
                    return Ok(existing);
                }
                return Err(error);
            }
        };
        let id = created
            .into_iter()
            .next()
            .ok_or_else(|| DbError::QueryFailed("create proposed_edge".into()))?
            .id;
        self.get_edge_proposal(&id)
            .await?
            .ok_or_else(|| DbError::QueryFailed("created proposal disappeared".into()))
    }

    /// Fetch a proposal by record id.
    #[instrument(skip(self))]
    pub async fn get_edge_proposal(&self, id: &RecordId) -> Result<Option<ProposedEdge>> {
        let proposals: Vec<ProposedEdgeRow> = self
            .db
            .query(proposal_select_sql("WHERE id = $id"))
            .bind(("id", id.clone()))
            .await?
            .take(0)?;
        proposals
            .into_iter()
            .next()
            .map(ProposedEdgeRow::into_domain)
            .transpose()
    }

    /// List proposals, optionally filtering by lifecycle status.
    #[instrument(skip(self))]
    pub async fn list_edge_proposals(
        &self,
        status: Option<ProposedEdgeStatus>,
        limit: usize,
    ) -> Result<Vec<ProposedEdge>> {
        let where_clause = if status.is_some() {
            "WHERE status = $status"
        } else {
            ""
        };
        let mut query = self
            .db
            .query(format!(
                "{} ORDER BY updated_at DESC LIMIT $limit",
                proposal_select_sql(where_clause)
            ))
            .bind(("limit", limit.max(1)));
        if let Some(status) = status {
            query = query.bind(("status", status.to_string()));
        }
        let rows: Vec<ProposedEdgeRow> = query.await?.take(0)?;
        rows.into_iter().map(ProposedEdgeRow::into_domain).collect()
    }

    /// Accept a pending proposal, creating one auditable accepted edge. Calling
    /// it again after acceptance returns the same accepted proposal unchanged.
    #[instrument(skip(self))]
    pub async fn accept_edge_proposal(
        &self,
        id: &RecordId,
        reviewer: Option<String>,
        action_reason: Option<String>,
        is_manual: bool,
    ) -> Result<ProposedEdge> {
        // All clones of one repository serialize completion. The database
        // state claim below remains the cross-process guard, while this lock
        // prevents an in-process loser from releasing a shared claim.
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        let proposal = self
            .get_edge_proposal(id)
            .await?
            .ok_or_else(|| DbError::NotFound("proposed_edge".into(), record_id_to_string(id)))?;
        match proposal.status {
            ProposedEdgeStatus::Accepted => {
                return self
                    .recover_or_return_accepted_proposal(id, proposal, is_manual)
                    .await;
            }
            ProposedEdgeStatus::Accepting => {
                return self.resume_acceptance(id, proposal, is_manual).await;
            }
            ProposedEdgeStatus::Pending => {}
            _ => {
                return Err(DbError::QueryFailed(format!(
                    "proposal {} is {}, not pending",
                    record_id_to_string(id),
                    proposal.status
                )));
            }
        }
        // Claim the pending row before creating the accepted edge. `accepting`
        // is recoverable: retries resume its idempotent edge creation rather
        // than exposing a completed `accepted` proposal without an edge id.
        if !self
            .claim_pending_proposal(
                id,
                ProposedEdgeStatus::Accepting,
                reviewer,
                action_reason,
                Some(is_manual),
            )
            .await?
        {
            return self.acceptance_claim_lost(id, is_manual).await;
        }
        let claimed = self
            .get_edge_proposal(id)
            .await?
            .ok_or_else(|| DbError::QueryFailed("acceptance claim disappeared".into()))?;
        self.resume_acceptance(id, claimed, is_manual).await
    }

    /// Complete (or retry) an `accepting` claim. The edge upsert is idempotent
    /// by dedupe key, so an interruption after edge creation can be recovered
    /// safely by repeating this method.
    async fn resume_acceptance(
        &self,
        id: &RecordId,
        proposal: ProposedEdge,
        is_manual: bool,
    ) -> Result<ProposedEdge> {
        if !self.note_is_visible(&proposal.from_id).await?
            || !self.note_is_visible(&proposal.to_id).await?
        {
            self.mark_claimed_proposal_stale(id, "proposal endpoint is no longer visible")
                .await?;
            return Err(DbError::QueryFailed(format!(
                "proposal {} is stale: an endpoint is no longer visible",
                record_id_to_string(id)
            )));
        }
        let (edge_id, edge_proposal_id) = match self
            .create_audited_edge(
                &proposal.from_id,
                &proposal.to_id,
                proposal.edge_type.clone(),
                Some(proposal.confidence),
                Some(&proposal.reason),
                &proposal.generator,
                Some(id),
                proposal.acceptance_is_manual.unwrap_or(is_manual),
            )
            .await
        {
            Ok(edge) => edge,
            // Keep the durable `accepting` claim intact. A transient failure
            // is recoverable by a later retry/batch run; releasing it here
            // could clear another caller's in-flight completion.
            Err(error) => return Err(error),
        };
        if edge_proposal_id.as_ref() != Some(id) {
            self.mark_claimed_proposal_materialized(id).await?;
            return Err(DbError::QueryFailed(format!(
                "proposal {} is superseded because an independent equivalent edge already exists",
                record_id_to_string(id)
            )));
        }
        if !self.finalize_acceptance_claim(id, edge_id).await? {
            let current = self.get_edge_proposal(id).await?.ok_or_else(|| {
                DbError::QueryFailed("acceptance finalization disappeared".into())
            })?;
            if current.status == ProposedEdgeStatus::Accepted && current.resulting_edge_id.is_some()
            {
                return Ok(current);
            }
            return Err(DbError::QueryFailed(format!(
                "proposal {} acceptance remains recoverable; retry the operation",
                record_id_to_string(id)
            )));
        }
        self.get_edge_proposal(id)
            .await?
            .ok_or_else(|| DbError::QueryFailed("accepted proposal disappeared".into()))
    }

    /// Reject a pending proposal. Repeating the same rejection is a no-op.
    #[instrument(skip(self))]
    pub async fn reject_edge_proposal(
        &self,
        id: &RecordId,
        reviewer: Option<String>,
        action_reason: Option<String>,
    ) -> Result<ProposedEdge> {
        let _lifecycle_guard = self.proposal_acceptance_lock.lock().await;
        let proposal = self
            .get_edge_proposal(id)
            .await?
            .ok_or_else(|| DbError::NotFound("proposed_edge".into(), record_id_to_string(id)))?;
        if proposal.status == ProposedEdgeStatus::Rejected {
            return Ok(proposal);
        }
        if proposal.status != ProposedEdgeStatus::Pending {
            return Err(DbError::QueryFailed(format!(
                "proposal {} is {}, not pending",
                record_id_to_string(id),
                proposal.status
            )));
        }
        if !self
            .claim_pending_proposal(
                id,
                ProposedEdgeStatus::Rejected,
                reviewer,
                action_reason,
                None,
            )
            .await?
        {
            let current = self.get_edge_proposal(id).await?.ok_or_else(|| {
                DbError::NotFound("proposed_edge".into(), record_id_to_string(id))
            })?;
            if current.status == ProposedEdgeStatus::Rejected {
                return Ok(current);
            }
            return Err(DbError::QueryFailed(format!(
                "proposal {} is {}, not pending",
                record_id_to_string(id),
                current.status
            )));
        }
        self.get_edge_proposal(id)
            .await?
            .ok_or_else(|| DbError::QueryFailed("rejected proposal disappeared".into()))
    }

    /// Atomically transition a pending proposal to a claimed state. Rejection
    /// is terminal immediately; acceptance is finalized only after its edge
    /// has been created and recorded.
    pub(crate) async fn claim_pending_proposal(
        &self,
        id: &RecordId,
        status: ProposedEdgeStatus,
        reviewer: Option<String>,
        action_reason: Option<String>,
        acceptance_is_manual: Option<bool>,
    ) -> Result<bool> {
        #[derive(Deserialize, SurrealValue)]
        struct ClaimRow {
            id: RecordId,
        }
        let claimed: Option<ClaimRow> = self
            .db
            .query(
                "UPDATE $id SET status = $status, reviewed_at = time::now(), reviewer = $reviewer, action_reason = $action_reason, acceptance_is_manual = $acceptance_is_manual, updated_at = time::now() WHERE status = 'pending' RETURN AFTER",
            )
            .bind(("id", id.clone()))
            .bind(("status", status.to_string()))
            .bind(("reviewer", reviewer))
            .bind(("action_reason", action_reason))
            .bind(("acceptance_is_manual", acceptance_is_manual))
            .await?
            .take(0)?;
        Ok(claimed.is_some())
    }

    async fn acceptance_claim_lost(&self, id: &RecordId, is_manual: bool) -> Result<ProposedEdge> {
        let current = self
            .get_edge_proposal(id)
            .await?
            .ok_or_else(|| DbError::NotFound("proposed_edge".into(), record_id_to_string(id)))?;
        match current.status {
            ProposedEdgeStatus::Accepted => {
                self.recover_or_return_accepted_proposal(id, current, is_manual)
                    .await
            }
            ProposedEdgeStatus::Accepting => self.resume_acceptance(id, current, is_manual).await,
            _ => Err(DbError::QueryFailed(format!(
                "proposal {} is {}, not pending",
                record_id_to_string(id),
                current.status
            ))),
        }
    }

    /// Repair legacy/incomplete `accepted` rows that have no resulting edge
    /// reference. New code never leaves this state, but retries can safely
    /// convert it into an `accepting` claim and recreate or rediscover the
    /// deduplicated edge.
    async fn recover_or_return_accepted_proposal(
        &self,
        id: &RecordId,
        proposal: ProposedEdge,
        is_manual: bool,
    ) -> Result<ProposedEdge> {
        if proposal.resulting_edge_id.is_some() {
            return Ok(proposal);
        }
        #[derive(Deserialize, SurrealValue)]
        struct RecoveryRow {
            id: RecordId,
        }
        let recovered: Option<RecoveryRow> = self
            .db
            .query(
                "UPDATE $id SET status = 'accepting', updated_at = time::now() WHERE status = 'accepted' AND resulting_edge_id IS NONE RETURN AFTER",
            )
            .bind(("id", id.clone()))
            .await?
            .take(0)?;
        if recovered.is_some() {
            let claimed = self
                .get_edge_proposal(id)
                .await?
                .ok_or_else(|| DbError::QueryFailed("acceptance recovery disappeared".into()))?;
            return self.resume_acceptance(id, claimed, is_manual).await;
        }
        let current = self
            .get_edge_proposal(id)
            .await?
            .ok_or_else(|| DbError::NotFound("proposed_edge".into(), record_id_to_string(id)))?;
        if current.status == ProposedEdgeStatus::Accepted && current.resulting_edge_id.is_some() {
            return Ok(current);
        }
        if current.status == ProposedEdgeStatus::Accepting {
            return self.resume_acceptance(id, current, is_manual).await;
        }
        Err(DbError::QueryFailed(format!(
            "proposal {} changed while acceptance recovery was being claimed",
            record_id_to_string(id)
        )))
    }

    async fn mark_claimed_proposal_stale(&self, id: &RecordId, reason: &str) -> Result<()> {
        self.db
            .query(
                "UPDATE $id SET status = 'superseded', superseded_at = time::now(), supersession_reason = $reason, resulting_edge_id = NONE, updated_at = time::now() WHERE status = 'accepting'",
            )
            .bind(("id", id.clone()))
            .bind(("reason", reason.to_string()))
            .await?
            .check()?;
        Ok(())
    }

    async fn mark_claimed_proposal_materialized(&self, id: &RecordId) -> Result<()> {
        self.db
            .query(
                "UPDATE $id SET status = 'superseded', superseded_at = time::now(), supersession_reason = 'equivalent edge already materialized independently', resulting_edge_id = NONE, updated_at = time::now() WHERE status = 'accepting'",
            )
            .bind(("id", id.clone()))
            .await?
            .check()?;
        Ok(())
    }

    async fn finalize_acceptance_claim(&self, id: &RecordId, edge_id: RecordId) -> Result<bool> {
        #[derive(Deserialize, SurrealValue)]
        struct FinalizedRow {
            id: RecordId,
        }
        let finalized: Option<FinalizedRow> = self
            .db
            .query(
                "UPDATE $id SET status = 'accepted', resulting_edge_id = $edge_id, updated_at = time::now() WHERE status = 'accepting' AND resulting_edge_id IS NONE RETURN AFTER",
            )
            .bind(("id", id.clone()))
            .bind(("edge_id", edge_id))
            .await?
            .take(0)?;
        Ok(finalized.is_some())
    }

    /// Accept all pending similarity proposals at or above a configured threshold.
    /// This is intentionally restricted to canonical `related_to` proposals.
    #[instrument(skip(self))]
    pub async fn accept_gardener_proposals_above(
        &self,
        min_confidence: f32,
        reviewer: Option<String>,
    ) -> Result<usize> {
        self.accept_gardener_proposals_above_with_audit(
            min_confidence,
            reviewer,
            "configured gardener auto-apply policy".into(),
            false,
        )
        .await
    }

    /// Accept every matching proposal with the supplied, auditable reviewer
    /// decision. Interactive/manual workflows must set `is_manual` to true;
    /// scheduled policy application uses the automatic default above.
    #[instrument(skip(self))]
    pub async fn accept_gardener_proposals_above_with_audit(
        &self,
        min_confidence: f32,
        reviewer: Option<String>,
        action_reason: String,
        is_manual: bool,
    ) -> Result<usize> {
        self.accept_gardener_proposals_above_in_pages(
            min_confidence,
            reviewer,
            action_reason,
            is_manual,
            250,
        )
        .await
    }

    /// Accept every matching pending proposal, using a stable record-id cursor
    /// so a large batch cannot silently stop at an arbitrary first page.
    pub(crate) async fn accept_gardener_proposals_above_in_pages(
        &self,
        min_confidence: f32,
        reviewer: Option<String>,
        action_reason: String,
        is_manual: bool,
        page_size: usize,
    ) -> Result<usize> {
        let page_size = page_size.max(1);
        let mut accepted = 0;
        let mut after_id = None;

        loop {
            let proposals = self
                .list_pending_gardener_proposals_page(min_confidence, after_id.clone(), page_size)
                .await?;
            if proposals.is_empty() {
                break;
            }
            let last_id = proposals
                .last()
                .and_then(|proposal| proposal.id.clone())
                .expect("stored proposal has id");

            for proposal in proposals {
                let id = proposal.id.expect("stored proposal has id");
                self.accept_edge_proposal(
                    &id,
                    reviewer.clone(),
                    Some(action_reason.clone()),
                    is_manual,
                )
                .await?;
                accepted += 1;
            }

            after_id = Some(last_id);
        }
        Ok(accepted)
    }

    async fn list_pending_gardener_proposals_page(
        &self,
        min_confidence: f32,
        after_id: Option<RecordId>,
        limit: usize,
    ) -> Result<Vec<ProposedEdge>> {
        let rows: Vec<ProposedEdgeRow> = self
            .db
            .query(format!(
                "{} WHERE (status = 'pending' OR status = 'accepting' OR (status = 'accepted' AND resulting_edge_id IS NONE)) AND edge_type = 'related_to' AND generator = 'gardener-similarity' AND confidence >= $min_confidence AND ($after_id = NONE OR id > $after_id) ORDER BY id ASC LIMIT $limit",
                proposal_select_sql("")
            ))
            .bind(("min_confidence", min_confidence.clamp(0.0, 1.0)))
            .bind(("after_id", after_id))
            .bind(("limit", limit.max(1)))
            .await?
            .take(0)?;
        rows.into_iter().map(ProposedEdgeRow::into_domain).collect()
    }

    /// Delete an accepted edge and mark its source proposal superseded. This is
    /// idempotent for a proposal that was already undone.
    #[instrument(skip(self))]
    pub async fn undo_edge(
        &self,
        edge_id: &RecordId,
        action_reason: Option<String>,
    ) -> Result<bool> {
        // Keep physical edge deletion and proposal audit retirement in the
        // same in-process lifecycle critical section as acceptance completion.
        // Otherwise an acceptance could finalize after this method deletes its
        // edge but before it supersedes the proposal, restoring a dangling
        // accepted state.
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        let table = edge_id.table.as_str();
        if !matches!(
            table,
            "supports" | "contradicts" | "derived_from" | "related_to"
        ) {
            return Err(DbError::QueryFailed(format!(
                "{} is not a note-edge record id",
                record_id_to_string(edge_id)
            )));
        }
        let reason = action_reason.unwrap_or_else(|| "accepted edge undone".into());
        let edge_existed = self.note_edge_exists(edge_id).await?;
        if edge_existed {
            self.db
                .query("DELETE $id")
                .bind(("id", edge_id.clone()))
                .await?
                .check()?;
        }
        // If a prior attempt deleted the edge but failed before this update,
        // a retry sees the absent edge and still repairs the proposal audit.
        let proposal_updated = self
            .supersede_proposal_for_undone_edge(edge_id, &reason)
            .await?;
        Ok(edge_existed || proposal_updated)
    }

    async fn supersede_proposal_for_undone_edge(
        &self,
        edge_id: &RecordId,
        reason: &str,
    ) -> Result<bool> {
        #[derive(Deserialize, SurrealValue)]
        struct UpdatedRow {
            id: RecordId,
        }
        let updated: Option<UpdatedRow> = self
            .db
            .query(
                "UPDATE proposed_edge SET status = 'superseded', superseded_at = time::now(), supersession_reason = $reason, resulting_edge_id = NONE, updated_at = time::now() WHERE resulting_edge_id = $id RETURN AFTER",
            )
            .bind(("id", edge_id.clone()))
            .bind(("reason", reason.to_string()))
            .await?
            .take(0)?;
        Ok(updated.is_some())
    }

    /// Return whether a supported note-edge record currently exists. This is
    /// intentionally read-only so CLI dry-runs can report their real outcome
    /// without relying on the mutating undo path.
    #[instrument(skip(self))]
    pub async fn note_edge_exists(&self, edge_id: &RecordId) -> Result<bool> {
        let table = edge_id.table.as_str();
        if !matches!(
            table,
            "supports" | "contradicts" | "derived_from" | "related_to"
        ) {
            return Err(DbError::QueryFailed(format!(
                "{} is not a note-edge record id",
                record_id_to_string(edge_id)
            )));
        }
        #[derive(Deserialize, SurrealValue)]
        struct ExistingEdge {
            id: RecordId,
        }
        let existing: Option<ExistingEdge> = self
            .db
            .query("SELECT id FROM $id")
            .bind(("id", edge_id.clone()))
            .await?
            .take(0)?;
        Ok(existing.is_some())
    }

    async fn find_proposal_by_dedupe_key(&self, dedupe_key: &str) -> Result<Option<ProposedEdge>> {
        let proposals: Vec<ProposedEdgeRow> = self
            .db
            .query(proposal_select_sql("WHERE dedupe_key = $dedupe_key"))
            .bind(("dedupe_key", dedupe_key.to_string()))
            .await?
            .take(0)?;
        proposals
            .into_iter()
            .next()
            .map(ProposedEdgeRow::into_domain)
            .transpose()
    }

    /// Proposal acceptance may only materialize relationships between notes
    /// that are currently visible to the corpus. This also makes a retry safe
    /// if a different process is interrupted after source promotion changes
    /// visibility but before it retires old-generation proposals.
    pub(crate) async fn note_is_visible(&self, id: &RecordId) -> Result<bool> {
        let existing: Option<Note> = self
            .db
            .query(format!(
                "SELECT * FROM note WHERE id = $id AND {VISIBLE_NOTE_CONDITION} LIMIT 1"
            ))
            .bind(("id", id.clone()))
            .await?
            .take(0)?;
        Ok(existing.is_some())
    }

    /// A graph/mention write may address a visible note or the source's
    /// current pending generation. Older hidden and failed generations are
    /// deliberately not writable, even when physical cleanup is deferred.
    pub(crate) async fn note_is_writable(&self, id: &RecordId) -> Result<bool> {
        let existing: Option<Note> = self
            .db
            .query(
                "SELECT * FROM note WHERE id = $id AND (source_id IS NONE OR source_generation IS NONE OR source_generation = source_id.successful_generation OR (source_generation = source_id.generation AND source_id.status = 'pending')) LIMIT 1",
            )
            .bind(("id", id.clone()))
            .await?
            .take(0)?;
        Ok(existing.is_some())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn create_audited_edge(
        &self,
        from_id: &RecordId,
        to_id: &RecordId,
        edge_type: EdgeType,
        confidence: Option<f32>,
        reason: Option<&str>,
        provenance: &str,
        proposal_id: Option<&RecordId>,
        is_manual: bool,
    ) -> Result<(RecordId, Option<RecordId>)> {
        validate_note_edge(from_id, to_id, &edge_type)?;
        let mut from_id = from_id.clone();
        let mut to_id = to_id.clone();
        canonicalize_note_edge(&mut from_id, &mut to_id, &edge_type);
        let table = note_edge_table(&edge_type)?;
        let dedupe_key = edge_dedupe_key(&from_id, &to_id, &edge_type);
        #[derive(Deserialize, SurrealValue)]
        struct IdRow {
            id: RecordId,
            #[serde(default)]
            proposal_id: Option<RecordId>,
        }
        let existing: Option<IdRow> = self
            .db
            .query(format!(
                "SELECT id, proposal_id FROM {table} WHERE dedupe_key = $dedupe_key LIMIT 1"
            ))
            .bind(("dedupe_key", dedupe_key.clone()))
            .await?
            .take(0)?;
        if let Some(existing) = existing {
            return Ok((existing.id, existing.proposal_id));
        }
        let insert = self.db.query(format!("INSERT INTO {table} (in, out, confidence, reason, provenance, proposal_id, is_manual, dedupe_key, created_at) VALUES ($from, $to, $confidence, $reason, $provenance, $proposal_id, $is_manual, $dedupe_key, time::now()) RETURN id"))
            .bind(("from", from_id)).bind(("to", to_id)).bind(("confidence", confidence.map(|value| value.clamp(0.0, 1.0))))
            .bind(("reason", reason.map(str::to_owned))).bind(("provenance", provenance.to_string())).bind(("proposal_id", proposal_id.cloned()))
            .bind(("is_manual", is_manual)).bind(("dedupe_key", dedupe_key.clone())).await;
        let created_result: Result<Vec<IdRow>> = match insert {
            Ok(mut response) => response.take(0).map_err(Into::into),
            Err(error) => Err(error.into()),
        };
        let created = match created_result {
            Ok(created) => created,
            Err(error) => {
                let existing: Option<IdRow> = self
                    .db
                    .query(format!(
                        "SELECT id, proposal_id FROM {table} WHERE dedupe_key = $dedupe_key LIMIT 1"
                    ))
                    .bind(("dedupe_key", dedupe_key))
                    .await?
                    .take(0)?;
                if let Some(existing) = existing {
                    return Ok((existing.id, existing.proposal_id));
                }
                return Err(error);
            }
        };
        created
            .into_iter()
            .next()
            .map(|row| (row.id, proposal_id.cloned()))
            .ok_or_else(|| DbError::QueryFailed(format!("create {table}")))
    }

    /// Get notes related to a given note (any direction)
    #[instrument(skip(self))]
    pub async fn get_related_notes(
        &self,
        note_id: &surrealdb::types::RecordId,
    ) -> Result<RelatedNotes> {
        let result: Vec<RelatedNotes> = self
            .db
            .query(format!(
                r#"
                SELECT 
                    (SELECT * FROM ->supports->note WHERE {VISIBLE_NOTE_CONDITION}) AS supporting,
                    (SELECT * FROM <-supports<-note WHERE {VISIBLE_NOTE_CONDITION}) AS supported_by,
                    (SELECT * FROM ->contradicts->note WHERE {VISIBLE_NOTE_CONDITION}) AS contradicting,
                    (SELECT * FROM <-contradicts<-note WHERE {VISIBLE_NOTE_CONDITION}) AS contradicted_by,
                    (SELECT * FROM ->related_to->note WHERE {VISIBLE_NOTE_CONDITION}) AS related,
                    (SELECT * FROM <-related_to<-note WHERE {VISIBLE_NOTE_CONDITION}) AS related_from
                FROM note
                WHERE id = $id AND {VISIBLE_NOTE_CONDITION}
            "#,
            ))
            .bind(("id", note_id.clone()))
            .await?
            .take(0)?;

        result
            .into_iter()
            .next()
            .ok_or_else(|| DbError::NotFound("note".into(), record_id_to_string(note_id)))
    }

    /// Find orphan notes (no connections)
    #[instrument(skip(self))]
    pub async fn find_orphan_notes(&self) -> Result<Vec<Note>> {
        let notes: Vec<Note> = self
            .db
            .query(format!(
                r#"
                SELECT * FROM note 
                WHERE 
                    {VISIBLE_NOTE_CONDITION} AND
                    array::len((SELECT * FROM ->supports->note WHERE {VISIBLE_NOTE_CONDITION})) = 0 AND
                    array::len((SELECT * FROM <-supports<-note WHERE {VISIBLE_NOTE_CONDITION})) = 0 AND
                    array::len((SELECT * FROM ->contradicts->note WHERE {VISIBLE_NOTE_CONDITION})) = 0 AND
                    array::len((SELECT * FROM <-contradicts<-note WHERE {VISIBLE_NOTE_CONDITION})) = 0 AND
                    array::len((SELECT * FROM ->related_to->note WHERE {VISIBLE_NOTE_CONDITION})) = 0 AND
                    array::len((SELECT * FROM <-related_to<-note WHERE {VISIBLE_NOTE_CONDITION})) = 0
            "#
            ))
            .await?
            .take(0)?;

        Ok(notes)
    }

    /// Find potentially related notes (for gardener suggestions)
    #[instrument(skip(self, embedding))]
    pub async fn find_similar_notes(
        &self,
        note_id: &str,
        embedding: Vec<f32>,
        threshold: f32,
        limit: usize,
    ) -> Result<Vec<SimilarNote>> {
        let results: Vec<SimilarNote> = self
            .db
            .query(format!(
                r#"
                SELECT
                    id,
                    title,
                    content,
                    vector::similarity::cosine(embedding, $embedding) AS similarity
                FROM note
                WHERE
                    {VISIBLE_NOTE_CONDITION} AND
                    id != $note_id AND
                    embedding IS NOT NONE AND
                    vector::similarity::cosine(embedding, $embedding) > $threshold
                ORDER BY similarity DESC
                LIMIT $limit
            "#
            ))
            .bind(("note_id", normalize_note_id(note_id)))
            .bind(("embedding", embedding))
            .bind(("threshold", threshold))
            .bind(("limit", limit))
            .await?
            .take(0)?;

        Ok(results)
    }

    // ==========================================
    // ENTITY OPERATIONS
    // ==========================================

    /// Create or get an entity by explicit scope or the legacy canonical key.
    #[instrument(skip(self))]
    pub async fn upsert_entity(&self, entity: Entity) -> Result<Entity> {
        let identity_key = entity.effective_identity_key();
        let Entity {
            id: _,
            entity_type,
            name,
            canonical_name,
            identity_key: _,
            embedding,
            metadata,
            created_at: _,
        } = entity;

        let result: Option<Entity> = self.db
            .query(r#"
                INSERT INTO entity (entity_type, name, canonical_name, identity_key, embedding, metadata, created_at)
                VALUES ($entity_type, $name, $canonical_name, $identity_key, $embedding, $metadata, time::now())
                ON DUPLICATE KEY UPDATE 
                    name = $name,
                    embedding = $embedding,
                    metadata = object::extend(
                        object::extend(metadata ?? {}, $metadata ?? {}),
                        {
                            extraction: IF $metadata.extraction = NONE THEN metadata.extraction ELSE object::extend($metadata.extraction, {
                                mention_spellings: array::distinct(array::concat(metadata.extraction.mention_spellings ?? [], $metadata.extraction.mention_spellings ?? [])),
                                alias_spellings: array::distinct(array::concat(metadata.extraction.alias_spellings ?? [], $metadata.extraction.alias_spellings ?? []))
                            }) END,
                            aliases: array::distinct(array::concat(
                                metadata.aliases ?? [],
                                $metadata.aliases ?? []
                            ))
                        }
                    )
            "#)
            .bind(("entity_type", entity_type.clone()))
            .bind(("name", name.clone()))
            .bind(("canonical_name", canonical_name.clone()))
            .bind(("identity_key", identity_key.clone()))
            .bind(("embedding", embedding.clone()))
            .bind(("metadata", metadata.clone()))
            .await?
            .take(0)?;

        if let Some(entity) = result {
            return Ok(entity);
        }

        // A label can legitimately identify several scoped rows.
        let fetched: Option<Entity> = self
            .db
            .query("SELECT * FROM entity WHERE identity_key = $identity_key LIMIT 1")
            .bind(("identity_key", identity_key))
            .await?
            .take(0)?;

        fetched.ok_or_else(|| DbError::CreateFailed("entity".into()))
    }

    /// Link a note to an entity
    #[instrument(skip(self))]
    pub async fn link_note_to_entity(
        &self,
        note_id: &surrealdb::types::RecordId,
        entity_id: &surrealdb::types::RecordId,
    ) -> Result<()> {
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        self.link_note_to_entity_locked(note_id, entity_id).await
    }

    /// Upsert extracted entities and attach the complete result set to a note
    /// while holding the source lifecycle lock.  Reconciliation snapshots
    /// dependent records under that same lock, so a concurrent import sees
    /// either the entire extraction result or none of it; it can never copy a
    /// prefix of a multi-entity extraction to a successor generation.
    #[instrument(skip(self, entities))]
    #[allow(clippy::mutable_key_type)] // Surreal `RecordId` is the database's canonical edge key.
    pub async fn upsert_entities_and_link_note(
        &self,
        note_id: &RecordId,
        entities: Vec<Entity>,
    ) -> Result<usize> {
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        if !self.note_is_writable(note_id).await? {
            return Err(DbError::NotFound(
                "note endpoint".into(),
                "an entity-link endpoint is hidden, failed, or no longer exists".into(),
            ));
        }

        // Preserve links that predate this batch: a rollback may only remove
        // records this call actually added, never a concurrent/manual link
        // that happened to target the same entity. The lifecycle lock keeps
        // this snapshot stable with respect to supported graph mutations.
        let existing_links: HashSet<RecordId> = self
            .db
            .query("SELECT VALUE out FROM mentions WHERE in = $note_id")
            .bind(("note_id", note_id.clone()))
            .await?
            .take(0)?;
        let mut created_links = Vec::new();
        let mut linked = 0;
        let result: Result<()> = async {
            for entity in entities {
                let entity = self.upsert_entity(entity).await?;
                let entity_id = entity.id.as_ref().ok_or_else(|| {
                    DbError::CreateFailed("upserted entity did not receive an id".into())
                })?;
                if !existing_links.contains(entity_id) {
                    self.link_note_to_entity_locked(note_id, entity_id).await?;
                    created_links.push(entity_id.clone());
                }
                linked += 1;
            }
            Ok(())
        }
        .await;

        if let Err(error) = result {
            for entity_id in created_links {
                self.db
                    .query("DELETE mentions WHERE in = $note_id AND out = $entity_id")
                    .bind(("note_id", note_id.clone()))
                    .bind(("entity_id", entity_id))
                    .await?;
            }
            return Err(error);
        }
        Ok(linked)
    }

    /// Replace a note's extracted entity mention set after inference has
    /// completed. The complete replacement is applied under the same source
    /// lifecycle lock as reconciliation, so a source refresh can snapshot
    /// either the old complete set or the new complete set, never the old
    /// delete/infer/insert gap.
    #[instrument(skip(self, entities))]
    #[allow(clippy::mutable_key_type)] // Deduplication must retain typed `RecordId`s for writes.
    pub async fn replace_note_entities(
        &self,
        note_id: &RecordId,
        entities: Vec<Entity>,
    ) -> Result<usize> {
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        if !self.note_is_writable(note_id).await? {
            return Err(DbError::NotFound(
                "note endpoint".into(),
                "an entity-replacement endpoint is hidden, failed, or no longer exists".into(),
            ));
        }

        // Complete all fallible inference-result persistence before replacing
        // mentions. A malformed entity therefore leaves the prior extraction
        // intact instead of clearing it first.
        let entity_ids = self.replacement_entity_ids(entities).await?;

        let previous_ids: Vec<RecordId> = self
            .db
            .query("SELECT VALUE out FROM mentions WHERE in = $note_id")
            .bind(("note_id", note_id.clone()))
            .await?
            .take(0)?;
        self.delete_mentions_for_note_locked(note_id).await?;

        let result: Result<()> = async {
            for entity_id in &entity_ids {
                self.link_note_to_entity_locked(note_id, entity_id).await?;
            }
            Ok(())
        }
        .await;
        if let Err(error) = result {
            // Restore the pre-replacement set if a database failure occurs
            // after deletion. The lock prevents source reconciliation from
            // observing the transient empty set.
            self.delete_mentions_for_note_locked(note_id).await?;
            for entity_id in previous_ids {
                self.link_note_to_entity_locked(note_id, &entity_id).await?;
            }
            return Err(error);
        }
        Ok(entity_ids.len())
    }

    pub(crate) async fn replacement_entity_ids(
        &self,
        entities: Vec<Entity>,
    ) -> Result<Vec<RecordId>> {
        let mut entity_ids = Vec::with_capacity(entities.len());
        let mut seen = HashSet::new();
        for entity in entities {
            let entity = self.upsert_entity(entity).await?;
            let entity_id = entity.id.ok_or_else(|| {
                DbError::CreateFailed("upserted entity did not receive an id".into())
            })?;
            if seen.insert(record_id_to_string(&entity_id)) {
                entity_ids.push(entity_id);
            }
        }
        Ok(entity_ids)
    }

    pub(crate) async fn link_note_to_entity_locked(
        &self,
        note_id: &surrealdb::types::RecordId,
        entity_id: &surrealdb::types::RecordId,
    ) -> Result<()> {
        if !self.note_is_writable(note_id).await? {
            return Err(DbError::NotFound(
                "note endpoint".into(),
                "a mention endpoint is hidden, failed, or no longer exists".into(),
            ));
        }
        #[derive(Deserialize, SurrealValue)]
        struct CountRow {
            count: Option<u64>,
        }

        let existing: Option<CountRow> = self
            .db
            .query(
                "SELECT count() FROM mentions WHERE in = $note_id AND out = $entity_id GROUP ALL",
            )
            .bind(("note_id", note_id.clone()))
            .bind(("entity_id", entity_id.clone()))
            .await?
            .take(0)?;

        let count = existing.and_then(|row| row.count).unwrap_or(0);
        if count == 0 {
            self.db
                .query("CREATE mentions SET in = $note_id, out = $entity_id")
                .bind(("note_id", note_id.clone()))
                .bind(("entity_id", entity_id.clone()))
                .await?;
        }

        Ok(())
    }

    /// Remove all mention links for a note
    #[instrument(skip(self))]
    pub async fn delete_mentions_for_note(
        &self,
        note_id: &surrealdb::types::RecordId,
    ) -> Result<()> {
        let _completion_guard = self.proposal_acceptance_lock.lock().await;
        self.delete_mentions_for_note_locked(note_id).await
    }

    async fn delete_mentions_for_note_locked(
        &self,
        note_id: &surrealdb::types::RecordId,
    ) -> Result<()> {
        if !self.note_is_writable(note_id).await? {
            return Err(DbError::NotFound(
                "note endpoint".into(),
                "a mention endpoint is hidden, failed, or no longer exists".into(),
            ));
        }
        self.db
            .query("DELETE mentions WHERE in = $note_id")
            .bind(("note_id", note_id.clone()))
            .await?;

        Ok(())
    }

    /// Get entities linked to a note
    #[instrument(skip(self))]
    pub async fn get_entities_for_note(&self, note_id: &str) -> Result<Vec<Entity>> {
        let raw = note_id.strip_prefix("note:").unwrap_or(note_id).to_string();
        let note_record_id = RecordId::new("note", raw);

        let entity_ids: Vec<RecordId> = self
            .db
            .query("SELECT VALUE out FROM mentions WHERE in = $note_id")
            .bind(("note_id", note_record_id))
            .await?
            .take(0)?;

        if entity_ids.is_empty() {
            return Ok(Vec::new());
        }

        let mut entities = Vec::with_capacity(entity_ids.len());
        for entity_id in entity_ids {
            let entity: Option<Entity> = self.db.select(entity_id).await?;
            if let Some(entity) = entity {
                entities.push(entity);
            }
        }

        Ok(entities)
    }

    /// Check whether a note has at least one linked entity matching the query.
    #[instrument(skip(self))]
    pub async fn note_has_entity_name(&self, note_id: &str, entity_query: &str) -> Result<bool> {
        #[derive(Deserialize, SurrealValue)]
        struct CountRow {
            #[serde(default)]
            count: Option<u64>,
        }

        let raw = note_id.strip_prefix("note:").unwrap_or(note_id).to_string();
        let note_record_id = RecordId::new("note", raw);
        let normalized = entity_query.trim().to_lowercase();

        if normalized.is_empty() {
            return Ok(true);
        }

        let existing: Option<CountRow> = self
            .db
            .query(
                r#"
                SELECT count() AS count
                FROM mentions
                WHERE in = $note_id
                  AND out IN (
                    SELECT VALUE id
                    FROM entity
                    WHERE canonical_name CONTAINS $entity_query
                  )
                GROUP ALL
            "#,
            )
            .bind(("note_id", note_record_id))
            .bind(("entity_query", normalized))
            .await?
            .take(0)?;

        let count = existing.and_then(|row| row.count).unwrap_or(0);
        Ok(count > 0)
    }

    /// Find a bounded, deterministic set of canonical entities that occur in
    /// the normalized query. Aliases are optional values stored in
    /// `entity.metadata.aliases`; this keeps query-time matching local and
    /// avoids requiring the extraction provider for search.
    #[instrument(skip(self))]
    pub async fn find_graph_entities(
        &self,
        normalized_query: &str,
        limit: usize,
    ) -> Result<Vec<GraphEntityMatch>> {
        let normalized_query = graph_query_normalize(normalized_query);
        if normalized_query.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let limit = i64::try_from(limit).map_err(|_| {
            DbError::QueryFailed("graph entity limit exceeds database integer range".into())
        })?;

        // Whole-query canonical/alias equality is always preferred over a
        // shorter entity phrase contained in the query. Only when neither
        // lexical tier produces a local entity do we consider typo-style
        // prefix seeds, preventing ordinary sentence words from crowding the
        // cap.
        let exact = self
            .query_graph_entities(&normalized_query, GraphEntityMatchTier::Exact, &[], limit)
            .await?;
        if !exact.is_empty() {
            return Ok(exact);
        }

        let phrases = self
            .query_graph_entities(
                &normalized_query,
                GraphEntityMatchTier::ContainedPhrase,
                &[],
                limit,
            )
            .await?;
        if !phrases.is_empty() {
            return Ok(phrases);
        }

        // Prefixes are a narrow recovery path for partial entity terms such
        // as `atla`. They require four characters and exclude common query
        // words, preserving the `ai`/`chair` boundary safety while avoiding
        // `what` -> `Whatever` false seeds.
        let prefixes = graph_prefix_terms(&normalized_query);
        if prefixes.is_empty() {
            return Ok(Vec::new());
        }
        self.query_graph_entities(
            &normalized_query,
            GraphEntityMatchTier::Prefix,
            &prefixes,
            limit,
        )
        .await
    }

    async fn query_graph_entities(
        &self,
        normalized_query: &str,
        tier: GraphEntityMatchTier,
        prefixes: &[String],
        limit: i64,
    ) -> Result<Vec<GraphEntityMatch>> {
        // Keep stored names and aliases on the exact same lexical boundary
        // contract as `graph_query_normalize`: punctuation becomes a space,
        // while Unicode letters and numbers remain terms. This lets `GPT-4`
        // be found from either a punctuated or a sentence query.
        let canonical_lexical = r#"string::trim(string::replace(string::lowercase(canonical_name), <regex>"[^\\p{L}\\p{N}]+", ' '))"#;
        let alias_lexical = r#"string::trim(string::replace(string::lowercase($alias), <regex>"[^\\p{L}\\p{N}]+", ' '))"#;
        let match_condition = match tier {
            GraphEntityMatchTier::Exact => format!(
                "{canonical_lexical} = $query OR array::any(metadata.aliases ?? [], |$alias| {alias_lexical} = $query)"
            ),
            GraphEntityMatchTier::ContainedPhrase => format!(
                "string::contains(string::concat(' ', $query, ' '), string::concat(' ', {canonical_lexical}, ' ')) OR array::any(metadata.aliases ?? [], |$alias| string::contains(string::concat(' ', $query, ' '), string::concat(' ', {alias_lexical}, ' ')))"
            ),
            GraphEntityMatchTier::Prefix => prefixes
                .iter()
                .enumerate()
                .map(|(index, _)| {
                    format!("string::starts_with({canonical_lexical}, $prefix_{index})")
                })
                .collect::<Vec<_>>()
                .join(" OR "),
        };
        let phrase_specificity = match tier {
            GraphEntityMatchTier::ContainedPhrase => {
                // A sentence can contain both `New` and `New York`. Rank by
                // the lexical phrase that actually matched (canonical or
                // alias), rather than the stored canonical name, so aliases
                // receive the same specificity treatment before the cap.
                let matching_alias_lengths = format!(
                    "array::map(array::filter(metadata.aliases ?? [], |$alias| string::contains(string::concat(' ', $query, ' '), string::concat(' ', {alias_lexical}, ' '))), |$alias| string::len({alias_lexical}))"
                );
                let phrase_specificity = format!(
                    "array::max([IF string::contains(string::concat(' ', $query, ' '), string::concat(' ', {canonical_lexical}, ' ')) THEN string::len({canonical_lexical}) ELSE 0 END, array::max({matching_alias_lengths})])"
                );
                Some(phrase_specificity)
            }
            GraphEntityMatchTier::Exact | GraphEntityMatchTier::Prefix => None,
        };
        let prefix_plausibility = match tier {
            GraphEntityMatchTier::Prefix => Some(format!(
                "array::min([{}])",
                prefixes
                    .iter()
                    .enumerate()
                    .map(|(index, prefix)| {
                        // Prefix recovery is for short, likely truncated
                        // entity fragments. Prefer the shortest matching
                        // query fragment so a context word such as
                        // `deployed` cannot crowd out `zeta` at a small cap.
                        format!(
                            "IF string::starts_with({canonical_lexical}, $prefix_{index}) THEN {} ELSE 2147483647 END",
                            prefix.chars().count()
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
            GraphEntityMatchTier::Exact | GraphEntityMatchTier::ContainedPhrase => None,
        };
        let select_specificity = phrase_specificity
            .as_ref()
            .map(|specificity| format!(", {specificity} AS graph_match_specificity"))
            .unwrap_or_default();
        let select_prefix_plausibility = prefix_plausibility
            .as_ref()
            .map(|plausibility| format!(", {plausibility} AS graph_prefix_plausibility"))
            .unwrap_or_default();
        let ordering = if phrase_specificity.is_some() {
            "graph_match_specificity DESC, canonical_name ASC, id ASC"
        } else if prefix_plausibility.is_some() {
            "graph_prefix_plausibility ASC, canonical_name ASC, id ASC"
        } else {
            "canonical_name ASC, id ASC"
        };
        let query = format!(
            r#"
                SELECT id, name, canonical_name, metadata{select_specificity}{select_prefix_plausibility}
                FROM entity
                WHERE {match_condition}
                ORDER BY {ordering}
                LIMIT $limit
                "#,
        );
        let mut query = self
            .db
            .query(query)
            .bind(("query", normalized_query.to_string()))
            .bind(("limit", limit));
        for (index, prefix) in prefixes.iter().enumerate() {
            query = query.bind((format!("prefix_{index}"), prefix.clone()));
        }
        query.await?.take(0).map_err(Into::into)
    }

    /// Fetch visible note IDs mentioned by any supplied entities in one
    /// query. The caller owns the cap, so an entity with a high degree cannot
    /// cause an unbounded graph seed set.
    #[instrument(skip(self, entity_ids))]
    pub async fn graph_notes_for_entities(
        &self,
        entity_ids: &[RecordId],
        limit: usize,
        since: Option<chrono::DateTime<chrono::Utc>>,
        source_uri: Option<String>,
    ) -> Result<Vec<GraphEntityNoteSeed>> {
        if entity_ids.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        // Query a bounded page for every ranked entity. A single global
        // relation `LIMIT` ordered by note ID lets a high-degree, lower-priority
        // entity starve later matches before the caller can allocate the
        // unique-note cap across them. `entity_ids` is itself bounded by graph
        // configuration, so this remains bounded while preserving coverage.
        let limit = i64::try_from(limit).map_err(|_| {
            DbError::QueryFailed("graph note limit exceeds database integer range".into())
        })?;
        let since = since.map(|timestamp| timestamp.to_rfc3339());
        let mut query = self
            .db
            .query(graph_mention_batch_sql(entity_ids.len()))
            .bind(("limit", limit))
            .bind(("since", since))
            .bind(("source_uri", source_uri));
        for (index, entity_id) in entity_ids.iter().enumerate() {
            query = query.bind((format!("entity_{index}"), entity_id.clone()));
        }
        let mut response = query.await?.check()?;
        let mut seeds = Vec::new();
        for index in 0..entity_ids.len() {
            let mut entity_seeds: Vec<GraphEntityNoteSeed> = response.take(index)?;
            seeds.append(&mut entity_seeds);
        }
        Ok(seeds)
    }

    /// Intersect query-ranked candidates with indexed entity mentions before
    /// the legacy ID page. Only bounded direct IDs are inspected; no note
    /// corpus scan or inference call is needed. Eligibility is checked again
    /// on endpoints, including current source generation.
    pub async fn graph_notes_for_entities_ranked(
        &self,
        entity_ids: &[RecordId],
        ranked_note_ids: &[RecordId],
        limit: usize,
        since: Option<chrono::DateTime<chrono::Utc>>,
        source_uri: Option<String>,
    ) -> Result<Vec<GraphEntityNoteSeed>> {
        if ranked_note_ids.is_empty() {
            return self
                .graph_notes_for_entities(entity_ids, limit, since, source_uri)
                .await;
        }
        if entity_ids.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let ranked_note_ids = ranked_note_ids
            .iter()
            .take(200)
            .cloned()
            .collect::<Vec<_>>();
        let limit = i64::try_from(limit).map_err(|_| {
            DbError::QueryFailed("graph note limit exceeds database integer range".into())
        })?;
        let eligible = graph_endpoint_eligible_sql("in");
        let mut sql = String::new();
        for index in 0..entity_ids.len() {
            sql.push_str(&format!(
                "SELECT in AS note_id, out AS entity_id FROM mentions WHERE out = $entity_{index} AND in IN $ranked_notes AND {eligible} ORDER BY in ASC LIMIT $ranked_limit;\
                 SELECT in AS note_id, out AS entity_id FROM mentions WHERE out = $entity_{index} AND in NOT IN $ranked_notes AND {eligible} ORDER BY in ASC LIMIT $limit;"
            ));
        }
        let ranks = ranked_note_ids
            .iter()
            .enumerate()
            .map(|(rank, id)| (record_id_to_string(id), rank))
            .collect::<HashMap<_, _>>();
        let mut query = self
            .db
            .query(sql)
            .bind(("ranked_limit", ranked_note_ids.len() as i64))
            .bind(("ranked_notes", ranked_note_ids))
            .bind(("limit", limit))
            .bind(("since", since.map(|value| value.to_rfc3339())))
            .bind(("source_uri", source_uri));
        for (index, id) in entity_ids.iter().enumerate() {
            query = query.bind((format!("entity_{index}"), id.clone()));
        }
        let mut response = query.await?.check()?;
        let mut seeds = Vec::new();
        for index in 0..entity_ids.len() {
            let mut preferred: Vec<GraphEntityNoteSeed> = response.take(index * 2)?;
            let fallback: Vec<GraphEntityNoteSeed> = response.take(index * 2 + 1)?;
            preferred.sort_by(|a, b| {
                ranks[&record_id_to_string(&a.note_id)]
                    .cmp(&ranks[&record_id_to_string(&b.note_id)])
                    .then_with(|| {
                        record_id_to_string(&a.note_id).cmp(&record_id_to_string(&b.note_id))
                    })
            });
            preferred.extend(fallback);
            preferred.truncate(limit as usize);
            seeds.extend(preferred);
        }
        Ok(seeds)
    }

    /// Fetch accepted persisted note-edge rows for many frontier notes.
    /// Every frontier note gets its own deterministic per-table budget; a
    /// high-degree early note cannot consume a shared table `LIMIT` and starve
    /// later seeds. Proposal tables are deliberately never consulted.
    #[instrument(skip(self, note_ids, edge_types))]
    #[allow(clippy::too_many_arguments)]
    pub async fn graph_note_edges(
        &self,
        note_ids: &[RecordId],
        edge_types: &[String],
        per_table_limit: usize,
        allow_outbound: bool,
        allow_inbound: bool,
        min_confidence: f32,
        since: Option<DateTime<Utc>>,
        source_uri: Option<String>,
    ) -> Result<Vec<NoteEdgeRow>> {
        self.graph_note_edges_excluding_visited(
            note_ids,
            edge_types,
            per_table_limit,
            allow_outbound,
            allow_inbound,
            min_confidence,
            since,
            source_uri,
            &HashMap::new(),
        )
        .await
    }

    /// Fetch bounded graph edges while excluding endpoint notes already on the
    /// path for each current source. Applying this exclusion in the database
    /// is necessary: otherwise a high-confidence back-edge can consume the
    /// per-source table limit before traversal rejects the cycle in memory.
    #[allow(clippy::too_many_arguments)]
    #[instrument(skip(self, note_ids, edge_types, visited_note_ids))]
    pub async fn graph_note_edges_excluding_visited(
        &self,
        note_ids: &[RecordId],
        edge_types: &[String],
        per_table_limit: usize,
        allow_outbound: bool,
        allow_inbound: bool,
        min_confidence: f32,
        since: Option<DateTime<Utc>>,
        source_uri: Option<String>,
        visited_note_ids: &HashMap<String, Vec<RecordId>>,
    ) -> Result<Vec<NoteEdgeRow>> {
        if note_ids.is_empty()
            || edge_types.is_empty()
            || per_table_limit == 0
            || (!allow_outbound && !allow_inbound)
        {
            return Ok(Vec::new());
        }
        let tables = ["supports", "contradicts", "related_to", "derived_from"]
            .into_iter()
            .filter(|table| edge_types.iter().any(|edge_type| edge_type == table))
            .collect::<Vec<_>>();
        if tables.is_empty() {
            return Ok(Vec::new());
        }
        let limit = i64::try_from(per_table_limit).map_err(|_| {
            DbError::QueryFailed("graph edge limit exceeds database integer range".into())
        })?;
        // Fetch only referenced endpoint documents across every table/source,
        // while keeping each source/table's LIMIT independent. A global edge
        // LIMIT would let high-degree early seeds starve later seeds.
        let sql = graph_edge_batch_sql(note_ids.len(), &tables, allow_outbound, allow_inbound);
        let mut query = self
            .db
            .query(sql)
            .bind(("limit", limit))
            .bind(("min_confidence", min_confidence))
            .bind(("since", since.map(|timestamp| timestamp.to_rfc3339())))
            .bind(("source_uri", source_uri));
        for (index, note_id) in note_ids.iter().enumerate() {
            query = query.bind((format!("note_{index}"), note_id.clone()));
            query = query.bind((
                format!("visited_{index}"),
                visited_note_ids
                    .get(&record_id_to_string(note_id))
                    .cloned()
                    .unwrap_or_default(),
            ));
        }
        let mut response = query.await?.check()?;
        let mut rows = HashMap::<String, NoteEdgeRow>::new();
        for index in 0..tables.len() * note_ids.len() {
            let edges: Vec<NoteEdgeRow> = response.take(index)?;
            for edge in edges {
                rows.entry(record_id_to_string(&edge.id)).or_insert(edge);
            }
        }
        let mut rows = rows.into_values().collect::<Vec<_>>();
        rows.sort_by(|left, right| {
            left.edge_type
                .cmp(&right.edge_type)
                .then_with(|| record_id_to_string(&left.id).cmp(&record_id_to_string(&right.id)))
        });
        Ok(rows)
    }

    /// Load graph-selected notes in one visibility-aware query so deleted or
    /// superseded endpoints are silently excluded from retrieval.
    #[instrument(skip(self, note_ids))]
    pub async fn graph_notes_by_ids(
        &self,
        note_ids: &[RecordId],
        since: Option<chrono::DateTime<chrono::Utc>>,
        source_uri: Option<String>,
    ) -> Result<Vec<SearchResult>> {
        if note_ids.is_empty() {
            return Ok(Vec::new());
        }
        // Selecting the bound records avoids scanning all notes to hydrate a
        // small graph candidate set. Deduplicate IDs and exclude other tables
        // and range selectors, preserving the former note-table membership
        // query rather than widening a caller's selection.
        let mut seen = HashSet::new();
        let note_ids = note_ids
            .iter()
            .filter(|id| id.table.as_str() == "note" && !id.key.is_range())
            .filter(|id| seen.insert(id.to_sql()))
            .cloned()
            .collect::<Vec<_>>();
        if note_ids.is_empty() {
            return Ok(Vec::new());
        }
        let since = since.map(|timestamp| timestamp.to_rfc3339());
        self.db
            .query(format!(
                "SELECT id, title, content, note_type, tags, created_at, source_id.uri AS source_uri \
                 FROM $note_ids WHERE ($since = NONE OR created_at >= <datetime>$since) \
                 AND ($source_uri = NONE OR source_id.uri = $source_uri) AND {VISIBLE_NOTE_CONDITION} \
                 ORDER BY id ASC"
            ))
            .bind(("note_ids", note_ids))
            .bind(("since", since))
            .bind(("source_uri", source_uri))
            .await?
            .take(0)
            .map_err(Into::into)
    }

    /// Return the original chat provenance record IDs for graph-selected
    /// notes in two bounded set queries. File-backed notes retain their
    /// source URI in [`SearchResult`]; chat-derived notes need these record
    /// IDs to keep an augmentation citation reconstructable without loading
    /// each note individually.
    #[instrument(skip(self, note_ids))]
    pub async fn graph_note_provenance_ids(
        &self,
        note_ids: &[RecordId],
    ) -> Result<HashMap<String, Vec<String>>> {
        #[derive(Deserialize, SurrealValue)]
        struct ProvenanceRow {
            r#in: RecordId,
            out: RecordId,
        }

        if note_ids.is_empty() {
            return Ok(HashMap::new());
        }
        let mut response = self
            .db
            .query(
                "SELECT in, out FROM note_from_conversation WHERE in IN $note_ids; \
                 SELECT in, out FROM note_from_message WHERE in IN $note_ids;",
            )
            .bind(("note_ids", note_ids.to_vec()))
            .await?
            .check()?;
        let mut provenance = HashMap::<String, Vec<String>>::new();
        for index in 0..2 {
            let rows: Vec<ProvenanceRow> = response.take(index)?;
            for row in rows {
                provenance
                    .entry(record_id_to_string(&row.r#in))
                    .or_default()
                    .push(record_id_to_string(&row.out));
            }
        }
        for ids in provenance.values_mut() {
            ids.sort();
            ids.dedup();
        }
        Ok(provenance)
    }

    /// List note-to-note edges across all edge tables
    #[instrument(skip(self))]
    pub async fn list_note_edges(&self, limit: usize) -> Result<Vec<NoteEdgeRow>> {
        let mut edges: Vec<NoteEdgeRow> = Vec::new();
        let limit = limit.max(1);

        edges.extend(self.query_edges_table("supports", limit).await?);
        edges.extend(self.query_edges_table("contradicts", limit).await?);
        edges.extend(self.query_edges_table("related_to", limit).await?);
        edges.extend(self.query_edges_table("derived_from", limit).await?);

        Ok(edges)
    }

    /// Get note-to-note edges for a specific note id (in or out)
    #[instrument(skip(self))]
    pub async fn get_note_edges(&self, note_id: &str) -> Result<Vec<NoteEdgeRow>> {
        let note_id = normalize_note_id(note_id);
        let mut edges: Vec<NoteEdgeRow> = Vec::new();

        edges.extend(self.query_edges_for_note("supports", &note_id).await?);
        edges.extend(self.query_edges_for_note("contradicts", &note_id).await?);
        edges.extend(self.query_edges_for_note("related_to", &note_id).await?);
        edges.extend(self.query_edges_for_note("derived_from", &note_id).await?);

        Ok(edges)
    }
}

fn proposal_select_sql(where_clause: &str) -> String {
    format!(
        "SELECT id, dedupe_key, in AS from_id, out AS to_id, edge_type, confidence, reason, generator, generator_version, model, status, created_at, updated_at, reviewed_at, reviewer, action_reason, acceptance_is_manual, resulting_edge_id, superseded_at, supersession_reason FROM proposed_edge {where_clause}"
    )
}

pub(super) fn note_edge_table(edge_type: &EdgeType) -> Result<&'static str> {
    match edge_type {
        EdgeType::Supports => Ok("supports"),
        EdgeType::Contradicts => Ok("contradicts"),
        EdgeType::DerivedFrom => Ok("derived_from"),
        EdgeType::RelatedTo => Ok("related_to"),
        EdgeType::References | EdgeType::Mentions | EdgeType::TaggedWith => {
            Err(DbError::QueryFailed(format!(
                "{edge_type} is not a persisted note-to-note edge type"
            )))
        }
    }
}

pub(crate) fn persisted_note_edge_type(value: &str) -> Result<EdgeType> {
    match value {
        "supports" => Ok(EdgeType::Supports),
        "contradicts" => Ok(EdgeType::Contradicts),
        "derived_from" => Ok(EdgeType::DerivedFrom),
        "related_to" => Ok(EdgeType::RelatedTo),
        other => Err(DbError::QueryFailed(format!(
            "unknown persisted note edge type {other:?}"
        ))),
    }
}

fn validate_note_edge(from_id: &RecordId, to_id: &RecordId, edge_type: &EdgeType) -> Result<()> {
    if !edge_type.is_note_edge() || matches!(edge_type, EdgeType::References) {
        return Err(DbError::QueryFailed(format!(
            "{edge_type} is not supported for persisted note edges"
        )));
    }
    if from_id.table.as_str() != "note" || to_id.table.as_str() != "note" {
        return Err(DbError::QueryFailed(
            "note edges require note record ids".into(),
        ));
    }
    if from_id == to_id {
        return Err(DbError::QueryFailed("self-edges are not allowed".into()));
    }
    Ok(())
}

pub(crate) fn canonicalize_note_edge(
    from_id: &mut RecordId,
    to_id: &mut RecordId,
    edge_type: &EdgeType,
) {
    if edge_type.is_symmetric() && record_id_to_string(from_id) > record_id_to_string(to_id) {
        std::mem::swap(from_id, to_id);
    }
}

pub(crate) fn edge_dedupe_key(
    from_id: &RecordId,
    to_id: &RecordId,
    edge_type: &EdgeType,
) -> String {
    format!(
        "{}:{}:{}",
        edge_type,
        record_id_to_string(from_id),
        record_id_to_string(to_id)
    )
}

impl Repository {
    async fn query_edges_table(&self, table: &str, limit: usize) -> Result<Vec<NoteEdgeRow>> {
        let query = format!(
            "SELECT id, '{table}' AS edge_type, in AS in_id, out AS out_id, proposal_id, confidence, reason, provenance, is_manual, created_at \
             FROM {table} WHERE {VISIBLE_NOTE_EDGE_ENDPOINTS_CONDITION} LIMIT $limit"
        );
        let edges: Vec<NoteEdgeRow> = self
            .db
            .query(&query)
            .bind(("limit", limit))
            .await?
            .take(0)?;
        Ok(edges)
    }

    async fn query_edges_for_note(
        &self,
        table: &str,
        note_id: &RecordId,
    ) -> Result<Vec<NoteEdgeRow>> {
        let query = format!(
            "SELECT id, '{table}' AS edge_type, in AS in_id, out AS out_id, proposal_id, confidence, reason, provenance, is_manual, created_at \
             FROM {table} WHERE (in = $note_id OR out = $note_id) \
             AND {VISIBLE_NOTE_EDGE_ENDPOINTS_CONDITION}"
        );
        let edges: Vec<NoteEdgeRow> = self
            .db
            .query(&query)
            .bind(("note_id", note_id.clone()))
            .await?
            .take(0)?;
        Ok(edges)
    }
}

// ==========================================
// RESULT TYPES
// ==========================================

/// Decision made before a file import starts.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SourceImportAction {
    Created,
    Updated,
    Unchanged,
}

/// Source state returned by the staged file-import API.
#[derive(Debug, Clone)]
pub struct SourceImportPlan {
    pub source: Source,
    pub action: SourceImportAction,
    /// Deletions recovered before returning an unchanged import decision.
    /// Other actions defer cleanup until their import successfully promotes.
    pub cleanup: SourceDeleteSummary,
}

/// Deterministic cascade counts used by dry-run and completed import output.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceDeleteSummary {
    pub notes: u64,
    pub mentions: u64,
    pub note_edges: u64,
    /// Pending, accepting, or accepted proposals transitioned to `superseded`.
    pub proposals: u64,
    pub note_conversation_provenance: u64,
    pub note_message_provenance: u64,
}

#[derive(Debug, Deserialize, SurrealValue)]
pub(crate) struct SourceDeleteCount {
    pub(crate) kind: String,
    #[serde(default)]
    pub(crate) count: u64,
}
