use crate::inference::{CancellableEmbedder, CancellableExtractor};
use crate::*;
use async_trait::async_trait;
use graphrag_agents::{
    AugmentOptions, GraphMode, LibrarianAgent, LibrarianRuntimeConfig, SearchAgent, SearchHitType,
    SearchScope, SharedEmbedder, SharedEntityExtractor,
};
use graphrag_core::{
    record_id_to_string, Note, ProposedEdge, ProposedEdgeStatus, SourceIngestionStatus,
};
use graphrag_db::{parse_record_id, repository::DbStats, DbError, RecordInspection, Repository};
use sha2::{Digest, Sha256};
use std::sync::Arc;

/// An adapter over resources already owned by the calling process.
pub struct EmbeddedApplication {
    pub(crate) repo: Repository,
    pub(crate) search: SearchAgent,
    pub(crate) embedder: SharedEmbedder,
    pub(crate) extractor: SharedEntityExtractor,
    pub(crate) runtime: LibrarianRuntimeConfig,
    pub(crate) augment_options: AugmentOptions,
}

impl EmbeddedApplication {
    pub fn new(
        repo: Repository,
        search: SearchAgent,
        embedder: SharedEmbedder,
        extractor: SharedEntityExtractor,
        runtime: LibrarianRuntimeConfig,
    ) -> Self {
        Self {
            repo,
            search,
            embedder,
            extractor,
            runtime,
            augment_options: AugmentOptions::default(),
        }
    }

    pub fn with_augment_options(mut self, options: AugmentOptions) -> Self {
        self.augment_options = options;
        self
    }

    pub(crate) async fn require_providers(
        &self,
        extraction: bool,
        cancel: &ActionCancellation,
    ) -> ApplicationResult<()> {
        let available = tokio::select! {
            biased;
            _ = cancel.cancelled() => return Err(ApplicationError::Cancelled),
            result = self.embedder.health() => result.unwrap_or(false),
        };
        if !available {
            return Err(ApplicationError::ProviderUnavailable(
                "Embeddings service unavailable. Use keyword search to browse offline.".into(),
            ));
        }
        if extraction {
            let available = tokio::select! {
                biased;
                _ = cancel.cancelled() => return Err(ApplicationError::Cancelled),
                result = self.extractor.health() => result.unwrap_or(false),
            };
            if !available {
                return Err(ApplicationError::ProviderUnavailable(
                    "Extraction service unavailable; the capture draft was retained.".into(),
                ));
            }
        }
        Ok(())
    }

    async fn summary(
        &self,
        id: String,
        hit_type: String,
        title: Option<String>,
        content: String,
        score: Option<f32>,
    ) -> RecordSummary {
        let inspected = self.repo.inspect_record(&id, 0).await;
        let (revision, provenance, warnings) = match inspected {
            Ok(record) if record.content == content =>
                (Some(record.revision), Some(record.provenance), record.warnings),
            _ => (None, None, vec![
                "Record changed or became unavailable. Run search again before inspecting or opening it.".into(),
            ]),
        };
        RecordSummary {
            id,
            hit_type,
            title,
            content,
            revision,
            provenance,
            score,
            warnings,
        }
    }

    pub async fn proposal_card(&self, proposal: ProposedEdge) -> ApplicationResult<ProposalCard> {
        proposal_card(&self.repo, proposal).await
    }
}

async fn endpoint(repo: &Repository, id: String) -> ApplicationResult<ProposalEndpoint> {
    match repo.inspect_record(&id, 0).await {
        Ok(record) => {
            let mut characters = record.content.chars();
            let mut excerpt = characters.by_ref().take(500).collect::<String>();
            if characters.next().is_some() {
                excerpt.pop();
                excerpt.push('…');
            }
            Ok(ProposalEndpoint {
                id,
                available: true,
                title: record.title,
                excerpt: Some(excerpt),
                revision: Some(record.revision),
                provenance: Some(record.provenance),
                warnings: record.warnings,
            })
        }
        Err(DbError::NotFound(..)) => Ok(ProposalEndpoint {
            id,
            available: false,
            title: None,
            excerpt: None,
            revision: None,
            provenance: None,
            warnings: vec![
                "Note is missing or belongs to an unavailable source generation.".into(),
            ],
        }),
        Err(error) => Err(error.into()),
    }
}

pub async fn proposal_card(
    repo: &Repository,
    proposal: ProposedEdge,
) -> ApplicationResult<ProposalCard> {
    let id = proposal
        .id
        .as_ref()
        .ok_or_else(|| ApplicationError::Internal("Stored proposal has no ID".into()))?;
    let from = endpoint(repo, record_id_to_string(&proposal.from_id)).await?;
    let to = endpoint(repo, record_id_to_string(&proposal.to_id)).await?;
    let blocked = if proposal.status != ProposedEdgeStatus::Pending {
        Some(format!("Proposal is {}, not pending.", proposal.status))
    } else if !from.available || !to.available {
        Some("Both notes must be visible before this proposal can be accepted.".into())
    } else {
        None
    };
    let revision = proposal_revision(&proposal, from.revision.as_deref(), to.revision.as_deref())?;
    Ok(ProposalCard {
        id: record_id_to_string(id),
        status: proposal.status,
        edge_type: proposal.edge_type.to_string(),
        confidence: proposal.confidence,
        reason: proposal.reason,
        generator: proposal.generator,
        generator_version: proposal.generator_version,
        model: proposal.model,
        updated_at: proposal.updated_at.to_rfc3339(),
        reviewed_at: proposal.reviewed_at.map(|date| date.to_rfc3339()),
        reviewer: proposal.reviewer,
        action_reason: proposal.action_reason,
        acceptance_is_manual: proposal.acceptance_is_manual,
        supersession_reason: proposal.supersession_reason,
        resulting_edge_id: proposal.resulting_edge_id.as_ref().map(record_id_to_string),
        from,
        to,
        accept_allowed: blocked.is_none(),
        accept_blocked_reason: blocked,
        revision,
    })
}
pub fn validate_record_id(id: &str) -> ApplicationResult<()> {
    if !id.split_once(':').is_some_and(|(table, key)| {
        matches!(table, "note" | "message" | "conversation")
            && !key.trim().is_empty()
            && !id.chars().any(char::is_control)
    }) {
        return Err(ApplicationError::Validation(
            "Use the full note:ID, message:ID, or conversation:ID printed by search; result numbers are display positions, not record IDs.".into(),
        ));
    }
    Ok(())
}

pub fn validate_neighbors(neighbors: usize) -> ApplicationResult<()> {
    if neighbors > graphrag_db::MAX_INSPECTION_NEIGHBORS {
        return Err(ApplicationError::Validation(
            "--neighbors must be between 0 and 20.".into(),
        ));
    }
    Ok(())
}

pub fn validate_record_revision(
    record: &RecordInspection,
    expected: Option<&str>,
) -> ApplicationResult<()> {
    if expected.is_some_and(|revision| revision != record.revision) {
        return Err(ApplicationError::RevisionConflict(format!(
            "The record {} changed since this result was displayed (revision mismatch). Run search again and use its new inspection command.", record.id,
        )));
    }
    Ok(())
}

/// Same proposal/endpoint fingerprint used by the existing review inbox.
pub fn proposal_revision(
    proposal: &ProposedEdge,
    from_revision: Option<&str>,
    to_revision: Option<&str>,
) -> ApplicationResult<String> {
    let bytes = serde_json::to_vec(&(proposal, from_revision, to_revision))
        .map_err(|error| ApplicationError::Internal(error.to_string()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn validate_limit(limit: usize) -> ApplicationResult<()> {
    if !(1..=MAX_RESULT_LIMIT).contains(&limit) {
        return Err(ApplicationError::Validation(
            "limit must be between 1 and 200".into(),
        ));
    }
    Ok(())
}

fn proposal_id(id: &str) -> ApplicationResult<surrealdb::types::RecordId> {
    if !id.starts_with("proposed_edge:") || id.chars().any(char::is_control) {
        return Err(ApplicationError::Validation(
            "review requires a full proposed_edge:ID".into(),
        ));
    }
    parse_record_id(id, Some("proposed_edge"))
        .map_err(|error| ApplicationError::Validation(error.to_string()))
}

#[async_trait]
impl ApplicationOperations for EmbeddedApplication {
    async fn search(
        &self,
        request: SearchRequest,
        cancel: ActionCancellation,
    ) -> ApplicationResult<Vec<RecordSummary>> {
        validate_limit(request.limit)?;
        if request.query.trim().is_empty() {
            return Err(ApplicationError::Validation(
                "search query cannot be empty".into(),
            ));
        }
        if request.mode == SearchMode::Keyword && request.graph == GraphPolicy::On {
            return Err(ApplicationError::Validation(
                "Keyword search requires graph off; select hybrid for graph expansion.".into(),
            ));
        }
        if cancel.is_cancelled() {
            return Err(ApplicationError::Cancelled);
        }
        if request.mode == SearchMode::Hybrid {
            self.require_providers(false, &cancel).await?;
        }
        let scope = match request.scope {
            Scope::Notes => SearchScope::Notes,
            Scope::Messages => SearchScope::Messages,
            Scope::All => SearchScope::All,
        };
        let graph = match request.graph {
            GraphPolicy::Off => GraphMode::Off,
            GraphPolicy::Auto => GraphMode::Auto,
            GraphPolicy::On => GraphMode::On,
        };
        let retrieval = async {
            if request.mode == SearchMode::Keyword {
                self.search
                    .keyword_search_with_scope(
                        &request.query,
                        request.limit,
                        scope,
                        request.since_days,
                        request.source_uri,
                    )
                    .await
            } else {
                self.search
                    .search_with_scope_graph(
                        &request.query,
                        request.limit,
                        scope,
                        request.since_days,
                        request.source_uri,
                        graph,
                    )
                    .await
            }
        };
        let results = tokio::select! {
            biased;
            _ = cancel.cancelled() => return Err(ApplicationError::Cancelled),
            result = retrieval => result?,
        };
        let mut summaries = Vec::with_capacity(results.hits.len());
        for hit in results.hits {
            if cancel.is_cancelled() {
                return Err(ApplicationError::Cancelled);
            }
            let kind = match hit.hit_type {
                SearchHitType::Note => "note",
                SearchHitType::Message => "message",
                SearchHitType::ConversationSummary => "conversation_summary",
            };
            summaries.push(
                self.summary(hit.id, kind.into(), hit.title, hit.content, Some(hit.score))
                    .await,
            );
        }
        Ok(summaries)
    }

    async fn recent(&self, limit: usize) -> ApplicationResult<Vec<RecordSummary>> {
        validate_limit(limit)?;
        let notes = self.repo.list_notes(limit).await?;
        let mut summaries = Vec::with_capacity(notes.len());
        for note in notes {
            summaries.push(
                self.summary(
                    record_id_to_string(&note.id),
                    "note".into(),
                    note.title,
                    note.content,
                    None,
                )
                .await,
            );
        }
        Ok(summaries)
    }

    async fn inspect(
        &self,
        reference: RecordRef,
        neighbors: usize,
    ) -> ApplicationResult<RecordInspection> {
        validate_record_id(&reference.id)?;
        validate_neighbors(neighbors)?;
        let record = self.repo.inspect_record(&reference.id, neighbors).await?;
        validate_record_revision(&record, reference.revision.as_deref())?;
        Ok(record)
    }

    async fn capture(
        &self,
        request: CaptureRequest,
        cancel: ActionCancellation,
    ) -> ApplicationResult<Note> {
        if request.content.trim().is_empty() {
            return Err(ApplicationError::Validation(
                "note content cannot be empty".into(),
            ));
        }
        self.require_providers(!self.runtime.skip_entity_extraction, &cancel)
            .await?;
        if cancel.is_cancelled() {
            return Err(ApplicationError::Cancelled);
        }
        let librarian = LibrarianAgent::new(
            self.repo.clone(),
            Arc::new(CancellableEmbedder(self.embedder.clone(), cancel.clone())),
            Arc::new(CancellableExtractor(self.extractor.clone(), cancel.clone())),
        )
        .with_runtime_config(self.runtime.clone())
        .with_cancellation_flag(cancel.flag());
        // Only the provider wrappers are cancellation-selectable. Await the
        // whole librarian call so an atomic transaction is never abandoned.
        Ok(librarian
            .capture_manual_note(request.content, request.title, request.tags)
            .await?)
    }

    async fn source_status(&self, limit: usize) -> ApplicationResult<Vec<SourceStatus>> {
        validate_limit(limit)?;
        self.repo
            .list_sources()
            .await?
            .into_iter()
            .take(limit)
            .map(|source| {
                let id = source
                    .id
                    .as_ref()
                    .ok_or_else(|| ApplicationError::Internal("Stored source has no ID".into()))?;
                let status = match source.status {
                    SourceIngestionStatus::Pending => "pending",
                    SourceIngestionStatus::Ready => "ready",
                    SourceIngestionStatus::Failed => "failed",
                };
                Ok(SourceStatus {
                    id: record_id_to_string(id),
                    title: source.title,
                    uri: source.normalized_uri.or(source.uri),
                    status: status.into(),
                    generation: source.generation,
                    successful_generation: source.successful_generation,
                    last_error: source.last_error,
                })
            })
            .collect()
    }

    async fn proposals(
        &self,
        status: Option<ProposedEdgeStatus>,
        limit: usize,
    ) -> ApplicationResult<Vec<ProposalCard>> {
        validate_limit(limit)?;
        let proposals = self.repo.list_edge_proposals(status, limit).await?;
        let mut cards = Vec::with_capacity(proposals.len());
        for proposal in proposals {
            cards.push(self.proposal_card(proposal).await?);
        }
        Ok(cards)
    }

    async fn proposal(&self, id: &str) -> ApplicationResult<ProposalCard> {
        let parsed = proposal_id(id)?;
        let proposal = self
            .repo
            .get_edge_proposal(&parsed)
            .await?
            .ok_or_else(|| ApplicationError::NotFound(format!("Proposal {id} not found")))?;
        self.proposal_card(proposal).await
    }

    async fn decide_proposal(
        &self,
        request: ProposalDecisionRequest,
    ) -> ApplicationResult<ProposalCard> {
        decide_proposal(&self.repo, request).await
    }

    async fn stats(&self) -> ApplicationResult<DbStats> {
        Ok(self.repo.get_stats().await?)
    }
}

pub async fn decide_proposal(
    repo: &Repository,
    request: ProposalDecisionRequest,
) -> ApplicationResult<ProposalCard> {
    if !request.confirmed {
        return Err(ApplicationError::Validation(
            "Confirm this proposal decision explicitly before applying it.".into(),
        ));
    }
    if request.reviewer.trim().is_empty() {
        return Err(ApplicationError::Validation(
            "A trusted adapter reviewer identity is required.".into(),
        ));
    }
    let id = proposal_id(&request.id)?;
    let proposal = repo
        .get_edge_proposal(&id)
        .await?
        .ok_or_else(|| ApplicationError::NotFound(format!("Proposal {} not found", request.id)))?;
    let current = proposal_card(repo, proposal).await?;
    if current.revision != request.revision {
        return Err(ApplicationError::RevisionConflict("Proposal or a note changed during review; review the updated card before deciding again.".into()));
    }
    match request.action {
        ProposalAction::Accept => {
            if !current.accept_allowed {
                return Err(ApplicationError::Validation(
                    current
                        .accept_blocked_reason
                        .unwrap_or_else(|| "Proposal unavailable".into()),
                ));
            }
            repo.accept_edge_proposal(&id, Some(request.reviewer), request.reason, true)
                .await?;
        }
        ProposalAction::Reject => {
            if current.status != ProposedEdgeStatus::Pending {
                return Err(ApplicationError::Validation(format!(
                    "Proposal is {}, not pending",
                    current.status
                )));
            }
            repo.reject_edge_proposal(&id, Some(request.reviewer), request.reason)
                .await?;
        }
        ProposalAction::Undo => {
            if current.status != ProposedEdgeStatus::Accepted {
                return Err(ApplicationError::Validation(format!(
                    "Proposal is {}, not accepted",
                    current.status
                )));
            }
            let edge = current.resulting_edge_id.ok_or_else(|| {
                ApplicationError::Validation(
                    "Accepted proposal has no recorded edge; recover acceptance before undoing it."
                        .into(),
                )
            })?;
            let edge = parse_record_id(&edge, None)
                .map_err(|error| ApplicationError::Validation(error.to_string()))?;
            repo.undo_edge(
                &edge,
                Some(
                    request.reason.unwrap_or_else(|| {
                        "accepted edge undone through interactive review".into()
                    }),
                ),
            )
            .await?;
        }
    }
    let proposal = repo
        .get_edge_proposal(&id)
        .await?
        .ok_or_else(|| ApplicationError::Internal("reviewed proposal disappeared".into()))?;
    proposal_card(repo, proposal).await
}
