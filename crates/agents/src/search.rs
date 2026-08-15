//! Search Agent - Handles user queries with hybrid search

use crate::{inference::validate_embedding_dim, Result, SharedEmbedder};
use chrono::{Duration, Utc};
use graphrag_core::{record_id_to_string, Entity};
use graphrag_db::compatibility::EmbeddingIdentity;
use graphrag_db::repository::{
    ConversationSearchResult, MessageSearchResult, NoteEdgeRow, RelatedNotes, SearchResult,
    SimilarNote,
};
use graphrag_db::{
    fusion::{self, FusionConfig, FusionEvidence},
    Repository,
};

use serde::Serialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use surrealdb::types::RecordId;
use tracing::{debug, info, instrument};

/// Per-invocation graph policy. `auto` runs only when a bounded local graph
/// candidate exists; `on` uses the same safe bounds but makes the request
/// explicit; `off` reproduces the pre-graph ranking path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphMode {
    Off,
    Auto,
    On,
}

/// Validated bounds for the accepted-edge graph retrieval channel.
#[derive(Debug, Clone)]
pub struct GraphRetrievalConfig {
    pub enabled: bool,
    pub max_seed_entities: usize,
    pub max_seed_notes: usize,
    pub max_hops: usize,
    pub per_node_fanout: usize,
    pub allowed_edge_types: Vec<String>,
    pub allow_outbound: bool,
    pub allow_inbound: bool,
    pub min_confidence: f32,
    pub per_hop_decay: f32,
    pub candidate_cap: usize,
    /// Fixed graph-channel score before hop/confidence decay. It shares the
    /// existing final sorter rather than creating a second ranker.
    pub seed_score: f32,
}

impl Default for GraphRetrievalConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_seed_entities: 4,
            max_seed_notes: 12,
            max_hops: 1,
            per_node_fanout: 8,
            allowed_edge_types: vec![
                "supports".into(),
                "contradicts".into(),
                "derived_from".into(),
                "related_to".into(),
            ],
            allow_outbound: true,
            allow_inbound: true,
            min_confidence: 0.0,
            per_hop_decay: 0.8,
            candidate_cap: 32,
            seed_score: 0.03,
        }
    }
}

/// Human-reconstructable graph path retained from retrieval through context
/// packing and CLI citations.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct GraphPathStep {
    pub edge_id: String,
    pub edge_type: String,
    pub direction: String,
    pub confidence: f32,
    pub from_id: String,
    pub to_id: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct GraphEvidence {
    pub query_entities: Vec<String>,
    pub seed_note_id: String,
    pub path: Vec<GraphPathStep>,
    pub hops: usize,
    pub decay: f32,
    pub score: f32,
    pub source_uri: Option<String>,
    /// Original chat record IDs when the note was derived from imported chat
    /// content. File-backed provenance remains in `source_uri`.
    pub provenance_ids: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct GraphRetrievalSummary {
    pub entities_matched: usize,
    pub candidates_considered: usize,
    pub candidates_selected: usize,
    pub candidates_dropped: usize,
}

#[derive(Debug, Clone)]
pub struct GraphSearchResults {
    pub hits: Vec<ScopedSearchResult>,
    pub summary: GraphRetrievalSummary,
}

/// Search result with optional graph context
#[derive(Debug)]
pub struct EnrichedSearchResult {
    pub result: SearchResult,
    pub related: Option<RelatedNotes>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchScope {
    Notes,
    Messages,
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchHitType {
    Note,
    Message,
    ConversationSummary,
}

#[derive(Debug, Clone)]
pub struct ScopedSearchResult {
    pub hit_type: SearchHitType,
    pub id: String,
    pub title: Option<String>,
    pub content: String,
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    pub source_uri: Option<String>,
    pub score: f32,
    /// Retrieval evidence is intentionally retained for a later `--explain`
    /// surface while current CLI output remains unchanged.
    pub fusion: FusionEvidence,
    pub conversation_uuid: Option<String>,
    pub message_index: Option<i64>,
    pub role: Option<String>,
    /// Present only when this note was reached through the bounded accepted
    /// graph channel. Ordinary hybrid candidates retain their existing
    /// evidence unchanged.
    pub graph: Option<GraphEvidence>,
}

pub use crate::context_packing::{
    AugmentChunk, AugmentContext, AugmentDiagnostics, AugmentOptions, ConservativeTokenCounter,
    TokenCountMode, TokenCounter,
};

/// The Search agent handles user queries
pub struct SearchAgent {
    repo: Repository,
    embedder: SharedEmbedder,
    fusion: FusionConfig,
    note_weight: f32,
    message_weight: f32,
    conversation_summary_weight: f32,
    graph: GraphRetrievalConfig,
}

impl SearchAgent {
    /// Create a new Search agent
    pub fn new(repo: Repository, embedder: SharedEmbedder) -> Self {
        Self {
            repo,
            embedder,
            fusion: FusionConfig::default(),
            note_weight: 1.0,
            message_weight: 1.0,
            conversation_summary_weight: 1.0,
            graph: GraphRetrievalConfig::default(),
        }
    }

    /// Set vector and full-text weights while retaining the RRF default.
    /// Prefer [`Self::with_fusion_config`] for all runtime controls.
    pub fn with_hybrid_weights(mut self, vector_weight: f32, fulltext_weight: f32) -> Self {
        self.fusion.vector_weight = vector_weight;
        self.fusion.fulltext_weight = fulltext_weight;
        self
    }

    /// Configure the one fusion component used for repository and all-scope
    /// search. Callers obtain validated values from `graphrag-config`.
    pub fn with_fusion_config(
        mut self,
        fusion: FusionConfig,
        note_weight: f32,
        message_weight: f32,
        conversation_summary_weight: f32,
    ) -> Self {
        self.fusion = fusion;
        self.note_weight = note_weight;
        self.message_weight = message_weight;
        self.conversation_summary_weight = conversation_summary_weight;
        self
    }

    pub fn with_graph_config(mut self, graph: GraphRetrievalConfig) -> Self {
        self.graph = graph;
        self
    }

    async fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let embedding = self.embedder.embed(query, true).await?;
        validate_embedding_dim(embedding.len())?;
        let capability = self.embedder.capabilities();
        self.repo
            .record_embedding_metadata(
                &EmbeddingIdentity::new(capability.provider, capability.model, embedding.len()),
                None,
            )
            .await?;
        Ok(embedding)
    }

    /// Perform hybrid search (vector + full-text)
    #[instrument(skip(self))]
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        self.search_notes(query, limit, None, None).await
    }

    #[instrument(skip(self))]
    pub async fn search_notes(
        &self,
        query: &str,
        limit: usize,
        since_days: Option<u32>,
        source_uri: Option<String>,
    ) -> Result<Vec<SearchResult>> {
        info!("Searching for: {}", query);

        // Generate query embedding
        debug!("Generating query embedding...");
        let embedding = self.embed_query(query).await?;
        let since = since_days.map(|days| Utc::now() - Duration::days(days as i64));

        // Perform hybrid search
        let results = self
            .repo
            .hybrid_search_notes_with_fusion(
                query,
                embedding,
                limit,
                since,
                source_uri,
                &self.fusion,
            )
            .await?;

        info!("Found {} results", results.len());

        Ok(results)
    }

    /// Search with graph context (includes related notes)
    #[instrument(skip(self))]
    pub async fn search_with_context(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<EnrichedSearchResult>> {
        self.search_with_context_filtered(query, limit, None, None)
            .await
    }

    #[instrument(skip(self))]
    pub async fn search_with_context_filtered(
        &self,
        query: &str,
        limit: usize,
        since_days: Option<u32>,
        source_uri: Option<String>,
    ) -> Result<Vec<EnrichedSearchResult>> {
        let results = self
            .search_notes(query, limit, since_days, source_uri)
            .await?;

        let mut enriched = Vec::new();

        for result in results {
            // Try to get related notes (best effort) using the full RecordId
            let related = self.repo.get_related_notes(&result.id).await.ok();

            enriched.push(EnrichedSearchResult { result, related });
        }

        Ok(enriched)
    }

    /// Vector-only search (semantic similarity)
    #[instrument(skip(self))]
    pub async fn semantic_search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        debug!("Performing semantic search for: {}", query);

        let embedding = self.embed_query(query).await?;
        let results = self.repo.vector_search(embedding, limit).await?;

        Ok(results)
    }

    /// Full-text only search (keyword matching)
    #[instrument(skip(self))]
    pub async fn keyword_search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        debug!("Performing keyword search for: {}", query);

        let results = self.repo.fulltext_search(query, limit).await?;

        Ok(results)
    }

    #[instrument(skip(self))]
    pub async fn search_with_scope(
        &self,
        query: &str,
        limit: usize,
        scope: SearchScope,
        since_days: Option<u32>,
        source_uri: Option<String>,
    ) -> Result<Vec<ScopedSearchResult>> {
        Ok(self
            .search_with_scope_graph(query, limit, scope, since_days, source_uri, GraphMode::Off)
            .await?
            .hits)
    }

    /// Search through the normal hybrid channels and, when requested, merge
    /// bounded accepted-edge candidates into the same deterministic sorter.
    #[instrument(skip(self))]
    pub async fn search_with_scope_graph(
        &self,
        query: &str,
        limit: usize,
        scope: SearchScope,
        since_days: Option<u32>,
        source_uri: Option<String>,
        graph_mode: GraphMode,
    ) -> Result<GraphSearchResults> {
        let since = since_days.map(|days| Utc::now() - Duration::days(days as i64));
        let embedding = self.embed_query(query).await?;
        let mut scoped_results = Vec::new();

        if matches!(scope, SearchScope::Notes | SearchScope::All) {
            let notes = self
                .repo
                .hybrid_search_notes_with_fusion(
                    query,
                    embedding.clone(),
                    limit,
                    since,
                    source_uri.clone(),
                    &self.fusion,
                )
                .await?;
            scoped_results.extend(
                notes
                    .into_iter()
                    .map(|result| self.from_note_result(result)),
            );
        }

        if matches!(scope, SearchScope::Messages | SearchScope::All) {
            let messages = self
                .repo
                .hybrid_search_messages_with_fusion(
                    query,
                    embedding.clone(),
                    limit,
                    since,
                    source_uri.clone(),
                    &self.fusion,
                )
                .await?;
            scoped_results.extend(
                messages
                    .into_iter()
                    .map(|result| self.from_message_result(result)),
            );
        }

        if matches!(scope, SearchScope::All) {
            let conversations = self
                .repo
                .hybrid_search_conversation_summaries_with_fusion(
                    query,
                    embedding,
                    limit,
                    since,
                    source_uri.clone(),
                    &self.fusion,
                )
                .await?;
            scoped_results.extend(
                conversations
                    .into_iter()
                    .map(|result| self.from_conversation_result(result)),
            );
        }

        let mut summary = GraphRetrievalSummary::default();
        if !matches!(graph_mode, GraphMode::Off)
            && self.graph.enabled
            && matches!(scope, SearchScope::Notes | SearchScope::All)
        {
            let graph = self
                .graph_candidates(
                    query,
                    &scoped_results,
                    since,
                    source_uri.clone(),
                    matches!(graph_mode, GraphMode::Auto),
                )
                .await?;
            summary = graph.summary;
            // `auto` has no special score path: if there are no useful local
            // entities/seeds/neighbors, graph.hits is empty and baseline
            // ranking is bit-for-bit preserved.
            if !graph.hits.is_empty() || matches!(graph_mode, GraphMode::On) {
                merge_graph_hits(&mut scoped_results, graph.hits);
            }
        }

        rank_scoped_results(&mut scoped_results);
        if scoped_results.len() > limit {
            scoped_results.truncate(limit);
        }
        summary.candidates_selected = scoped_results
            .iter()
            .filter(|hit| hit.graph.is_some())
            .count();
        summary.candidates_dropped = summary
            .candidates_considered
            .saturating_sub(summary.candidates_selected);

        Ok(GraphSearchResults {
            hits: scoped_results,
            summary,
        })
    }

    /// Retrieve ranked context snippets and package them for prompt augmentation.
    #[instrument(skip(self))]
    pub async fn build_augmented_context(
        &self,
        query: &str,
        scope: SearchScope,
        since_days: Option<u32>,
        source_uri: Option<String>,
        entity_filter: Option<String>,
        options: AugmentOptions,
    ) -> Result<AugmentContext> {
        self.build_augmented_context_with_graph(
            query,
            scope,
            since_days,
            source_uri,
            entity_filter,
            options,
            GraphMode::Auto,
        )
        .await
    }

    /// Augmentation counterpart to [`Self::search_with_scope_graph`]. Graph
    /// candidates are packed by the existing budget/diversity implementation.
    #[instrument(skip(self))]
    pub async fn build_augmented_context_with_graph(
        &self,
        query: &str,
        scope: SearchScope,
        since_days: Option<u32>,
        source_uri: Option<String>,
        entity_filter: Option<String>,
        options: AugmentOptions,
        graph_mode: GraphMode,
    ) -> Result<AugmentContext> {
        if options.max_chunks == 0 || options.max_total_tokens == 0 || options.max_chunk_tokens == 0
        {
            return Ok(crate::context_packing::empty_context(
                query.to_string(),
                scope,
                entity_filter,
                options.token_counter.mode(),
                0,
            ));
        }

        let fetch_limit = (options.max_chunks * 4).clamp(options.max_chunks, 200);
        let graph_results = self
            .search_with_scope_graph(
                query,
                fetch_limit,
                scope,
                since_days,
                source_uri,
                graph_mode,
            )
            .await?;
        let mut hits = graph_results.hits;

        let mut dropped_for_entity_filter = 0usize;
        if let Some(filter) = entity_filter.as_ref() {
            let mut filtered = Vec::with_capacity(hits.len());
            for hit in hits {
                if hit.hit_type == SearchHitType::Note
                    && !self.repo.note_has_entity_name(&hit.id, filter).await?
                {
                    dropped_for_entity_filter += 1;
                    continue;
                }
                filtered.push(hit);
            }
            hits = filtered;
        }

        Ok(
            crate::context_packing::build_augment_context_from_hits_with_graph(
                query.to_string(),
                scope,
                entity_filter,
                hits,
                options,
                dropped_for_entity_filter,
                graph_results.summary,
            ),
        )
    }

    async fn graph_candidates(
        &self,
        query: &str,
        baseline: &[ScopedSearchResult],
        since: Option<chrono::DateTime<Utc>>,
        source_uri: Option<String>,
        require_entity_seed: bool,
    ) -> Result<GraphSearchResults> {
        let normalized_query = Entity::canonicalize(query);
        let entities = self
            .repo
            .find_graph_entities(&normalized_query, self.graph.max_seed_entities)
            .await?;
        let entity_ids = entities
            .iter()
            .map(|entity| entity.id.clone())
            .collect::<Vec<_>>();
        let entity_names = entities
            .iter()
            .map(|entity| (record_id_to_string(&entity.id), entity.name.clone()))
            .collect::<HashMap<_, _>>();
        let entity_seed_ids = self
            .repo
            .graph_notes_for_entities(
                &entity_ids,
                self.graph.max_seed_notes,
                since,
                source_uri.clone(),
            )
            .await?;

        // `auto` is deliberately conservative: it activates only when a
        // local entity match provides useful graph evidence. Explicit `on`
        // may additionally use ordinary hybrid notes as bounded seeds.
        let mut frontier = BTreeMap::<String, GraphFrontier>::new();
        for seed in entity_seed_ids {
            let id = record_id_to_string(&seed.note_id);
            let entity_name = entity_names
                .get(&record_id_to_string(&seed.entity_id))
                .cloned()
                .into_iter()
                .collect::<Vec<_>>();
            if let Some(existing) = frontier.get_mut(&id) {
                existing.query_entities.extend(entity_name);
                existing.query_entities.sort();
                existing.query_entities.dedup();
            } else if frontier.len() < self.graph.max_seed_notes {
                frontier.insert(
                    id.clone(),
                    GraphFrontier::seed(seed.note_id, id, entity_name, self.graph.seed_score),
                );
            } else {
                // The relation query intentionally includes enough rows to
                // preserve associations for retained notes. Ignore only a
                // new note after the unique seed bound is reached.
                continue;
            }
        }
        if !require_entity_seed {
            for hit in baseline
                .iter()
                .filter(|hit| hit.hit_type == SearchHitType::Note)
            {
                if frontier.len() >= self.graph.max_seed_notes {
                    break;
                }
                let raw = hit.id.strip_prefix("note:").unwrap_or(&hit.id);
                let note_id = RecordId::new("note", raw);
                frontier.entry(hit.id.clone()).or_insert_with(|| {
                    GraphFrontier::seed(note_id, hit.id.clone(), Vec::new(), self.graph.seed_score)
                });
            }
        }

        let mut summary = GraphRetrievalSummary {
            entities_matched: entities.len(),
            ..Default::default()
        };
        if frontier.is_empty() || self.graph.candidate_cap == 0 {
            return Ok(GraphSearchResults {
                hits: Vec::new(),
                summary,
            });
        }

        let mut candidates = BTreeMap::<String, GraphEvidence>::new();
        for state in frontier.values().take(self.graph.candidate_cap) {
            if !state.query_entities.is_empty() {
                candidates.insert(state.id.clone(), state.evidence(None));
            }
        }
        let mut current = frontier;
        for hop in 1..=self.graph.max_hops.min(2) {
            if current.is_empty() || candidates.len() >= self.graph.candidate_cap {
                break;
            }
            let ids = current
                .values()
                .map(|state| state.note_id.clone())
                .collect::<Vec<_>>();
            let edges = self
                .repo
                .graph_note_edges(
                    &ids,
                    &self.graph.allowed_edge_types,
                    self.graph.per_node_fanout,
                    self.graph.allow_outbound,
                    self.graph.allow_inbound,
                    self.graph.min_confidence,
                )
                .await?;
            let mut next = BTreeMap::<String, GraphFrontier>::new();
            let mut fanout = HashMap::<String, usize>::new();
            for edge in edges {
                for (from, neighbor, direction) in
                    graph_edge_transitions(&edge, &current, &self.graph)
                {
                    let neighbor_id = record_id_to_string(&neighbor);
                    if from.visited.contains(&neighbor_id) || neighbor_id == from.id {
                        continue;
                    }
                    let confidence = edge.confidence.unwrap_or(1.0).clamp(0.0, 1.0);
                    if confidence < self.graph.min_confidence {
                        continue;
                    }
                    // Count only usable transitions. A rejected cycle or
                    // low-confidence edge must not consume this node's
                    // bounded fanout and hide a later valid neighbor.
                    let used = fanout.entry(from.id.clone()).or_default();
                    if *used >= self.graph.per_node_fanout {
                        continue;
                    }
                    *used += 1;
                    let score = from.score * self.graph.per_hop_decay * confidence;
                    let decay = from.decay * self.graph.per_hop_decay;
                    let mut path = from.path.clone();
                    path.push(GraphPathStep {
                        edge_id: record_id_to_string(&edge.id),
                        edge_type: edge.edge_type.clone(),
                        direction: direction.to_string(),
                        confidence,
                        from_id: from.id.clone(),
                        to_id: neighbor_id.clone(),
                    });
                    let mut visited = from.visited.clone();
                    visited.insert(neighbor_id.clone());
                    let state = GraphFrontier {
                        note_id: neighbor,
                        id: neighbor_id.clone(),
                        query_entities: from.query_entities.clone(),
                        seed_note_id: from.seed_note_id.clone(),
                        path,
                        visited,
                        score,
                        decay,
                    };
                    let replace = next
                        .get(&neighbor_id)
                        .is_none_or(|existing| state.score > existing.score);
                    if replace {
                        next.insert(neighbor_id.clone(), state);
                    }
                }
            }
            for state in next.values() {
                if candidates.len() >= self.graph.candidate_cap {
                    break;
                }
                candidates
                    .entry(state.id.clone())
                    .or_insert_with(|| state.evidence(Some(hop)));
            }
            current = next;
        }

        summary.candidates_considered = candidates.len();
        let ids = candidates
            .keys()
            .map(|id| RecordId::new("note", id.strip_prefix("note:").unwrap_or(id)))
            .collect::<Vec<_>>();
        let records = self
            .repo
            .graph_notes_by_ids(&ids, since, source_uri)
            .await?;
        let provenance = self
            .repo
            .graph_note_provenance_ids(
                &records
                    .iter()
                    .map(|record| record.id.clone())
                    .collect::<Vec<_>>(),
            )
            .await?;
        let mut hits = records
            .into_iter()
            .filter_map(|record| {
                let id = record_id_to_string(&record.id);
                candidates.remove(&id).map(|mut evidence| {
                    evidence.source_uri = record.source_uri.clone();
                    evidence.provenance_ids = provenance.get(&id).cloned().unwrap_or_default();
                    self.from_graph_note_result(record, evidence)
                })
            })
            .collect::<Vec<_>>();
        hits.sort_by(|left, right| left.id.cmp(&right.id));
        Ok(GraphSearchResults { hits, summary })
    }

    fn from_graph_note_result(
        &self,
        result: SearchResult,
        graph: GraphEvidence,
    ) -> ScopedSearchResult {
        let mut hit = self.from_note_result(result);
        hit.score = graph.score;
        hit.fusion = FusionEvidence {
            fused_score: graph.score,
            ..Default::default()
        };
        hit.graph = Some(graph);
        hit
    }

    fn from_note_result(&self, result: SearchResult) -> ScopedSearchResult {
        ScopedSearchResult {
            hit_type: SearchHitType::Note,
            id: record_id_to_string(&result.id),
            title: result.title,
            content: result.content,
            created_at: Some(result.created_at),
            source_uri: result.source_uri,
            score: fusion::apply_hit_type_weight(&result.fusion, self.note_weight),
            fusion: result.fusion,
            conversation_uuid: None,
            message_index: None,
            role: None,
            graph: None,
        }
    }

    fn from_message_result(&self, result: MessageSearchResult) -> ScopedSearchResult {
        ScopedSearchResult {
            hit_type: SearchHitType::Message,
            id: record_id_to_string(&result.id),
            title: Some(format!(
                "{} message #{}",
                result.role,
                result.message_index + 1
            )),
            content: result.content,
            created_at: result.created_at,
            source_uri: result.source_uri,
            score: fusion::apply_hit_type_weight(&result.fusion, self.message_weight),
            fusion: result.fusion,
            conversation_uuid: Some(result.conversation_uuid),
            message_index: Some(result.message_index),
            role: Some(result.role),
            graph: None,
        }
    }

    fn from_conversation_result(&self, result: ConversationSearchResult) -> ScopedSearchResult {
        let title = result
            .title
            .clone()
            .or_else(|| Some(format!("Conversation {}", result.uuid)));
        let content = result.summary.unwrap_or_default();
        ScopedSearchResult {
            hit_type: SearchHitType::ConversationSummary,
            id: record_id_to_string(&result.id),
            title,
            content,
            created_at: Some(result.updated_at),
            source_uri: result.source_uri,
            score: fusion::apply_hit_type_weight(&result.fusion, self.conversation_summary_weight),
            fusion: result.fusion,
            conversation_uuid: Some(result.uuid),
            message_index: None,
            role: None,
            graph: None,
        }
    }

    /// Find notes similar to a given note
    #[instrument(skip(self))]
    pub async fn find_similar(&self, note_id: &str, limit: usize) -> Result<Vec<SimilarNote>> {
        // First get the note to get its embedding
        let note = self
            .repo
            .get_note(note_id)
            .await?
            .ok_or_else(|| crate::AgentError::NotFound(format!("Note {}", note_id)))?;

        if note.embedding.is_empty() {
            return Err(crate::AgentError::NotFound("Note has no embedding".into()));
        }

        let similar = self
            .repo
            .find_similar_notes(note_id, note.embedding, 0.5, limit)
            .await?;

        Ok(similar)
    }
}

#[derive(Clone)]
struct GraphFrontier {
    note_id: RecordId,
    id: String,
    query_entities: Vec<String>,
    seed_note_id: String,
    path: Vec<GraphPathStep>,
    visited: HashSet<String>,
    score: f32,
    decay: f32,
}

impl GraphFrontier {
    fn seed(note_id: RecordId, id: String, query_entities: Vec<String>, score: f32) -> Self {
        let mut visited = HashSet::new();
        visited.insert(id.clone());
        Self {
            note_id,
            seed_note_id: id.clone(),
            id,
            query_entities,
            path: Vec::new(),
            visited,
            score,
            decay: 1.0,
        }
    }

    fn evidence(&self, hop: Option<usize>) -> GraphEvidence {
        GraphEvidence {
            query_entities: self.query_entities.clone(),
            seed_note_id: self.seed_note_id.clone(),
            path: self.path.clone(),
            hops: hop.unwrap_or(self.path.len()),
            decay: self.decay,
            score: self.score,
            source_uri: None,
            provenance_ids: Vec::new(),
        }
    }
}

fn graph_edge_transitions<'a>(
    edge: &'a NoteEdgeRow,
    frontier: &'a BTreeMap<String, GraphFrontier>,
    config: &'a GraphRetrievalConfig,
) -> Vec<(&'a GraphFrontier, RecordId, &'static str)> {
    let mut transitions = Vec::with_capacity(2);
    let in_id = record_id_to_string(&edge.in_id);
    let out_id = record_id_to_string(&edge.out_id);
    if config.allow_outbound {
        if let Some(state) = frontier.get(&in_id) {
            transitions.push((state, edge.out_id.clone(), "outbound"));
        }
    }
    if config.allow_inbound {
        if let Some(state) = frontier.get(&out_id) {
            transitions.push((state, edge.in_id.clone(), "inbound"));
        }
    }
    transitions
}

/// Merge graph results by canonical record ID. Hybrid candidates keep their
/// calibrated score while receiving graph evidence; graph-only notes enter the
/// same final sorter as a new channel.
fn merge_graph_hits(results: &mut Vec<ScopedSearchResult>, graph_hits: Vec<ScopedSearchResult>) {
    let mut positions = results
        .iter()
        .enumerate()
        .map(|(index, hit)| (hit.id.clone(), index))
        .collect::<HashMap<_, _>>();
    for graph_hit in graph_hits {
        if let Some(index) = positions.get(&graph_hit.id).copied() {
            results[index].graph = graph_hit.graph;
        } else {
            positions.insert(graph_hit.id.clone(), results.len());
            results.push(graph_hit);
        }
    }
}

fn hit_type_order(hit_type: SearchHitType) -> usize {
    match hit_type {
        SearchHitType::Note => 0,
        SearchHitType::Message => 1,
        SearchHitType::ConversationSummary => 2,
    }
}

fn rank_scoped_results(results: &mut [ScopedSearchResult]) {
    results.sort_by(|left, right| {
        fusion::compare_scoped(
            left.score,
            &left.fusion,
            hit_type_order(left.hit_type),
            &left.id,
            right.score,
            &right.fusion,
            hit_type_order(right.hit_type),
            &right.id,
        )
    });
    for (index, result) in results.iter_mut().enumerate() {
        result.fusion.final_rank = index + 1;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::context_packing::build_augment_context_from_hits;
    use crate::DeterministicEmbedder;
    use graphrag_core::{EdgeType, Entity, EntityType, Note, SourceType};
    use graphrag_db::{compatibility::embedding_metadata, init_memory, Repository};
    use std::sync::Arc;

    #[tokio::test]
    async fn vector_queries_initialize_metadata_and_reject_a_model_change() {
        let db = init_memory().await.unwrap();
        let repo = Repository::new(db.clone());
        let original = SearchAgent::new(
            repo.clone(),
            Arc::new(DeterministicEmbedder::default().with_identity("fixture", "model-a")),
        );
        original.embed_query("first query").await.unwrap();
        assert_eq!(
            embedding_metadata(&db)
                .await
                .unwrap()
                .unwrap()
                .embedding
                .model,
            "model-a"
        );

        let changed = SearchAgent::new(
            repo,
            Arc::new(DeterministicEmbedder::default().with_identity("fixture", "model-b")),
        );
        let error = changed.embed_query("second query").await.unwrap_err();
        assert!(error.to_string().contains("graphrag reindex --all"));
    }

    #[tokio::test]
    async fn unavailable_embedder_does_not_initialize_metadata() {
        let db = init_memory().await.unwrap();
        let search = SearchAgent::new(
            Repository::new(db.clone()),
            Arc::new(DeterministicEmbedder::default().fail_next_requests(1, "offline")),
        );
        assert!(search.embed_query("offline query").await.is_err());
        assert!(embedding_metadata(&db).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn graph_on_reaches_an_accepted_edge_and_off_preserves_baseline() {
        let repo = Repository::new(init_memory().await.unwrap());
        let seed = repo
            .create_note(Note::new("Atlas planning overview"))
            .await
            .unwrap();
        let target = repo
            .create_note(Note::new("deferred migration evidence"))
            .await
            .unwrap();
        let mut atlas = Entity::new("Atlas", EntityType::Project);
        atlas.metadata = serde_json::json!({});
        let entity = repo.upsert_entity(atlas).await.unwrap();
        repo.link_note_to_entity(seed.id.as_ref().unwrap(), entity.id.as_ref().unwrap())
            .await
            .unwrap();
        repo.create_edge(
            seed.id.as_ref().unwrap(),
            target.id.as_ref().unwrap(),
            EdgeType::Supports,
            Some(0.9),
        )
        .await
        .unwrap();

        let search = SearchAgent::new(repo, Arc::new(DeterministicEmbedder::default()));
        let off = search
            .search_with_scope_graph("Atlas", 10, SearchScope::Notes, None, None, GraphMode::Off)
            .await
            .unwrap();
        let target_id = record_id_to_string(target.id.as_ref().unwrap());
        assert!(off.hits.iter().all(|hit| hit.id != target_id));

        let on = search
            .search_with_scope_graph("Atlas", 10, SearchScope::Notes, None, None, GraphMode::On)
            .await
            .unwrap();
        let target = on.hits.iter().find(|hit| hit.id == target_id).unwrap();
        let evidence = target.graph.as_ref().unwrap();
        assert_eq!(evidence.hops, 1);
        assert_eq!(evidence.path[0].edge_type, "supports");
        assert_eq!(evidence.path[0].direction, "outbound");
        assert_eq!(on.summary.entities_matched, 1);
    }

    #[tokio::test]
    async fn graph_two_hops_obeys_direction_confidence_and_cycle_bounds() {
        let repo = Repository::new(init_memory().await.unwrap());
        let seed = repo
            .create_note(Note::new("Beacon overview"))
            .await
            .unwrap();
        let middle = repo
            .create_note(Note::new("middle evidence"))
            .await
            .unwrap();
        let target = repo
            .create_note(Note::new("final linked evidence"))
            .await
            .unwrap();
        let low_confidence = repo.create_note(Note::new("unsafe drift")).await.unwrap();
        let mut beacon = Entity::new("Beacon", EntityType::Project);
        beacon.metadata = serde_json::json!({});
        let beacon = repo.upsert_entity(beacon).await.unwrap();
        repo.link_note_to_entity(seed.id.as_ref().unwrap(), beacon.id.as_ref().unwrap())
            .await
            .unwrap();
        repo.create_edge(
            seed.id.as_ref().unwrap(),
            middle.id.as_ref().unwrap(),
            EdgeType::Supports,
            Some(0.9),
        )
        .await
        .unwrap();
        repo.create_edge(
            middle.id.as_ref().unwrap(),
            target.id.as_ref().unwrap(),
            EdgeType::DerivedFrom,
            Some(0.9),
        )
        .await
        .unwrap();
        repo.create_edge(
            seed.id.as_ref().unwrap(),
            low_confidence.id.as_ref().unwrap(),
            EdgeType::Supports,
            Some(0.2),
        )
        .await
        .unwrap();
        // This back edge must not reintroduce the seed as a graph result.
        repo.create_edge(
            target.id.as_ref().unwrap(),
            seed.id.as_ref().unwrap(),
            EdgeType::RelatedTo,
            Some(1.0),
        )
        .await
        .unwrap();

        let mut config = GraphRetrievalConfig::default();
        config.max_hops = 2;
        config.max_seed_notes = 1;
        config.allowed_edge_types = vec!["supports".into(), "derived_from".into()];
        config.allow_inbound = false;
        config.min_confidence = 0.8;
        let outbound = SearchAgent::new(repo.clone(), Arc::new(DeterministicEmbedder::default()))
            .with_graph_config(config.clone())
            .search_with_scope_graph("Beacon", 10, SearchScope::Notes, None, None, GraphMode::On)
            .await
            .unwrap();
        let target_id = record_id_to_string(target.id.as_ref().unwrap());
        assert_eq!(
            outbound
                .hits
                .iter()
                .find(|hit| hit.id == target_id)
                .and_then(|hit| hit.graph.as_ref())
                .unwrap()
                .hops,
            2
        );
        assert!(outbound
            .hits
            .iter()
            .find(|hit| hit.id == record_id_to_string(low_confidence.id.as_ref().unwrap()))
            .is_none_or(|hit| hit.graph.is_none()));

        config.allow_outbound = false;
        config.allow_inbound = true;
        let inbound_only = SearchAgent::new(repo, Arc::new(DeterministicEmbedder::default()))
            .with_graph_config(config)
            .search_with_scope_graph("Beacon", 10, SearchScope::Notes, None, None, GraphMode::On)
            .await
            .unwrap();
        assert!(inbound_only
            .hits
            .iter()
            .find(|hit| hit.id == target_id)
            .is_none_or(|hit| hit.graph.is_none()));
    }

    #[tokio::test]
    async fn pending_proposals_are_not_a_graph_retrieval_channel() {
        let repo = Repository::new(init_memory().await.unwrap());
        let seed = repo
            .create_note(Note::new("Orchid overview"))
            .await
            .unwrap();
        let proposed = repo
            .create_note(Note::new("proposal-only neighbor"))
            .await
            .unwrap();
        let mut orchid = Entity::new("Orchid", EntityType::Project);
        orchid.metadata = serde_json::json!({});
        let orchid = repo.upsert_entity(orchid).await.unwrap();
        repo.link_note_to_entity(seed.id.as_ref().unwrap(), orchid.id.as_ref().unwrap())
            .await
            .unwrap();
        repo.upsert_gardener_proposal(
            seed.id.as_ref().unwrap(),
            proposed.id.as_ref().unwrap(),
            0.99,
            "similarity only".into(),
            None,
            None,
        )
        .await
        .unwrap();

        let results = SearchAgent::new(repo, Arc::new(DeterministicEmbedder::default()))
            .search_with_scope_graph("Orchid", 10, SearchScope::Notes, None, None, GraphMode::On)
            .await
            .unwrap();
        let proposed_id = record_id_to_string(proposed.id.as_ref().unwrap());
        assert!(results
            .hits
            .iter()
            .find(|hit| hit.id == proposed_id)
            .is_none_or(|hit| hit.graph.is_none()));
    }

    #[tokio::test]
    async fn graph_entity_alias_seeds_local_accepted_edge_retrieval() {
        let repo = Repository::new(init_memory().await.unwrap());
        let seed = repo
            .create_note(Note::new("internal service"))
            .await
            .unwrap();
        let target = repo
            .create_note(Note::new("linked operational runbook"))
            .await
            .unwrap();
        let mut service = Entity::new("Long service name", EntityType::Project);
        service.metadata = serde_json::json!({"aliases": ["atlas"]});
        let service = repo.upsert_entity(service).await.unwrap();
        repo.link_note_to_entity(seed.id.as_ref().unwrap(), service.id.as_ref().unwrap())
            .await
            .unwrap();
        repo.create_edge(
            seed.id.as_ref().unwrap(),
            target.id.as_ref().unwrap(),
            EdgeType::Supports,
            Some(1.0),
        )
        .await
        .unwrap();

        let results = SearchAgent::new(repo, Arc::new(DeterministicEmbedder::default()))
            .search_with_scope_graph(
                "where is atlas deployed",
                10,
                SearchScope::Notes,
                None,
                None,
                GraphMode::On,
            )
            .await
            .unwrap();
        let target_id = record_id_to_string(target.id.as_ref().unwrap());
        let target_graph = results
            .hits
            .iter()
            .find(|hit| hit.id == target_id)
            .and_then(|hit| hit.graph.as_ref())
            .expect("the alias seed should reach its accepted neighbor");
        assert_eq!(target_graph.query_entities, vec!["Long service name"]);
        assert_eq!(target_graph.path[0].edge_type, "supports");
    }

    #[tokio::test]
    async fn graph_candidate_cap_and_fanout_bound_high_degree_seeds() {
        let repo = Repository::new(init_memory().await.unwrap());
        let seed = repo.create_note(Note::new("hub note")).await.unwrap();
        let mut hub = Entity::new("Hub", EntityType::Project);
        hub.metadata = serde_json::json!({});
        let hub = repo.upsert_entity(hub).await.unwrap();
        repo.link_note_to_entity(seed.id.as_ref().unwrap(), hub.id.as_ref().unwrap())
            .await
            .unwrap();
        for index in 0..3 {
            let neighbor = repo
                .create_note(Note::new(format!("high degree neighbor {index}")))
                .await
                .unwrap();
            repo.create_edge(
                seed.id.as_ref().unwrap(),
                neighbor.id.as_ref().unwrap(),
                EdgeType::Supports,
                Some(1.0),
            )
            .await
            .unwrap();
        }

        let mut config = GraphRetrievalConfig::default();
        config.max_seed_notes = 1;
        config.per_node_fanout = 1;
        config.candidate_cap = 2; // one entity seed plus one neighbor
        let results = SearchAgent::new(repo, Arc::new(DeterministicEmbedder::default()))
            .with_graph_config(config)
            .search_with_scope_graph("Hub", 10, SearchScope::Notes, None, None, GraphMode::On)
            .await
            .unwrap();

        assert_eq!(results.summary.candidates_considered, 2);
        assert_eq!(
            results
                .hits
                .iter()
                .filter(|hit| hit.graph.as_ref().is_some_and(|graph| graph.hops == 1))
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn graph_fanout_is_applied_per_frontier_seed_not_as_a_global_table_limit() {
        let repo = Repository::new(init_memory().await.unwrap());
        let first_seed = repo.create_note(Note::new("first seed")).await.unwrap();
        let second_seed = repo.create_note(Note::new("second seed")).await.unwrap();
        let mut first_entity = Entity::new("Alpha", EntityType::Project);
        first_entity.metadata = serde_json::json!({});
        let first_entity = repo.upsert_entity(first_entity).await.unwrap();
        let mut second_entity = Entity::new("Beta", EntityType::Project);
        second_entity.metadata = serde_json::json!({});
        let second_entity = repo.upsert_entity(second_entity).await.unwrap();
        repo.link_note_to_entity(
            first_seed.id.as_ref().unwrap(),
            first_entity.id.as_ref().unwrap(),
        )
        .await
        .unwrap();
        repo.link_note_to_entity(
            second_seed.id.as_ref().unwrap(),
            second_entity.id.as_ref().unwrap(),
        )
        .await
        .unwrap();
        // Create several first-seed edges first so a former shared LIMIT
        // would return only these and starve the second frontier seed.
        for index in 0..3 {
            let neighbor = repo
                .create_note(Note::new(format!("first-seed neighbor {index}")))
                .await
                .unwrap();
            repo.create_edge(
                first_seed.id.as_ref().unwrap(),
                neighbor.id.as_ref().unwrap(),
                EdgeType::Supports,
                Some(1.0),
            )
            .await
            .unwrap();
        }
        let second_neighbor = repo
            .create_note(Note::new("second-seed reachable neighbor"))
            .await
            .unwrap();
        repo.create_edge(
            second_seed.id.as_ref().unwrap(),
            second_neighbor.id.as_ref().unwrap(),
            EdgeType::Supports,
            Some(1.0),
        )
        .await
        .unwrap();

        let mut config = GraphRetrievalConfig::default();
        config.max_seed_notes = 2;
        config.per_node_fanout = 1;
        config.candidate_cap = 4;
        let results = SearchAgent::new(repo, Arc::new(DeterministicEmbedder::default()))
            .with_graph_config(config)
            .search_with_scope_graph(
                "Alpha Beta",
                10,
                SearchScope::Notes,
                None,
                None,
                GraphMode::On,
            )
            .await
            .unwrap();
        let second_neighbor_id = record_id_to_string(second_neighbor.id.as_ref().unwrap());
        let second_graph = results
            .hits
            .iter()
            .find(|hit| hit.id == second_neighbor_id)
            .and_then(|hit| hit.graph.as_ref())
            .expect("the second seed must retain its own per-node fanout");
        assert_eq!(second_graph.hops, 1);
        assert_eq!(second_graph.query_entities, vec!["Beta"]);
    }

    #[tokio::test]
    async fn graph_edge_query_filters_direction_and_confidence_before_fanout_limit() {
        let repo = Repository::new(init_memory().await.unwrap());
        let seed = repo.create_note(Note::new("filter seed")).await.unwrap();
        let low_confidence = repo
            .create_note(Note::new("low confidence first row"))
            .await
            .unwrap();
        let inbound = repo
            .create_note(Note::new("inbound first row"))
            .await
            .unwrap();
        let valid = repo
            .create_note(Note::new("valid outbound high confidence"))
            .await
            .unwrap();
        let mut entity = Entity::new("Filter", EntityType::Project);
        entity.metadata = serde_json::json!({});
        let entity = repo.upsert_entity(entity).await.unwrap();
        repo.link_note_to_entity(seed.id.as_ref().unwrap(), entity.id.as_ref().unwrap())
            .await
            .unwrap();
        // These earlier rows must be eliminated by the DB predicate before
        // the source's one-edge fanout LIMIT is evaluated.
        repo.create_edge(
            seed.id.as_ref().unwrap(),
            low_confidence.id.as_ref().unwrap(),
            EdgeType::Supports,
            Some(0.1),
        )
        .await
        .unwrap();
        repo.create_edge(
            inbound.id.as_ref().unwrap(),
            seed.id.as_ref().unwrap(),
            EdgeType::Supports,
            Some(1.0),
        )
        .await
        .unwrap();
        repo.create_edge(
            seed.id.as_ref().unwrap(),
            valid.id.as_ref().unwrap(),
            EdgeType::Supports,
            Some(1.0),
        )
        .await
        .unwrap();

        let mut config = GraphRetrievalConfig::default();
        config.max_seed_notes = 1;
        config.per_node_fanout = 1;
        config.candidate_cap = 2;
        config.allow_inbound = false;
        config.min_confidence = 0.8;
        let results = SearchAgent::new(repo, Arc::new(DeterministicEmbedder::default()))
            .with_graph_config(config)
            .search_with_scope_graph("Filter", 10, SearchScope::Notes, None, None, GraphMode::On)
            .await
            .unwrap();
        let valid_id = record_id_to_string(valid.id.as_ref().unwrap());
        assert!(results
            .hits
            .iter()
            .find(|hit| hit.id == valid_id)
            .is_some_and(|hit| hit.graph.as_ref().is_some_and(|graph| graph.hops == 1)));
    }

    #[tokio::test]
    async fn graph_seed_cap_limits_distinct_notes_but_keeps_retained_entity_associations() {
        let repo = Repository::new(init_memory().await.unwrap());
        for name in ["Alpha", "Beta", "Gamma"] {
            let note = repo
                .create_note(Note::new(format!("{name} disjoint seed")))
                .await
                .unwrap();
            let mut entity = Entity::new(name, EntityType::Project);
            entity.metadata = serde_json::json!({});
            let entity = repo.upsert_entity(entity).await.unwrap();
            repo.link_note_to_entity(note.id.as_ref().unwrap(), entity.id.as_ref().unwrap())
                .await
                .unwrap();
        }

        let mut config = GraphRetrievalConfig::default();
        config.max_seed_entities = 3;
        config.max_seed_notes = 2;
        config.candidate_cap = 8;
        let results = SearchAgent::new(repo, Arc::new(DeterministicEmbedder::default()))
            .with_graph_config(config)
            .search_with_scope_graph(
                "Alpha Beta Gamma",
                10,
                SearchScope::Notes,
                None,
                None,
                GraphMode::On,
            )
            .await
            .unwrap();
        let graph_seeds = results
            .hits
            .iter()
            .filter_map(|hit| hit.graph.as_ref())
            .filter(|graph| graph.hops == 0)
            .collect::<Vec<_>>();
        assert_eq!(graph_seeds.len(), 2);
        assert!(graph_seeds.iter().all(|graph| {
            graph.query_entities.len() == 1
                && matches!(graph.query_entities[0].as_str(), "Alpha" | "Beta" | "Gamma")
        }));
    }

    #[tokio::test]
    async fn graph_entity_seed_filters_apply_before_the_unique_seed_limit() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut excluded = repo
            .begin_file_import(
                SourceType::Markdown,
                "excluded.md".into(),
                "fixture://excluded.md".into(),
                "excluded".into(),
                "sha256:excluded".into(),
                false,
            )
            .await
            .unwrap();
        let excluded_note = repo
            .create_note(
                Note::new("excluded seed")
                    .with_source(excluded.source.id.as_ref().unwrap().clone())
                    .with_source_generation(excluded.source.generation),
            )
            .await
            .unwrap();
        repo.complete_file_import(&mut excluded.source)
            .await
            .unwrap();

        let mut eligible = repo
            .begin_file_import(
                SourceType::Markdown,
                "eligible.md".into(),
                "fixture://eligible.md".into(),
                "eligible".into(),
                "sha256:eligible".into(),
                false,
            )
            .await
            .unwrap();
        let eligible_note = repo
            .create_note(
                Note::new("eligible seed")
                    .with_source(eligible.source.id.as_ref().unwrap().clone())
                    .with_source_generation(eligible.source.generation),
            )
            .await
            .unwrap();
        repo.complete_file_import(&mut eligible.source)
            .await
            .unwrap();

        let mut entity = Entity::new("Scoped", EntityType::Project);
        entity.metadata = serde_json::json!({});
        let entity = repo.upsert_entity(entity).await.unwrap();
        for note in [&excluded_note, &eligible_note] {
            repo.link_note_to_entity(note.id.as_ref().unwrap(), entity.id.as_ref().unwrap())
                .await
                .unwrap();
        }

        let seeds = repo
            .graph_notes_for_entities(
                &[entity.id.as_ref().unwrap().clone()],
                1,
                Some(Utc::now() - Duration::hours(1)),
                Some("fixture://eligible.md".into()),
            )
            .await
            .unwrap();
        assert_eq!(seeds.len(), 1);
        assert_eq!(
            record_id_to_string(&seeds[0].note_id),
            record_id_to_string(eligible_note.id.as_ref().unwrap())
        );
    }

    #[tokio::test]
    async fn graph_entity_matching_uses_whole_tokens_or_safe_prefixes_not_substrings() {
        let repo = Repository::new(init_memory().await.unwrap());
        for name in ["Chair", "Email relay"] {
            let mut entity = Entity::new(name, EntityType::Project);
            entity.metadata = serde_json::json!({});
            repo.upsert_entity(entity).await.unwrap();
        }
        let mut atlas = Entity::new("Atlas service", EntityType::Project);
        atlas.metadata = serde_json::json!({"aliases": ["atlas"]});
        let atlas = repo.upsert_entity(atlas).await.unwrap();

        assert!(repo.find_graph_entities("ai", 10).await.unwrap().is_empty());
        let prefix_matches = repo
            .find_graph_entities("atla deployment", 10)
            .await
            .unwrap();
        assert!(prefix_matches
            .iter()
            .any(|entity| entity.id == *atlas.id.as_ref().unwrap()));
        let alias_phrase_matches = repo
            .find_graph_entities("where is atlas deployed", 10)
            .await
            .unwrap();
        assert!(alias_phrase_matches
            .iter()
            .any(|entity| entity.id == *atlas.id.as_ref().unwrap()));
    }

    #[tokio::test]
    async fn graph_auto_requires_a_local_entity_seed_but_on_can_use_a_hybrid_seed() {
        let repo = Repository::new(init_memory().await.unwrap());
        let seed = repo
            .create_note(Note::new("unindexed hybrid seed phrase"))
            .await
            .unwrap();
        let target = repo
            .create_note(Note::new("reachable only over an accepted edge"))
            .await
            .unwrap();
        repo.create_edge(
            seed.id.as_ref().unwrap(),
            target.id.as_ref().unwrap(),
            EdgeType::Supports,
            Some(1.0),
        )
        .await
        .unwrap();
        let search = SearchAgent::new(repo, Arc::new(DeterministicEmbedder::default()));
        let target_id = record_id_to_string(target.id.as_ref().unwrap());

        let auto = search
            .search_with_scope_graph(
                "unindexed hybrid seed phrase",
                10,
                SearchScope::Notes,
                None,
                None,
                GraphMode::Auto,
            )
            .await
            .unwrap();
        assert!(auto
            .hits
            .iter()
            .find(|hit| hit.id == target_id)
            .is_none_or(|hit| hit.graph.is_none()));

        let on = search
            .search_with_scope_graph(
                "unindexed hybrid seed phrase",
                10,
                SearchScope::Notes,
                None,
                None,
                GraphMode::On,
            )
            .await
            .unwrap();
        assert!(on
            .hits
            .iter()
            .find(|hit| hit.id == target_id)
            .is_some_and(|hit| hit.graph.is_some()));
    }

    fn make_hit(id: &str, score: f32, content: &str) -> ScopedSearchResult {
        ScopedSearchResult {
            hit_type: SearchHitType::Note,
            id: id.to_string(),
            title: Some("title".to_string()),
            content: content.to_string(),
            created_at: None,
            source_uri: None,
            score,
            fusion: FusionEvidence {
                fused_score: score,
                ..FusionEvidence::default()
            },
            conversation_uuid: None,
            message_index: None,
            role: None,
            graph: None,
        }
    }

    #[test]
    fn deduplicates_similar_content() {
        let hits = vec![
            make_hit("note:a", 0.9, "Alpha beta gamma"),
            make_hit("note:b", 0.8, "alpha beta gamma"),
            make_hit("note:c", 0.7, "delta epsilon"),
        ];

        let ctx = build_augment_context_from_hits(
            "query".to_string(),
            SearchScope::Notes,
            None,
            hits,
            AugmentOptions {
                max_chunks: 5,
                max_total_tokens: 200,
                max_chunk_tokens: 30,
                ..Default::default()
            },
            0,
        );

        assert_eq!(ctx.chunks.len(), 2);
        assert_eq!(ctx.dropped_duplicates, 0);
        assert_eq!(ctx.diagnostics.dropped_near_duplicates, 1);
        assert_eq!(ctx.chunks[0].id, "note:a");
        assert_eq!(ctx.chunks[1].id, "note:c");
    }

    #[test]
    fn retains_source_provenance_in_augment_chunks() {
        let mut hit = make_hit("note:a", 0.9, "Alpha beta gamma");
        hit.source_uri = Some("file:///notes/alpha.md".to_string());

        let ctx = build_augment_context_from_hits(
            "query".to_string(),
            SearchScope::Notes,
            None,
            vec![hit],
            AugmentOptions::default(),
            0,
        );

        assert_eq!(
            ctx.chunks[0].source_uri.as_deref(),
            Some("file:///notes/alpha.md")
        );
    }

    #[test]
    fn scope_ties_are_stable_by_hit_type_then_record_id() {
        let mut message = make_hit("message:z", 0.5, "message");
        message.hit_type = SearchHitType::Message;
        let note_b = make_hit("note:b", 0.5, "note b");
        let note_a = make_hit("note:a", 0.5, "note a");
        let mut hits = vec![message, note_b, note_a];

        rank_scoped_results(&mut hits);

        assert_eq!(
            hits.iter().map(|hit| hit.id.as_str()).collect::<Vec<_>>(),
            vec!["note:a", "note:b", "message:z"]
        );
        assert_eq!(hits[0].fusion.final_rank, 1);
    }

    #[test]
    fn enforces_total_token_budget() {
        let hits = vec![
            make_hit("note:a", 0.9, "one two three four five six"),
            make_hit("note:b", 0.8, "seven eight nine ten eleven"),
        ];

        let ctx = build_augment_context_from_hits(
            "query".to_string(),
            SearchScope::Notes,
            None,
            hits,
            AugmentOptions {
                max_chunks: 5,
                max_total_tokens: 75,
                max_chunk_tokens: 30,
                ..Default::default()
            },
            0,
        );

        assert_eq!(ctx.chunks.len(), 1);
        assert!(ctx.total_tokens <= 75);
        assert_eq!(ctx.dropped_for_budget, 1);
    }

    #[test]
    fn truncates_each_chunk_to_token_limit() {
        let hits = vec![make_hit("note:a", 0.9, "one two three four five six")];

        let ctx = build_augment_context_from_hits(
            "query".to_string(),
            SearchScope::Notes,
            None,
            hits,
            AugmentOptions {
                max_chunks: 2,
                max_total_tokens: 200,
                max_chunk_tokens: 15,
                ..Default::default()
            },
            0,
        );

        assert_eq!(ctx.chunks.len(), 1);
        assert!(ctx.chunks[0].approx_tokens <= 15);
        assert!(ctx.chunks[0].truncated);
    }

    #[test]
    fn library_defaults_use_rrf_with_balanced_scope_weights() {
        let config = FusionConfig::default();
        assert_eq!(config.strategy, fusion::FusionStrategy::ReciprocalRank);
        assert_eq!(config.rrf_k, 60);
        assert_eq!(config.vector_weight, 0.7);
        assert_eq!(config.fulltext_weight, 0.3);
    }
}
