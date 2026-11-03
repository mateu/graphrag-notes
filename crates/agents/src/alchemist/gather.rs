//! Bounded evidence gathering and revision pinning. Every operation here is a
//! read through [`AlchemistReads`] or the existing `build_context` path.

use super::{
    verify::DraftSynthesis, AlchemistError, AlchemistReads, NoteMeta, SynthesisBudgets,
    MAX_SEED_HOPS,
};
use crate::search::{AugmentChunk, AugmentOptions, GraphMode, SearchAgent, SearchScope};
use crate::TokenCounter;
use chrono::{DateTime, Utc};
use graphrag_core::record_id_to_string;
use graphrag_db::repository::{parse_record_id, NoteEdgeRow};
use graphrag_db::{DbError, RecordInspection};
use std::collections::{HashMap, HashSet};
use surrealdb::types::RecordId;

/// Accepted edges examined per frontier note and relation table while
/// expanding a seed.
pub(crate) const SEED_FANOUT: usize = 8;
/// Excerpts shorter than this are not worth a prompt slot.
const MIN_EXCERPT_TOKENS: usize = 16;

/// One traversal step, mirroring `GraphPathStep`: `from_id` is where the
/// traversal came from and `to_id` the note it reached.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PathStep {
    pub edge_id: String,
    pub edge_type: String,
    pub direction: String,
    pub from_id: String,
    pub to_id: String,
}

/// One pinned evidence chunk. `content` is the full record text at
/// `revision`; `text` is the budgeted excerpt shown to the model.
#[derive(Debug, Clone)]
pub(crate) struct EvidenceChunk {
    pub label: String,
    pub id: String,
    pub revision: String,
    pub content: String,
    pub text: String,
    pub title: Option<String>,
    pub created_at: Option<DateTime<Utc>>,
    pub source_id: Option<String>,
    pub source_uri: Option<String>,
    pub conversation_id: Option<String>,
    pub path: Vec<PathStep>,
    /// Cleared when the post-generation re-check finds a newer revision.
    pub current: bool,
}

#[derive(Debug, Default)]
pub(crate) struct EvidenceSet {
    pub chunks: Vec<EvidenceChunk>,
    pub relations: Vec<NoteEdgeRow>,
    pub truncated: bool,
    pub unpinned: Vec<String>,
    pub considered: usize,
    pub context_tokens: usize,
}

impl EvidenceSet {
    /// Add a pinned chunk if its excerpt fits the remaining token budget.
    /// Returns false once the context budget is exhausted.
    fn push(
        &mut self,
        pinned: RecordInspection,
        excerpt: &str,
        meta: NoteMeta,
        path: Vec<PathStep>,
        budgets: &SynthesisBudgets,
        counter: &dyn TokenCounter,
    ) -> bool {
        let remaining = budgets
            .max_context_tokens
            .saturating_sub(self.context_tokens);
        let per_chunk = per_chunk_tokens(budgets).min(remaining);
        let text = truncate_to_tokens(excerpt, per_chunk, counter);
        let tokens = counter.count(text);
        if tokens == 0 || (tokens < MIN_EXCERPT_TOKENS && tokens < counter.count(excerpt)) {
            self.truncated = true;
            return false;
        }
        self.context_tokens += tokens;
        self.chunks.push(EvidenceChunk {
            label: format!("C{}", self.chunks.len() + 1),
            id: pinned.id,
            revision: pinned.revision,
            text: text.to_string(),
            content: pinned.content,
            title: meta.title.or(pinned.title),
            created_at: meta.created_at,
            source_id: pinned.provenance.source_id,
            source_uri: meta.source_uri.or(pinned.provenance.source_uri),
            conversation_id: pinned.provenance.conversation_id,
            path,
            current: true,
        });
        true
    }
}

/// Spread the context budget across the chunk budget.
pub(crate) fn per_chunk_tokens(budgets: &SynthesisBudgets) -> usize {
    (budgets.max_context_tokens / budgets.max_chunks.max(1)).max(MIN_EXCERPT_TOKENS)
}

/// Longest prefix of `text`, on a character boundary, within `max` tokens.
pub(crate) fn truncate_to_tokens<'a>(
    text: &'a str,
    max: usize,
    counter: &dyn TokenCounter,
) -> &'a str {
    if counter.count(text) <= max {
        return text;
    }
    let boundaries: Vec<usize> = text
        .char_indices()
        .map(|(index, _)| index)
        .chain([text.len()])
        .collect();
    let (mut low, mut high) = (0, boundaries.len() - 1);
    while low < high {
        let middle = (low + high).div_ceil(2);
        if counter.count(&text[..boundaries[middle]]) <= max {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    &text[..boundaries[low]]
}

/// The exact pinned text behind a `build_context` chunk. Packing may add
/// ellipsis markers, so prefer the selected byte span of the pinned content.
fn excerpt<'a>(pinned: &'a str, chunk: &'a AugmentChunk) -> &'a str {
    if let (Some(start), Some(end)) = (chunk.selected_span_start, chunk.selected_span_end) {
        if let Some(span) = pinned.get(start..end) {
            if !span.trim().is_empty() {
                return span;
            }
        }
    }
    chunk.snippet.trim().trim_matches('…').trim()
}

async fn pin(
    reads: &dyn AlchemistReads,
    id: &str,
) -> Result<Option<RecordInspection>, AlchemistError> {
    match reads.pinned_record(id).await {
        Ok(record) => Ok(Some(record)),
        Err(DbError::NotFound(..)) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

/// Query mode: reuse `build_context` retrieval and its graph paths, then pin
/// each selected chunk's exact content and revision.
pub(crate) async fn query(
    reads: &dyn AlchemistReads,
    search: &SearchAgent,
    query: &str,
    scope: SearchScope,
    graph: GraphMode,
    budgets: &SynthesisBudgets,
    augment: &AugmentOptions,
) -> Result<EvidenceSet, AlchemistError> {
    let mut options = augment.clone();
    options.max_chunks = budgets.max_chunks;
    options.max_chunks_per_source = None;
    options.max_total_tokens = budgets.max_context_tokens;
    options.max_chunk_tokens = per_chunk_tokens(budgets);
    let context = search
        .build_augmented_context_with_graph(query, scope, None, None, None, options, graph)
        .await?;
    let counter = augment.token_counter.clone();
    let mut set = EvidenceSet {
        considered: context.chunks.len(),
        ..Default::default()
    };
    for chunk in &context.chunks {
        if set.chunks.len() >= budgets.max_chunks {
            set.truncated = true;
            break;
        }
        let Some(pinned) = pin(reads, &chunk.id).await? else {
            set.unpinned.push(chunk.id.clone());
            continue;
        };
        let text = excerpt(&pinned.content, chunk).to_string();
        let path = chunk
            .graph
            .as_ref()
            .map(|graph| {
                graph
                    .path
                    .iter()
                    .map(|step| PathStep {
                        edge_id: step.edge_id.clone(),
                        edge_type: step.edge_type.clone(),
                        direction: step.direction.clone(),
                        from_id: step.from_id.clone(),
                        to_id: step.to_id.clone(),
                    })
                    .collect()
            })
            .unwrap_or_default();
        let meta = NoteMeta {
            title: chunk.title.clone(),
            created_at: chunk.created_at,
            source_uri: chunk.source_uri.clone(),
        };
        if !set.push(pinned, &text, meta, path, budgets, counter.as_ref()) {
            break;
        }
    }
    Ok(set)
}

/// Seed mode: a deterministic breadth-first walk over accepted edges only,
/// both directions, at most `max_hops` (≤ 2) hops and `max_chunks` notes.
pub(crate) async fn seed(
    reads: &dyn AlchemistReads,
    id: &str,
    budgets: &SynthesisBudgets,
    augment: &AugmentOptions,
) -> Result<EvidenceSet, AlchemistError> {
    let seed_id = parse_record_id(id, Some("note")).map_err(|_| {
        AlchemistError::Invalid("seed must be a canonical note:ID (a note chunk)".into())
    })?;
    let seed_key = record_id_to_string(&seed_id);
    let not_found = || AlchemistError::NotFound(format!("seed {seed_key} is not a visible note"));
    let mut meta = reads.visible_notes(std::slice::from_ref(&seed_id)).await?;
    let Some(seed_meta) = meta.remove(&seed_key) else {
        return Err(not_found());
    };
    let Some(pinned) = pin(reads, &seed_key).await? else {
        return Err(not_found());
    };
    let counter = augment.token_counter.clone();
    let mut set = EvidenceSet {
        considered: 1,
        ..Default::default()
    };
    let content = pinned.content.clone();
    set.push(
        pinned,
        &content,
        seed_meta,
        Vec::new(),
        budgets,
        counter.as_ref(),
    );

    let mut visited: Vec<RecordId> = vec![seed_id.clone()];
    let mut seen: HashSet<String> = HashSet::from([seed_key.clone()]);
    let mut paths: HashMap<String, Vec<PathStep>> = HashMap::from([(seed_key, Vec::new())]);
    let mut frontier = vec![seed_id];
    let mut exhausted = false;
    for _ in 0..budgets.max_hops.min(MAX_SEED_HOPS) {
        if frontier.is_empty() || exhausted {
            break;
        }
        let exclude: HashMap<String, Vec<RecordId>> = frontier
            .iter()
            .map(|note| (record_id_to_string(note), visited.clone()))
            .collect();
        let edges = reads
            .accepted_edges(&frontier, SEED_FANOUT, &exclude)
            .await?;
        let frontier_keys: Vec<String> = frontier.iter().map(record_id_to_string).collect();
        // A full per-note page means the fanout limit may have hidden more.
        let mut pages: HashMap<(String, &str), usize> = HashMap::new();
        for edge in &edges {
            for endpoint in [&edge.in_id, &edge.out_id] {
                let key = record_id_to_string(endpoint);
                if frontier_keys.contains(&key) {
                    *pages.entry((key, edge.edge_type.as_str())).or_default() += 1;
                }
            }
        }
        if pages.values().any(|count| *count >= SEED_FANOUT) {
            set.truncated = true;
        }
        // Deterministic candidate order: frontier order, then edge type and ID.
        let mut candidates: Vec<(RecordId, PathStep)> = Vec::new();
        for from in &frontier_keys {
            for edge in &edges {
                let (edge_in, edge_out) = (
                    record_id_to_string(&edge.in_id),
                    record_id_to_string(&edge.out_id),
                );
                let (neighbour, direction) = if &edge_in == from {
                    (edge.out_id.clone(), "outbound")
                } else if &edge_out == from {
                    (edge.in_id.clone(), "inbound")
                } else {
                    continue;
                };
                let to = record_id_to_string(&neighbour);
                if seen.contains(&to) || candidates.iter().any(|(id, _)| id == &neighbour) {
                    continue;
                }
                candidates.push((
                    neighbour,
                    PathStep {
                        edge_id: record_id_to_string(&edge.id),
                        edge_type: edge.edge_type.clone(),
                        direction: direction.into(),
                        from_id: from.clone(),
                        to_id: to,
                    },
                ));
            }
        }
        set.considered += candidates.len();
        let ids: Vec<RecordId> = candidates.iter().map(|(id, _)| id.clone()).collect();
        let mut visible = reads.visible_notes(&ids).await?;
        let mut next = Vec::new();
        for (neighbour, step) in candidates {
            let key = record_id_to_string(&neighbour);
            let Some(meta) = visible.remove(&key) else {
                continue;
            };
            if set.chunks.len() >= budgets.max_chunks {
                set.truncated = true;
                exhausted = true;
                break;
            }
            let Some(pinned) = pin(reads, &key).await? else {
                set.unpinned.push(key);
                continue;
            };
            let mut path = paths.get(&step.from_id).cloned().unwrap_or_default();
            path.push(step);
            let content = pinned.content.clone();
            if !set.push(
                pinned,
                &content,
                meta,
                path.clone(),
                budgets,
                counter.as_ref(),
            ) {
                exhausted = true;
                break;
            }
            seen.insert(key.clone());
            paths.insert(key, path);
            visited.push(neighbour.clone());
            next.push(neighbour);
        }
        frontier = next;
    }
    Ok(set)
}

/// Accepted edges whose two endpoints are both evidence notes. Endpoints are
/// filtered to the evidence set before any limit applies, so high-degree
/// notes cannot crowd out a lineage or contradiction edge between members.
pub(crate) async fn relations_among(
    reads: &dyn AlchemistReads,
    set: &mut EvidenceSet,
) -> Result<(), AlchemistError> {
    let ids: Vec<RecordId> = set
        .chunks
        .iter()
        .filter_map(|chunk| parse_record_id(&chunk.id, Some("note")).ok())
        .collect();
    let (relations, truncated) = reads.accepted_edges_among(&ids).await?;
    set.relations = relations;
    set.truncated |= truncated;
    Ok(())
}

/// Re-read every chunk the draft cites. A chunk edited, superseded or
/// removed since it was pinned can no longer be cited.
pub(crate) async fn recheck_cited(
    reads: &dyn AlchemistReads,
    set: &mut EvidenceSet,
    draft: &DraftSynthesis,
) -> Result<usize, AlchemistError> {
    let cited: HashSet<usize> = draft
        .claims
        .iter()
        .flat_map(|claim| claim.citations.iter())
        .filter_map(|citation| super::verify::resolve_label(&citation.chunk, &set.chunks))
        .collect();
    let mut cited: Vec<usize> = cited.into_iter().collect();
    cited.sort_unstable();
    for index in &cited {
        let chunk = &mut set.chunks[*index];
        chunk.current = match pin(reads, &chunk.id).await? {
            Some(fresh) => fresh.revision == chunk.revision && fresh.content == chunk.content,
            None => false,
        };
    }
    Ok(cited.len())
}
