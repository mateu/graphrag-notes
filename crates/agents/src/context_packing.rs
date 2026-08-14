//! Deterministic, budget-safe prompt-context packing.
//!
//! This module intentionally has no network or model-download path. A caller
//! with a locally available provider tokenizer can inject it through
//! [`TokenCounter`]; the default is a conservative, deterministic estimate.

use crate::search::{ScopedSearchResult, SearchHitType, SearchScope};
use chrono::{DateTime, Utc};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::ops::Range;
use std::sync::Arc;
use unicode_casefold::UnicodeCaseFold;
use unicode_normalization::{char::canonical_combining_class, UnicodeNormalization};

/// Whether token counts were produced by a supplied tokenizer or the safe
/// local estimate. The mode is deliberately part of diagnostics: an estimate
/// must never be presented as a model's exact token count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenCountMode {
    Exact,
    Estimated,
}

/// Counts tokens without requiring a remote inference request.
///
/// Production callers may install a tokenizer-backed implementation when its
/// model files are already present locally. GraphRAG Notes never downloads a
/// tokenizer as part of context construction.
pub trait TokenCounter: Send + Sync {
    fn count(&self, text: &str) -> usize;
    fn mode(&self) -> TokenCountMode;
}

/// A deterministic, deliberately conservative tokenizer-free estimate.
///
/// Every scalar, including whitespace, is charged by its UTF-8 byte length.
/// This deliberately treats hashes, URLs, opaque identifiers, CJK, emoji, and
/// long indentation as a safe upper bound rather than under-counting a model
/// tokenizer. This is suitable for hard-budget fallback mode.
#[derive(Debug, Default)]
pub struct ConservativeTokenCounter;

impl TokenCounter for ConservativeTokenCounter {
    fn count(&self, text: &str) -> usize {
        text.chars().map(char::len_utf8).sum()
    }

    fn mode(&self) -> TokenCountMode {
        TokenCountMode::Estimated
    }
}

#[derive(Clone)]
pub struct AugmentOptions {
    pub max_chunks: usize,
    pub max_total_tokens: usize,
    pub max_chunk_tokens: usize,
    /// Weight of novelty in greedy MMR-style selection. `0.0` preserves the
    /// incoming retrieval order; `1.0` is allowed but `min_relevance` still
    /// gates candidates.
    pub novelty_weight: f32,
    /// Candidates below this retrieval score are never selected for diversity.
    pub min_relevance: f32,
    /// Jaccard similarity at or above this value is treated as a near duplicate.
    pub near_duplicate_threshold: f32,
    /// A locally supplied exact/model tokenizer, or the conservative fallback.
    pub token_counter: Arc<dyn TokenCounter>,
}

impl fmt::Debug for AugmentOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AugmentOptions")
            .field("max_chunks", &self.max_chunks)
            .field("max_total_tokens", &self.max_total_tokens)
            .field("max_chunk_tokens", &self.max_chunk_tokens)
            .field("novelty_weight", &self.novelty_weight)
            .field("min_relevance", &self.min_relevance)
            .field("near_duplicate_threshold", &self.near_duplicate_threshold)
            .field("token_count_mode", &self.token_counter.mode())
            .finish()
    }
}

impl Default for AugmentOptions {
    fn default() -> Self {
        Self {
            max_chunks: 8,
            max_total_tokens: 1200,
            max_chunk_tokens: 180,
            novelty_weight: 0.25,
            min_relevance: 0.0,
            near_duplicate_threshold: 0.85,
            token_counter: Arc::new(ConservativeTokenCounter),
        }
    }
}

impl AugmentOptions {
    pub fn with_token_counter(mut self, token_counter: Arc<dyn TokenCounter>) -> Self {
        self.token_counter = token_counter;
        self
    }
}

#[derive(Debug, Clone)]
pub struct AugmentChunk {
    pub citation: usize,
    pub hit_type: SearchHitType,
    /// Full record ID, never a shortened display identifier.
    pub id: String,
    pub title: Option<String>,
    pub snippet: String,
    pub created_at: Option<DateTime<Utc>>,
    pub source_uri: Option<String>,
    /// Original retrieval evidence score, before diversity selection.
    pub score: f32,
    pub conversation_uuid: Option<String>,
    pub message_index: Option<i64>,
    pub role: Option<String>,
    /// Snippet-only token count; `AugmentContext::total_tokens` includes all
    /// rendered headers, citation labels, and context framing.
    pub approx_tokens: usize,
    pub rendered_tokens: usize,
    pub truncated: bool,
    /// Byte offsets into the original hit content when a query-aware span was
    /// selected. They are UTF-8 boundaries and are `None` only for empty text.
    pub selected_span_start: Option<usize>,
    pub selected_span_end: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AugmentDiagnostics {
    pub token_count_mode: TokenCountMode,
    pub header_tokens: usize,
    pub dropped_duplicates: usize,
    pub dropped_near_duplicates: usize,
    pub dropped_for_relevance: usize,
    pub dropped_for_budget: usize,
    pub dropped_for_entity_filter: usize,
}

#[derive(Debug, Clone)]
pub struct AugmentContext {
    pub query: String,
    pub scope: SearchScope,
    pub entity_filter: Option<String>,
    pub chunks: Vec<AugmentChunk>,
    /// Count of the complete rendered prompt block under `diagnostics` mode.
    pub total_tokens: usize,
    pub diagnostics: AugmentDiagnostics,
    // Compatibility fields retained for programmatic callers that consumed the
    // first augmentation API. New code should use `diagnostics`.
    pub dropped_duplicates: usize,
    pub dropped_for_budget: usize,
    pub dropped_for_entity_filter: usize,
}

impl AugmentContext {
    pub fn render_prompt_block(&self) -> String {
        render_prompt_block(&self.chunks)
    }
}

pub(crate) fn build_augment_context_from_hits(
    query: String,
    scope: SearchScope,
    entity_filter: Option<String>,
    hits: Vec<ScopedSearchResult>,
    options: AugmentOptions,
    dropped_for_entity_filter: usize,
) -> AugmentContext {
    let counter = options.token_counter.as_ref();
    if options.max_chunks == 0 || options.max_total_tokens == 0 || options.max_chunk_tokens == 0 {
        return empty_context(
            query,
            scope,
            entity_filter,
            counter.mode(),
            dropped_for_entity_filter,
        );
    }

    let mut candidates = Vec::new();
    let mut seen_ids = HashSet::new();
    let mut dropped_duplicates = 0usize;
    let mut dropped_for_relevance = 0usize;
    for (rank, hit) in hits.into_iter().enumerate() {
        if !seen_ids.insert(hit.id.clone()) || hit.content.trim().is_empty() {
            dropped_duplicates += 1;
            continue;
        }
        let mut candidate = Candidate::from_hit(hit);
        candidate.rank = rank;
        candidates.push(candidate);
    }
    normalize_candidate_relevance(&mut candidates);
    candidates.retain(|candidate| {
        let keep = candidate.relevance >= options.min_relevance;
        if !keep {
            dropped_for_relevance += 1;
        }
        keep
    });

    let mut selected = Vec::new();
    let mut dropped_near_duplicates = 0usize;
    let mut dropped_for_budget = 0usize;

    while !candidates.is_empty() && selected.len() < options.max_chunks {
        let selected_tokens = selected
            .iter()
            .map(|selected: &SelectedChunk| &selected.full_tokens)
            .collect::<Vec<_>>();
        let next_index = candidates
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| {
                selection_score(left, &selected_tokens, options.novelty_weight)
                    .total_cmp(&selection_score(
                        right,
                        &selected_tokens,
                        options.novelty_weight,
                    ))
                    .then_with(|| right.rank.cmp(&left.rank))
                    .then_with(|| right.hit.id.cmp(&left.hit.id))
            })
            .map(|(index, _)| index)
            .expect("candidates is non-empty");
        let candidate = candidates.swap_remove(next_index);

        if selected.iter().any(|selected| {
            multiset_jaccard_similarity(&candidate.full_tokens, &selected.full_tokens)
                >= options.near_duplicate_threshold
        }) {
            dropped_near_duplicates += 1;
            continue;
        }

        let citation = selected.len() + 1;
        let rendered_chunks = selected
            .iter()
            .map(|selected| selected.chunk.clone())
            .collect::<Vec<_>>();
        let Some(chunk) = fit_candidate(&candidate, &query, citation, &rendered_chunks, &options)
        else {
            dropped_for_budget += 1;
            continue;
        };
        selected.push(SelectedChunk {
            chunk,
            full_tokens: candidate.full_tokens,
        });
    }

    let chunks = selected
        .into_iter()
        .map(|selected| selected.chunk)
        .collect::<Vec<_>>();

    let total_tokens = counter.count(&render_prompt_block(&chunks));
    let snippet_tokens = chunks
        .iter()
        .map(|chunk| chunk.approx_tokens)
        .sum::<usize>();
    let diagnostics = AugmentDiagnostics {
        token_count_mode: counter.mode(),
        header_tokens: total_tokens.saturating_sub(snippet_tokens),
        dropped_duplicates,
        dropped_near_duplicates,
        dropped_for_relevance,
        dropped_for_budget,
        dropped_for_entity_filter,
    };
    AugmentContext {
        query,
        scope,
        entity_filter,
        chunks,
        total_tokens,
        dropped_duplicates,
        dropped_for_budget,
        dropped_for_entity_filter,
        diagnostics,
    }
}

pub(crate) fn empty_context(
    query: String,
    scope: SearchScope,
    entity_filter: Option<String>,
    mode: TokenCountMode,
    dropped_for_entity_filter: usize,
) -> AugmentContext {
    let diagnostics = AugmentDiagnostics {
        token_count_mode: mode,
        header_tokens: 0,
        dropped_duplicates: 0,
        dropped_near_duplicates: 0,
        dropped_for_relevance: 0,
        dropped_for_budget: 0,
        dropped_for_entity_filter,
    };
    AugmentContext {
        query,
        scope,
        entity_filter,
        chunks: Vec::new(),
        total_tokens: 0,
        dropped_duplicates: 0,
        dropped_for_budget: 0,
        dropped_for_entity_filter,
        diagnostics,
    }
}

#[derive(Debug)]
struct Candidate {
    hit: ScopedSearchResult,
    content_start: usize,
    rank: usize,
    /// Full-record similarity features are computed once during pool
    /// construction and reused by every MMR and duplicate comparison.
    full_tokens: HashMap<String, usize>,
    /// Relative retrieval relevance normalized across the eligible candidate
    /// pool before MMR combines it with novelty.
    relevance: f32,
}

#[derive(Debug)]
struct SelectedChunk {
    chunk: AugmentChunk,
    /// The full untruncated record token set. Diversity and duplicate checks
    /// deliberately never compare a new record to only a selected snippet.
    full_tokens: HashMap<String, usize>,
}

impl Candidate {
    fn from_hit(hit: ScopedSearchResult) -> Self {
        let content_start = hit.content.len() - hit.content.trim_start().len();
        let full_tokens = similarity_tokens(hit.content.trim());
        Self {
            hit,
            content_start,
            rank: 0,
            full_tokens,
            relevance: 0.0,
        }
    }

    fn content(&self) -> &str {
        self.hit.content.trim()
    }
}

fn selection_score(
    candidate: &Candidate,
    selected: &[&HashMap<String, usize>],
    novelty_weight: f32,
) -> f32 {
    let novelty = selected
        .iter()
        .map(|chosen| 1.0 - multiset_jaccard_similarity(&candidate.full_tokens, chosen))
        .fold(1.0_f32, f32::min);
    let novelty_weight = novelty_weight.clamp(0.0, 1.0);
    candidate.relevance * (1.0 - novelty_weight) + novelty * novelty_weight
}

fn normalize_candidate_relevance(candidates: &mut [Candidate]) {
    let Some(min_score) = candidates
        .iter()
        .map(|candidate| candidate.hit.score)
        .min_by(|left, right| left.total_cmp(right))
    else {
        return;
    };
    let max_score = candidates
        .iter()
        .map(|candidate| candidate.hit.score)
        .max_by(|left, right| left.total_cmp(right))
        .expect("candidate pool is non-empty");
    let span = max_score - min_score;
    for candidate in candidates {
        candidate.relevance = if span.is_finite() && span > f32::EPSILON {
            ((candidate.hit.score - min_score) / span).clamp(0.0, 1.0)
        } else {
            // Equal scores are equivalent retrieval evidence; preserve their
            // deterministic rank ordering rather than inventing relevance.
            1.0
        };
    }
}

fn fit_candidate(
    candidate: &Candidate,
    query: &str,
    citation: usize,
    chunks: &[AugmentChunk],
    options: &AugmentOptions,
) -> Option<AugmentChunk> {
    let counter = options.token_counter.as_ref();
    let mut fixed_prompt = chunks.to_vec();
    fixed_prompt.push(AugmentChunk {
        citation,
        hit_type: candidate.hit.hit_type,
        id: candidate.hit.id.clone(),
        title: candidate.hit.title.clone(),
        snippet: String::new(),
        created_at: candidate.hit.created_at,
        source_uri: candidate.hit.source_uri.clone(),
        score: candidate.hit.score,
        conversation_uuid: candidate.hit.conversation_uuid.clone(),
        message_index: candidate.hit.message_index,
        role: candidate.hit.role.clone(),
        approx_tokens: 0,
        rendered_tokens: 0,
        truncated: true,
        selected_span_start: None,
        selected_span_end: None,
    });
    let fixed_tokens = counter.count(&render_prompt_block(&fixed_prompt));
    if fixed_tokens >= options.max_total_tokens {
        return None;
    }
    let usable_cap = options
        .max_chunk_tokens
        .min(options.max_total_tokens.saturating_sub(fixed_tokens));
    if usable_cap == 0 {
        return None;
    }
    let initial = clip_query_aware(candidate.content(), query, usable_cap, counter);
    if initial.snippet.is_empty() {
        return None;
    }
    let shrink_anchor = initial.start.zip(initial.end).and_then(|(start, end)| {
        let segment = &candidate.content()[start..end];
        phrase_match_range(segment, query)
            .or_else(|| lexical_match_range(segment, &normalized_tokens(query)))
    });
    let mut cap = usable_cap;
    while cap > 0 {
        let clipped = if cap == usable_cap {
            initial.clone()
        } else {
            shrink_clipped_span(
                candidate.content(),
                &initial,
                shrink_anchor.clone(),
                cap,
                counter,
            )
        };
        if clipped.snippet.is_empty() {
            return None;
        }
        let snippet_tokens = counter.count(&clipped.snippet);
        let mut prospective = chunks.to_vec();
        prospective.push(AugmentChunk {
            citation,
            hit_type: candidate.hit.hit_type,
            id: candidate.hit.id.clone(),
            title: candidate.hit.title.clone(),
            snippet: clipped.snippet,
            created_at: candidate.hit.created_at,
            source_uri: candidate.hit.source_uri.clone(),
            score: candidate.hit.score,
            conversation_uuid: candidate.hit.conversation_uuid.clone(),
            message_index: candidate.hit.message_index,
            role: candidate.hit.role.clone(),
            approx_tokens: snippet_tokens,
            rendered_tokens: 0,
            truncated: clipped.truncated,
            selected_span_start: clipped.start.map(|start| candidate.content_start + start),
            selected_span_end: clipped.end.map(|end| candidate.content_start + end),
        });
        let rendered_tokens = counter.count(&render_prompt_block(&prospective));
        if rendered_tokens <= options.max_total_tokens {
            let mut chunk = prospective.pop().expect("candidate was pushed");
            chunk.rendered_tokens =
                rendered_tokens.saturating_sub(counter.count(&render_prompt_block(chunks)));
            return Some(chunk);
        }
        let observed_overhead = rendered_tokens.saturating_sub(snippet_tokens);
        let next_cap = options
            .max_total_tokens
            .saturating_sub(observed_overhead)
            .min(cap);
        if next_cap >= cap {
            return None;
        }
        cap = next_cap;
    }
    None
}

#[derive(Debug, Clone)]
struct ClippedSpan {
    snippet: String,
    truncated: bool,
    start: Option<usize>,
    end: Option<usize>,
}

fn shrink_clipped_span(
    text: &str,
    initial: &ClippedSpan,
    anchor: Option<Range<usize>>,
    max_tokens: usize,
    counter: &dyn TokenCounter,
) -> ClippedSpan {
    let Some((start, end)) = initial.start.zip(initial.end) else {
        return ClippedSpan {
            snippet: String::new(),
            truncated: true,
            start: None,
            end: None,
        };
    };
    let segment = &text[start..end];
    let Some(selected) = anchor.or_else(|| centered_character_range(segment)) else {
        return ClippedSpan {
            snippet: String::new(),
            truncated: true,
            start: None,
            end: None,
        };
    };
    let clipped = clip_around_preselected_characters(
        segment,
        &(0..segment.len()),
        selected,
        max_tokens,
        counter,
    );
    ClippedSpan {
        snippet: clipped.snippet,
        truncated: true,
        start: clipped.start.map(|offset| start + offset),
        end: clipped.end.map(|offset| start + offset),
    }
}

fn clip_query_aware(
    text: &str,
    query: &str,
    max_tokens: usize,
    counter: &dyn TokenCounter,
) -> ClippedSpan {
    let text = text.trim();
    if text.is_empty() || max_tokens == 0 {
        return ClippedSpan {
            snippet: String::new(),
            truncated: false,
            start: None,
            end: None,
        };
    }
    if counter.count(text) <= max_tokens {
        return ClippedSpan {
            snippet: text.to_string(),
            truncated: false,
            start: Some(0),
            end: Some(text.len()),
        };
    }
    let spans = sentence_spans(text);
    let query_terms = normalized_tokens(query);
    let (best_index, _) = spans
        .iter()
        .enumerate()
        .map(|(index, span)| {
            let terms = normalized_tokens(&text[span.clone()]);
            let matches = if phrase_match_range(&text[span.clone()], query).is_some() {
                // Prefer a complete phrase over an early shared CJK bigram.
                query.chars().count().max(1)
            } else {
                query_terms.intersection(&terms).count()
            };
            (index, matches)
        })
        .max_by(|(left_index, left_matches), (right_index, right_matches)| {
            left_matches
                .cmp(right_matches)
                .then_with(|| right_index.cmp(left_index))
        })
        .unwrap_or((0, 0));

    let mut start_index = best_index;
    let mut end_index = best_index + 1;
    let mut best = span_text(text, &spans, start_index, end_index);
    if counter.count(&best) > max_tokens {
        // A giant individual span (usually one long sentence or code block):
        // center a safe UTF-8 window on the best lexical match.
        return clip_around_match(
            text,
            &spans[best_index],
            query,
            &query_terms,
            max_tokens,
            counter,
        );
    }
    loop {
        let left = start_index
            .checked_sub(1)
            .map(|next| span_text(text, &spans, next, end_index));
        let right =
            (end_index < spans.len()).then(|| span_text(text, &spans, start_index, end_index + 1));
        let next = match (left, right) {
            (Some(left), Some(right)) => {
                let left_count = counter.count(&left);
                let right_count = counter.count(&right);
                if left_count <= max_tokens
                    && (right_count > max_tokens || left_count <= right_count)
                {
                    start_index -= 1;
                    left
                } else if right_count <= max_tokens {
                    end_index += 1;
                    right
                } else {
                    break;
                }
            }
            (Some(left), None) if counter.count(&left) <= max_tokens => {
                start_index -= 1;
                left
            }
            (None, Some(right)) if counter.count(&right) <= max_tokens => {
                end_index += 1;
                right
            }
            _ => break,
        };
        best = next;
    }
    let start = spans[start_index].start;
    let end = spans[end_index - 1].end;
    ClippedSpan {
        snippet: remove_unmatched_fence_markers(&best),
        truncated: true,
        start: Some(start),
        end: Some(end),
    }
}

fn span_text(text: &str, spans: &[Range<usize>], start: usize, end: usize) -> String {
    text[spans[start].start..spans[end - 1].end]
        .trim()
        .to_string()
}

fn sentence_spans(text: &str) -> Vec<Range<usize>> {
    let mut spans = Vec::new();
    let mut line_start = 0;
    let mut fence_start = None;
    for line in text.split_inclusive('\n') {
        let line_end = line_start + line.len();
        if is_standalone_fence_line(line) {
            if let Some(start) = fence_start.take() {
                spans.push(start..line_end);
            } else {
                fence_start = Some(line_start);
            }
        } else if fence_start.is_none() {
            push_sentence_spans(line, line_start, &mut spans);
        }
        line_start = line_end;
    }
    if let Some(start) = fence_start {
        spans.push(start..text.len());
    }
    if spans.is_empty() {
        spans.push(0..text.len());
    }
    spans
}

fn push_sentence_spans(line: &str, base: usize, spans: &mut Vec<Range<usize>>) {
    let mut start = 0;
    for (index, ch) in line.char_indices() {
        if matches!(ch, '.' | '!' | '?') {
            let end = index + ch.len_utf8();
            if !line[start..end].trim().is_empty() {
                spans.push((base + start)..(base + end));
            }
            start = end;
        }
    }
    if !line[start..].trim().is_empty() {
        spans.push((base + start)..(base + line.len()));
    }
}

fn clip_around_match(
    text: &str,
    span: &Range<usize>,
    query: &str,
    query_terms: &HashSet<String>,
    max_tokens: usize,
    counter: &dyn TokenCounter,
) -> ClippedSpan {
    let segment = &text[span.clone()];
    let phrase_range = phrase_match_range(segment, query);
    // CJK and URL-like content can be a single lexical run. Word-boundary
    // expansion would make that whole run the initial candidate and drop it
    // under a small budget, so use UTF-8 character boundaries in that case.
    if !segment.chars().any(char::is_whitespace)
        || phrase_range
            .as_ref()
            .is_some_and(|range| segment[range.clone()].chars().any(is_unspaced_script_char))
    {
        return clip_around_characters(text, span, query, query_terms, max_tokens, counter);
    }
    if phrase_range.as_ref().is_some_and(|range| {
        counter.count(&render_clipped_segment(
            &segment[range.clone()],
            range.start > 0,
            range.end < segment.len(),
        )) > max_tokens
    }) {
        return clip_around_characters(text, span, query, query_terms, max_tokens, counter);
    }
    let anchor = phrase_range.as_ref().map_or_else(
        || lexical_anchor(segment, query_terms).unwrap_or(segment.len() / 2),
        |range| range.start,
    );
    let mut start = nearest_boundary_left(segment, anchor);
    let mut end = phrase_range.as_ref().map_or_else(
        || nearest_boundary_right(segment, anchor),
        |range| range.end,
    );
    if start == end {
        end = segment.len();
    }
    let mut best = String::new();
    let mut best_start = start;
    let mut best_end = end;
    loop {
        let candidate =
            render_clipped_segment(&segment[start..end], start > 0, end < segment.len());
        if counter.count(&candidate) > max_tokens {
            break;
        }
        best = candidate;
        best_start = start;
        best_end = end;
        let next_start = if start > 0 {
            nearest_boundary_left(segment, start.saturating_sub(1))
        } else {
            start
        };
        let next_end = if end < segment.len() {
            nearest_boundary_right(segment, end + 1)
        } else {
            end
        };
        if next_start == start && next_end == end {
            break;
        }
        start = next_start;
        end = next_end;
    }
    if best.is_empty() {
        return ClippedSpan {
            snippet: String::new(),
            truncated: true,
            start: None,
            end: None,
        };
    }
    ClippedSpan {
        snippet: best,
        truncated: true,
        start: Some(span.start + best_start),
        end: Some(span.start + best_end),
    }
}

fn clip_around_characters(
    text: &str,
    span: &Range<usize>,
    query: &str,
    query_terms: &HashSet<String>,
    max_tokens: usize,
    counter: &dyn TokenCounter,
) -> ClippedSpan {
    let segment = &text[span.clone()];
    let phrase_range = phrase_match_range(segment, query);
    let Some(selected) = phrase_range
        .clone()
        .or_else(|| lexical_match_range(segment, query_terms))
        .or_else(|| centered_character_range(segment))
    else {
        return ClippedSpan {
            snippet: String::new(),
            truncated: true,
            start: None,
            end: None,
        };
    };
    clip_around_preselected_characters(text, span, selected, max_tokens, counter)
}

fn clip_around_preselected_characters(
    text: &str,
    span: &Range<usize>,
    mut selected: Range<usize>,
    max_tokens: usize,
    counter: &dyn TokenCounter,
) -> ClippedSpan {
    let segment = &text[span.clone()];
    let mut best = None;

    loop {
        let candidate = render_clipped_segment(
            &segment[selected.clone()],
            selected.start > 0,
            selected.end < segment.len(),
        );
        if counter.count(&candidate) <= max_tokens {
            best = Some((candidate, selected.clone()));
        } else if best.is_none() {
            // The matched term itself is too large; a smaller character range
            // has a better chance of fitting than abandoning the hit. Start
            // from its center so an oversized phrase remains query-centered.
            let narrowed = centered_character_range_in(segment, &selected);
            if narrowed == selected {
                break;
            }
            selected = narrowed;
            continue;
        }

        let mut expansions = Vec::with_capacity(2);
        if selected.start > 0 {
            expansions.push(previous_character_boundary(segment, selected.start)..selected.end);
        }
        if selected.end < segment.len() {
            expansions.push(selected.start..next_character_boundary(segment, selected.end));
        }
        let Some(next) = expansions
            .into_iter()
            .filter(|range| {
                counter.count(&render_clipped_segment(
                    &segment[range.clone()],
                    range.start > 0,
                    range.end < segment.len(),
                )) <= max_tokens
            })
            .max_by_key(|range| range.end - range.start)
        else {
            break;
        };
        selected = next;
    }

    let Some((snippet, selected)) = best else {
        return ClippedSpan {
            snippet: String::new(),
            truncated: true,
            start: None,
            end: None,
        };
    };
    ClippedSpan {
        snippet,
        truncated: true,
        start: Some(span.start + selected.start),
        end: Some(span.start + selected.end),
    }
}

fn render_clipped_segment(segment: &str, omitted_left: bool, omitted_right: bool) -> String {
    // Dropping fence marker lines is preferable to emitting an unmatched fence
    // when a code block cannot fit as a whole.
    let mut value = segment.trim().to_string();
    value = remove_unmatched_fence_markers(&value);
    match (omitted_left, omitted_right) {
        (true, true) => format!("… {value} …"),
        (true, false) => format!("… {value}"),
        (false, true) => format!("{value} …"),
        (false, false) => value,
    }
}

fn remove_unmatched_fence_markers(value: &str) -> String {
    if value.matches("```").count() % 2 == 0 {
        return value.to_string();
    }
    value
        .lines()
        .filter_map(|line| {
            if is_standalone_fence_line(line) {
                None
            } else {
                // An inline marker cannot delimit a whole block. Keep the
                // prose/code line and remove only the unmatched marker.
                Some(line.replace("```", ""))
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn is_standalone_fence_line(line: &str) -> bool {
    let trimmed = line.trim();
    let Some(info) = trimmed.strip_prefix("```") else {
        return false;
    };
    // A regular fence is just the delimiter plus an optional compact info
    // string (for example `rust` or `rust,ignore`). Lines with prose or code
    // after an inline marker must remain visible in a clipped snippet.
    !info.contains("```") && !info.chars().any(char::is_whitespace)
}

fn lexical_anchor(text: &str, terms: &HashSet<String>) -> Option<usize> {
    lexical_match_range(text, terms).map(|range| range.start)
}

fn lexical_match_range(text: &str, terms: &HashSet<String>) -> Option<Range<usize>> {
    terms
        .iter()
        .filter_map(|term| {
            source_searchable_term(term).and_then(|term| phrase_match_range(text, &term))
        })
        .min_by_key(|range| range.start)
}

fn source_searchable_term(term: &str) -> Option<String> {
    if term.contains('\u{1f}') {
        let contiguous = term.replace('\u{1f}', "");
        (!contiguous.is_empty()).then_some(contiguous)
    } else {
        Some(term.to_string())
    }
}

/// Find a Unicode case-insensitive phrase without deriving offsets from a
/// case-folded allocation. Returned byte ranges always index `text` itself.
fn phrase_match_range(text: &str, phrase: &str) -> Option<Range<usize>> {
    let wanted = phrase.nfc().case_fold().collect::<Vec<_>>();
    if wanted.is_empty() {
        return None;
    }
    let folded = normalized_casefolded_source(text);
    (0..folded.len()).find_map(|start| {
        let matched = folded
            .iter()
            .skip(start)
            .zip(&wanted)
            .all(|(source, wanted)| source.value == *wanted);
        if !matched || folded.len().saturating_sub(start) < wanted.len() {
            return None;
        }
        let range = folded[start].start..folded[start + wanted.len() - 1].end;
        phrase_edges_are_bounded(text, &range, phrase).then_some(range)
    })
}

fn phrase_edges_are_bounded(text: &str, range: &Range<usize>, phrase: &str) -> bool {
    let first = phrase.chars().find(|ch| !ch.is_whitespace());
    let last = phrase.chars().rev().find(|ch| !ch.is_whitespace());
    let before_is_alphanumeric = text[..range.start]
        .chars()
        .next_back()
        .is_some_and(char::is_alphanumeric);
    let after_is_alphanumeric = text[range.end..]
        .chars()
        .next()
        .is_some_and(char::is_alphanumeric);
    let requires_left_boundary =
        first.is_some_and(|ch| ch.is_alphanumeric() && !is_unspaced_script_char(ch));
    let requires_right_boundary =
        last.is_some_and(|ch| ch.is_alphanumeric() && !is_unspaced_script_char(ch));
    (!requires_left_boundary || !before_is_alphanumeric)
        && (!requires_right_boundary || !after_is_alphanumeric)
}

#[derive(Debug, Clone, Copy)]
struct SourceFoldedChar {
    value: char,
    start: usize,
    end: usize,
}

fn normalized_casefolded_source(text: &str) -> Vec<SourceFoldedChar> {
    let mut normalized = Vec::<SourceFoldedChar>::new();
    let mut cluster = String::new();
    let mut cluster_start = 0usize;
    let mut cluster_end = 0usize;
    for (start, source) in text.char_indices() {
        let end = start + source.len_utf8();
        if canonical_combining_class(source) == 0
            && !cluster.is_empty()
            && !hangul_jamo_composes_with(cluster.chars().next_back(), source)
        {
            append_normalized_cluster(&mut normalized, &cluster, cluster_start, cluster_end);
            cluster.clear();
            cluster_start = start;
        } else if cluster.is_empty() {
            cluster_start = start;
        }
        cluster.push(source);
        cluster_end = end;
    }
    if !cluster.is_empty() {
        append_normalized_cluster(&mut normalized, &cluster, cluster_start, cluster_end);
    }

    normalized
        .into_iter()
        .flat_map(|mapped| {
            mapped
                .value
                .case_fold()
                .map(move |value| SourceFoldedChar { value, ..mapped })
        })
        .collect()
}

fn hangul_jamo_composes_with(previous: Option<char>, next: char) -> bool {
    let Some(previous) = previous else {
        return false;
    };
    (is_hangul_leading_jamo(previous) && is_hangul_vowel_jamo(next))
        || (is_hangul_vowel_jamo(previous) && is_hangul_trailing_jamo(next))
        || (is_hangul_lv_syllable(previous) && is_hangul_trailing_jamo(next))
}

fn is_hangul_leading_jamo(ch: char) -> bool {
    matches!(ch as u32, 0x1100..=0x115f | 0xa960..=0xa97c)
}

fn is_hangul_vowel_jamo(ch: char) -> bool {
    matches!(ch as u32, 0x1160..=0x11a7 | 0xd7b0..=0xd7c6)
}

fn is_hangul_trailing_jamo(ch: char) -> bool {
    matches!(ch as u32, 0x11a8..=0x11ff | 0xd7cb..=0xd7fb)
}

fn is_hangul_lv_syllable(ch: char) -> bool {
    let scalar = ch as u32;
    (0xac00..=0xd7a3).contains(&scalar) && (scalar - 0xac00) % 28 == 0
}

fn append_normalized_cluster(
    output: &mut Vec<SourceFoldedChar>,
    cluster: &str,
    start: usize,
    end: usize,
) {
    output.extend(
        cluster
            .nfc()
            .map(|value| SourceFoldedChar { value, start, end }),
    );
}

fn centered_character_range(text: &str) -> Option<Range<usize>> {
    let start = text
        .char_indices()
        .nth(text.chars().count() / 2)
        .map(|(index, _)| index)?;
    Some(start..next_character_boundary(text, start))
}

fn centered_character_range_in(text: &str, range: &Range<usize>) -> Range<usize> {
    let selected = &text[range.clone()];
    let offset = selected
        .char_indices()
        .nth(selected.chars().count() / 2)
        .map(|(index, _)| index)
        .unwrap_or(0);
    let start = range.start + offset;
    start..next_character_boundary(text, start)
}

fn previous_character_boundary(text: &str, index: usize) -> usize {
    text[..index]
        .char_indices()
        .next_back()
        .map(|(start, _)| start)
        .unwrap_or(0)
}

fn next_character_boundary(text: &str, index: usize) -> usize {
    index
        + text[index..]
            .chars()
            .next()
            .map(char::len_utf8)
            .unwrap_or(0)
}

fn nearest_boundary_left(text: &str, mut index: usize) -> usize {
    index = index.min(text.len());
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    while index > 0 {
        let previous = text[..index]
            .chars()
            .next_back()
            .expect("non-empty prefix has a final character");
        if previous.is_whitespace() {
            break;
        }
        index -= previous.len_utf8();
    }
    index
}

fn nearest_boundary_right(text: &str, mut index: usize) -> usize {
    index = index.min(text.len());
    while index < text.len() && !text.is_char_boundary(index) {
        index += 1;
    }
    while index < text.len() && !text[index..].starts_with(char::is_whitespace) {
        index += text[index..]
            .chars()
            .next()
            .map(char::len_utf8)
            .unwrap_or(0);
    }
    index
}

fn normalized_tokens(text: &str) -> HashSet<String> {
    let normalized = text.nfc().collect::<String>();
    let mut tokens = normalized
        .split(|ch: char| !ch.is_alphanumeric() && !is_unspaced_script_char(ch))
        .filter(|token| !token.is_empty())
        .flat_map(|token| {
            // Whitespace-delimited scripts retain word tokens. A run in a
            // script that normally omits spaces (CJK/Japanese/Korean) uses
            // order-sensitive character bigrams so similarity does not treat
            // a reordered character inventory as the same passage.
            if token.chars().any(is_unspaced_script_char) {
                unspaced_script_bigrams(token)
            } else {
                vec![token.nfc().case_fold().collect::<String>()]
            }
        })
        .collect::<HashSet<_>>();
    // Lexical fallback terms must remain source-searchable, so symbols use
    // their canonical scalar rather than a representation-only prefix.
    for symbol in normalized
        .chars()
        .filter(|ch| !ch.is_alphanumeric() && !ch.is_whitespace() && !is_unspaced_script_char(*ch))
    {
        tokens.insert(symbol.to_string().case_fold().collect::<String>());
    }
    if tokens.is_empty() {
        let canonical = text.nfc().case_fold().collect::<String>();
        if !canonical.is_empty() {
            tokens.insert(canonical);
        }
    }
    tokens
}

/// Tokens used exclusively for diversity and duplicate comparisons. Unspaced
/// scripts use order-sensitive bigram shingles with multiplicity, which keeps
/// shifted passages similar without collapsing a reordered passage to an
/// identical set inventory.
fn similarity_tokens(text: &str) -> HashMap<String, usize> {
    let mut tokens = HashMap::new();
    let normalized = text.nfc().collect::<String>();
    for token in normalized
        .split(|ch: char| !ch.is_alphanumeric() && !is_unspaced_script_char(ch))
        .filter(|token| !token.is_empty())
    {
        if token.chars().any(is_unspaced_script_char) {
            for bigram in unspaced_script_bigrams(token) {
                *tokens.entry(format!("cjk:{bigram}")).or_default() += 1;
            }
        } else {
            *tokens
                .entry(format!(
                    "word:{}",
                    token.nfc().case_fold().collect::<String>()
                ))
                .or_default() += 1;
        }
    }
    // Symbols carry semantic state alongside ordinary words (for example,
    // "deployment ✅" and "deployment ❌"). Include them in every
    // representation, not only in the symbol-only fallback, so duplicate
    // suppression cannot erase that distinction.
    for symbol in normalized
        .chars()
        .filter(|ch| !ch.is_alphanumeric() && !ch.is_whitespace())
    {
        let canonical = symbol.to_string().case_fold().collect::<String>();
        *tokens.entry(format!("symbol:{canonical}")).or_default() += 1;
    }
    if tokens.is_empty() {
        let canonical = text.nfc().case_fold().collect::<String>();
        if !canonical.is_empty() {
            tokens.insert(format!("symbol:{canonical}"), 1);
        }
    }
    tokens
}

fn unspaced_script_bigrams(token: &str) -> Vec<String> {
    let folded = token.nfc().case_fold().collect::<String>();
    let characters = folded.chars().map(|ch| ch.to_string()).collect::<Vec<_>>();
    if characters.len() <= 1 {
        return characters;
    }
    characters
        .windows(2)
        .map(|pair| format!("{}\u{1f}{}", pair[0], pair[1]))
        .collect()
}

fn is_unspaced_script_char(ch: char) -> bool {
    matches!(ch as u32,
        0x3040..=0x30ff // Hiragana and Katakana
        | 0x3400..=0x4dbf // CJK Unified Ideographs Extension A
        | 0x4e00..=0x9fff // CJK Unified Ideographs
        | 0xac00..=0xd7af // Hangul syllables
        | 0xf900..=0xfaff // CJK compatibility ideographs
        | 0x20000..=0x2ebef // supplementary CJK extensions
        | 0x0e00..=0x0e7f // Thai
        | 0x0e80..=0x0eff // Lao
        | 0x1000..=0x109f // Myanmar
        | 0x1780..=0x17ff // Khmer
    )
}

fn jaccard_similarity(left: &HashSet<String>, right: &HashSet<String>) -> f32 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let intersection = left.intersection(right).count() as f32;
    intersection / (left.len() + right.len() - intersection as usize) as f32
}

fn multiset_jaccard_similarity(
    left: &HashMap<String, usize>,
    right: &HashMap<String, usize>,
) -> f32 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let intersection = left
        .iter()
        .map(|(token, count)| (*count).min(right.get(token).copied().unwrap_or_default()))
        .sum::<usize>();
    let union = left
        .values()
        .sum::<usize>()
        .saturating_add(right.values().sum::<usize>())
        .saturating_sub(intersection);
    intersection as f32 / union as f32
}

fn hit_type_label(hit_type: SearchHitType) -> &'static str {
    match hit_type {
        SearchHitType::Note => "note",
        SearchHitType::Message => "message",
        SearchHitType::ConversationSummary => "conversation-summary",
    }
}

fn render_prompt_block(chunks: &[AugmentChunk]) -> String {
    if chunks.is_empty() {
        return String::new();
    }
    let mut out = String::from("<context>\n");
    for chunk in chunks {
        let title = chunk.title.as_deref().unwrap_or("(untitled)");
        out.push_str(&format!(
            "[C{}] [{}] {}\n{}\n\n",
            chunk.citation,
            hit_type_label(chunk.hit_type),
            title,
            chunk.snippet
        ));
    }
    out.push_str("</context>");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::ScopedSearchResult;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Debug)]
    struct ExactWords;
    impl TokenCounter for ExactWords {
        fn count(&self, text: &str) -> usize {
            text.split_whitespace().count()
        }
        fn mode(&self) -> TokenCountMode {
            TokenCountMode::Exact
        }
    }

    #[derive(Debug)]
    struct CountingBytes(Arc<AtomicUsize>);

    impl TokenCounter for CountingBytes {
        fn count(&self, text: &str) -> usize {
            self.0.fetch_add(1, Ordering::Relaxed);
            text.len()
        }

        fn mode(&self) -> TokenCountMode {
            TokenCountMode::Exact
        }
    }

    fn hit(id: &str, score: f32, content: &str) -> ScopedSearchResult {
        ScopedSearchResult {
            hit_type: SearchHitType::Note,
            id: id.into(),
            title: Some("title".into()),
            content: content.into(),
            created_at: None,
            source_uri: None,
            score,
            fusion: graphrag_db::fusion::FusionEvidence {
                fused_score: score,
                ..Default::default()
            },
            conversation_uuid: None,
            message_index: None,
            role: None,
        }
    }

    fn options() -> AugmentOptions {
        AugmentOptions {
            max_chunks: 4,
            max_total_tokens: 100,
            max_chunk_tokens: 30,
            ..Default::default()
        }
    }

    #[test]
    fn fallback_is_conservative_and_exact_mode_is_reported() {
        assert!(ConservativeTokenCounter.count("extraordinary") > 1);
        assert_eq!(ConservativeTokenCounter.count("sha256:deadBEEF"), 15);
        let context = build_augment_context_from_hits(
            "q".into(),
            SearchScope::Notes,
            None,
            vec![hit("n:1", 1.0, "one two")],
            options().with_token_counter(Arc::new(ExactWords)),
            0,
        );
        assert_eq!(context.diagnostics.token_count_mode, TokenCountMode::Exact);
    }

    #[test]
    fn accounts_for_rendered_header_and_never_exceeds_budget() {
        let mut options = options().with_token_counter(Arc::new(ExactWords));
        options.max_total_tokens = 7;
        let context = build_augment_context_from_hits(
            "q".into(),
            SearchScope::Notes,
            None,
            vec![hit("n:1", 1.0, "one two three")],
            options,
            0,
        );
        assert!(context.total_tokens <= 7);
        assert_eq!(
            context.total_tokens,
            ExactWords.count(&context.render_prompt_block())
        );
        assert!(context.diagnostics.header_tokens > 0 || context.chunks.is_empty());
    }

    #[test]
    fn fixed_rendered_overhead_short_circuits_huge_chunk_budget_without_retries() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut constrained = options();
        constrained.max_total_tokens = 10;
        constrained.max_chunk_tokens = 1_000_000;
        constrained.token_counter = Arc::new(CountingBytes(calls.clone()));

        let context = build_augment_context_from_hits(
            "needle".into(),
            SearchScope::Notes,
            None,
            vec![hit(
                "n:oversized",
                0.9,
                "needle with a very large configured chunk budget",
            )],
            constrained,
            0,
        );

        assert!(context.chunks.is_empty());
        // A fixed header already exceeds the budget. The old decrementing
        // retry loop would have attempted up to one million candidate caps.
        assert!(calls.load(Ordering::Relaxed) < 10);
    }

    #[test]
    fn candidate_caches_full_similarity_features_at_pool_construction() {
        let text = "repeated context tokens need one complete representation";
        let candidate = Candidate::from_hit(hit("n:cache", 0.9, text));

        assert_eq!(candidate.full_tokens, similarity_tokens(text));
    }

    #[test]
    fn late_query_span_is_preferred_without_invalid_utf8_or_unbalanced_fences() {
        let text = "irrelevant first sentence. ```rust\nlet early = true;\n```\nmore filler. the unicode café target phrase is here. trailing filler.";
        let clipped = clip_query_aware(text, "café target", 20, &ConservativeTokenCounter);
        assert!(clipped.snippet.contains("café target"));
        assert_eq!(clipped.snippet.matches("```").count() % 2, 0);
        assert!(std::str::from_utf8(clipped.snippet.as_bytes()).is_ok());
    }

    #[test]
    fn near_duplicates_are_suppressed_and_selection_is_deterministic() {
        let hits = vec![
            hit("n:1", 0.9, "rust ownership borrowing lifetimes guide"),
            hit("n:2", 0.8, "rust ownership borrowing lifetimes reference"),
            hit("n:3", 0.7, "database indexing and query planning"),
        ];
        let mut duplicate_options = options();
        duplicate_options.near_duplicate_threshold = 0.6;
        let first = build_augment_context_from_hits(
            "rust".to_string(),
            SearchScope::Notes,
            None,
            hits.clone(),
            duplicate_options.clone(),
            0,
        );
        let second = build_augment_context_from_hits(
            "rust".to_string(),
            SearchScope::Notes,
            None,
            hits,
            duplicate_options,
            0,
        );
        assert_eq!(
            first
                .chunks
                .iter()
                .map(|chunk| &chunk.id)
                .collect::<Vec<_>>(),
            second
                .chunks
                .iter()
                .map(|chunk| &chunk.id)
                .collect::<Vec<_>>()
        );
        assert!(first.diagnostics.dropped_near_duplicates >= 1);
    }

    #[test]
    fn long_identical_records_are_compared_as_full_records_not_clipped_snippets() {
        let long_record = [
            "introductory material repeated across the record",
            "the query target appears in this deliberately distant sentence",
            "additional unique-looking tail material that does not fit the snippet cap",
        ]
        .join(". ");
        let mut duplicate_options = options();
        duplicate_options.max_chunk_tokens = 16;
        duplicate_options.near_duplicate_threshold = 0.95;
        let context = build_augment_context_from_hits(
            "target".into(),
            SearchScope::Notes,
            None,
            vec![hit("n:1", 0.9, &long_record), hit("n:2", 0.8, &long_record)],
            duplicate_options,
            0,
        );

        assert_eq!(context.chunks.len(), 1);
        assert_eq!(context.diagnostics.dropped_near_duplicates, 1);
    }

    #[test]
    fn near_identical_unspaced_cjk_records_share_granular_tokens() {
        let first = "天地玄黃宇宙洪荒日月盈昃辰宿列張";
        let second = "天地玄黃宇宙洪荒日月盈昃辰宿列章";
        let first_tokens = normalized_tokens(first);
        let second_tokens = normalized_tokens(second);
        assert!(jaccard_similarity(&first_tokens, &second_tokens) >= 0.85);
        assert_eq!(normalized_tokens("alpha beta").len(), 2);
        assert_eq!(normalized_tokens("привет мир").len(), 2);

        let mut duplicate_options = options();
        duplicate_options.near_duplicate_threshold = 0.85;
        let context = build_augment_context_from_hits(
            "天地".into(),
            SearchScope::Notes,
            None,
            vec![hit("n:cjk-a", 0.9, first), hit("n:cjk-b", 0.8, second)],
            duplicate_options,
            0,
        );
        assert_eq!(context.chunks.len(), 1);
        assert_eq!(context.diagnostics.dropped_near_duplicates, 1);
    }

    #[test]
    fn thai_near_copies_use_granular_shingles_for_similarity() {
        let first = "ภาษาไทยทดสอบระบบการค้นหา";
        let second = "ภาษาไทยทดสอบระบบการค้นหข";
        assert!(
            multiset_jaccard_similarity(&similarity_tokens(first), &similarity_tokens(second))
                >= 0.85
        );

        let mut duplicate_options = options();
        duplicate_options.max_total_tokens = 300;
        duplicate_options.max_chunk_tokens = 100;
        let context = build_augment_context_from_hits(
            "ระบบ".into(),
            SearchScope::Notes,
            None,
            vec![hit("n:thai-a", 0.9, first), hit("n:thai-b", 0.8, second)],
            duplicate_options,
            0,
        );
        assert_eq!(context.chunks.len(), 1);
        assert_eq!(context.diagnostics.dropped_near_duplicates, 1);
    }

    #[test]
    fn reordered_unspaced_cjk_records_are_not_treated_as_duplicates() {
        let first = "天地玄黃宇宙洪荒";
        let second = "洪荒宇宙玄黃天地";
        assert!(jaccard_similarity(&normalized_tokens(first), &normalized_tokens(second)) < 0.85);

        let context = build_augment_context_from_hits(
            "天地".into(),
            SearchScope::Notes,
            None,
            vec![hit("n:cjk-a", 0.9, first), hit("n:cjk-b", 0.8, second)],
            options(),
            0,
        );
        assert_eq!(context.chunks.len(), 2);
        assert_eq!(context.diagnostics.dropped_near_duplicates, 0);
    }

    #[test]
    fn repeated_cjk_bigrams_keep_position_and_order_for_similarity() {
        let first = "甲乙甲乙";
        let second = "乙甲乙甲";
        assert_eq!(
            jaccard_similarity(&normalized_tokens(first), &normalized_tokens(second)),
            1.0
        );
        assert!(
            multiset_jaccard_similarity(&similarity_tokens(first), &similarity_tokens(second))
                < 0.85
        );

        let mut duplicate_options = options();
        duplicate_options.max_total_tokens = 300;
        duplicate_options.near_duplicate_threshold = 0.85;
        let context = build_augment_context_from_hits(
            "甲乙".into(),
            SearchScope::Notes,
            None,
            vec![hit("n:first", 0.9, first), hit("n:second", 0.8, second)],
            duplicate_options,
            0,
        );
        assert_eq!(context.chunks.len(), 2);
        assert_eq!(context.diagnostics.dropped_near_duplicates, 0);
    }

    #[test]
    fn shifted_cjk_passages_remain_near_duplicates() {
        let original = "天地玄黃宇宙洪荒日月盈昃";
        let prefixed = "序".to_string() + original;
        assert!(
            multiset_jaccard_similarity(
                &similarity_tokens(original),
                &similarity_tokens(&prefixed)
            ) >= 0.85
        );

        let mut duplicate_options = options();
        duplicate_options.max_total_tokens = 300;
        duplicate_options.near_duplicate_threshold = 0.85;
        let context = build_augment_context_from_hits(
            "天地".into(),
            SearchScope::Notes,
            None,
            vec![
                hit("n:original", 0.9, original),
                hit("n:prefixed", 0.8, &prefixed),
            ],
            duplicate_options,
            0,
        );
        assert_eq!(context.chunks.len(), 1);
        assert_eq!(context.diagnostics.dropped_near_duplicates, 1);
    }

    #[test]
    fn zero_novelty_preserves_incoming_scoped_tie_order_for_citations() {
        // This is the retrieval order established by search's scoped tie
        // contract: hit type first, then record ID, after fused-score ties.
        let mut note_a = hit("note:a", 0.5, "first distinct context");
        let mut note_b = hit("note:b", 0.5, "second distinct context");
        let mut message_z = hit("message:z", 0.5, "third distinct context");
        note_a.hit_type = SearchHitType::Note;
        note_b.hit_type = SearchHitType::Note;
        message_z.hit_type = SearchHitType::Message;
        let mut packing_options = options();
        packing_options.max_chunks = 3;
        packing_options.max_total_tokens = 300;
        packing_options.max_chunk_tokens = 80;
        packing_options.novelty_weight = 0.0;
        packing_options.near_duplicate_threshold = 1.0;
        let context = build_augment_context_from_hits(
            "context".into(),
            SearchScope::All,
            None,
            vec![note_a, note_b, message_z],
            packing_options,
            0,
        );

        assert_eq!(
            context
                .chunks
                .iter()
                .map(|chunk| (chunk.citation, chunk.id.as_str()))
                .collect::<Vec<_>>(),
            vec![(1, "note:a"), (2, "note:b"), (3, "message:z")]
        );
    }

    #[test]
    fn identical_symbol_only_records_are_near_duplicate_suppressed() {
        let symbols = "🎉 !!! ✨";
        assert!(!normalized_tokens(symbols).is_empty());
        let context = build_augment_context_from_hits(
            "celebration".into(),
            SearchScope::Notes,
            None,
            vec![
                hit("n:symbol-a", 0.9, symbols),
                hit("n:symbol-b", 0.8, symbols),
            ],
            options(),
            0,
        );
        assert_eq!(context.chunks.len(), 1);
        assert_eq!(context.diagnostics.dropped_near_duplicates, 1);
    }

    #[test]
    fn symbols_remain_distinct_when_records_also_have_words() {
        let succeeded = "deployment ✅";
        let failed = "deployment ❌";
        assert!(
            multiset_jaccard_similarity(&similarity_tokens(succeeded), &similarity_tokens(failed))
                < 0.85
        );

        let context = build_augment_context_from_hits(
            "deployment".into(),
            SearchScope::Notes,
            None,
            vec![
                hit("n:deployment-ok", 0.9, succeeded),
                hit("n:deployment-failed", 0.8, failed),
            ],
            options(),
            0,
        );
        assert_eq!(context.chunks.len(), 2);
        assert_eq!(context.diagnostics.dropped_near_duplicates, 0);
    }

    #[test]
    fn lexical_fallback_uses_source_searchable_symbols_and_full_casefolding() {
        let symbol_terms = normalized_tokens("missing ❌");
        assert!(symbol_terms.contains("❌"));
        let symbol_range = lexical_match_range("deployment ❌ is the late signal", &symbol_terms)
            .expect("symbol term should be searchable in the source");
        assert_eq!(&"deployment ❌ is the late signal"[symbol_range], "❌");

        let casefolded_terms = normalized_tokens("STRASSE");
        let source = "prefix Straße suffix";
        let range = lexical_match_range(source, &casefolded_terms)
            .expect("lexical fallback should share phrase matching case folding");
        assert_eq!(&source[range], "Straße");

        let clipped = clip_query_aware(
            "early ✅ update. late deployment ❌ evidence.",
            "missing ❌",
            30,
            &ConservativeTokenCounter,
        );
        assert!(clipped.snippet.contains("❌"));
        assert!(clipped.snippet.contains("late deployment"));
    }

    #[test]
    fn casefolded_matching_returns_original_kelvin_sign_byte_offsets() {
        let text = "prefix Kelvin marker suffix";
        let range = phrase_match_range(text, "kelvin marker").unwrap();
        assert_eq!(&text[range], "Kelvin marker");
    }

    #[test]
    fn casefolded_match_accepts_a_query_ending_inside_a_scalar_expansion() {
        let text = "İ marker";
        let range = phrase_match_range(text, "i").unwrap();
        assert_eq!(&text[range.clone()], "İ");
        assert_eq!(range.end, "İ".len());
    }

    #[test]
    fn full_casefolding_matches_sharp_s_and_final_sigma_in_clipped_text() {
        assert_eq!(
            &"The Straße marker"[phrase_match_range("The Straße marker", "STRASSE").unwrap()],
            "Straße"
        );
        assert_eq!(&"σος"[phrase_match_range("σος", "ΣΟΣ").unwrap()], "σος");
        let clipped = clip_query_aware(
            "intro sentence. This deliberately oversized sentence contains the Straße marker and enough trailing words that it must be clipped before ending. final sentence.",
            "STRASSE marker",
            25,
            &ConservativeTokenCounter,
        );
        assert!(clipped.truncated);
        assert!(clipped.snippet.contains("Straße marker"));
        assert!(ConservativeTokenCounter.count(&clipped.snippet) <= 25);
    }

    #[test]
    fn canonical_unicode_forms_match_and_deduplicate() {
        let composed = "café notes";
        let decomposed = "cafe\u{301} notes";
        let range = phrase_match_range(decomposed, "CAFÉ").unwrap();
        assert_eq!(&decomposed[range], "cafe\u{301}");
        assert_eq!(similarity_tokens(composed), similarity_tokens(decomposed));

        let context = build_augment_context_from_hits(
            "café".into(),
            SearchScope::Notes,
            None,
            vec![
                hit("n:composed", 0.9, composed),
                hit("n:decomposed", 0.8, decomposed),
            ],
            options(),
            0,
        );
        assert_eq!(context.chunks.len(), 1);
        assert_eq!(context.diagnostics.dropped_near_duplicates, 1);
    }

    #[test]
    fn normalized_source_mapping_composes_hangul_jamo_with_original_offsets() {
        let text = "prefix 가 marker suffix";
        let range = phrase_match_range(text, "가 marker").unwrap();
        assert_eq!(&text[range], "가 marker");
    }

    #[test]
    fn normalized_source_mapping_composes_hangul_lv_syllable_and_trailing_jamo() {
        let text = "prefix 각 marker suffix";
        let range = phrase_match_range(text, "각 marker").unwrap();
        assert_eq!(&text[range], "각 marker");
    }

    #[test]
    fn normalized_source_mapping_handles_long_text_without_prefix_rescanning() {
        let text = format!("{}cafe\u{301} marker", "filler ".repeat(20_000));
        let range = phrase_match_range(&text, "CAFÉ marker").unwrap();
        assert_eq!(&text[range], "cafe\u{301} marker");
    }

    #[test]
    fn oversized_phrase_window_is_clipped_instead_of_dropped() {
        let text = "prefix oversized phrase target afterword";
        let clipped = clip_query_aware(
            text,
            "oversized phrase target",
            10,
            &ConservativeTokenCounter,
        );
        assert!(clipped.truncated);
        assert!(!clipped.snippet.is_empty());
        assert!(ConservativeTokenCounter.count(&clipped.snippet) <= 10);
        assert!(std::str::from_utf8(clipped.snippet.as_bytes()).is_ok());
    }

    #[test]
    fn whitespace_phrase_matching_rejects_substrings_but_cjk_allows_them() {
        assert!(phrase_match_range("partial", "art").is_none());
        assert!(phrase_match_range("cart 東京", "art 東京").is_none());
        let mixed = "art 東京 guide";
        assert_eq!(
            &mixed[phrase_match_range(mixed, "art 東京").unwrap()],
            "art 東京"
        );
        let cjk = "前綴目標片段後綴";
        assert_eq!(
            &cjk[phrase_match_range(cjk, "目標片段").unwrap()],
            "目標片段"
        );

        let clipped = clip_query_aware(
            "partial appears early. art appears later as a standalone term.",
            "art",
            30,
            &ConservativeTokenCounter,
        );
        assert!(clipped.snippet.contains("art"));
        assert!(!clipped.snippet.contains("partial"));
    }

    #[test]
    fn full_cjk_query_beats_an_earlier_shared_character() {
        let text = "目甲乙丙丁戊己庚辛壬癸目標片段後續說明";
        let clipped = clip_query_aware(text, "目標片段", 20, &ConservativeTokenCounter);
        assert!(clipped.snippet.contains("目標片段"));
        assert!(ConservativeTokenCounter.count(&clipped.snippet) <= 20);
    }

    #[test]
    fn partial_cjk_overlap_anchors_on_a_source_searchable_bigram() {
        let clipped = clip_query_aware(
            "前置說明甲乙後置內容",
            "甲乙丙",
            20,
            &ConservativeTokenCounter,
        );
        assert!(clipped.snippet.contains("甲乙"));
        assert!(ConservativeTokenCounter.count(&clipped.snippet) <= 20);
    }

    #[test]
    fn unmatched_fence_cleanup_preserves_lines_with_inline_markers() {
        let cleaned = remove_unmatched_fence_markers("intro ``` marker\n```rust\nlet x = 1;\n```");
        assert!(cleaned.contains("intro  marker"));
        assert!(cleaned.contains("let x = 1;"));
        assert!(!cleaned.contains("```"));
    }

    #[test]
    fn clipped_span_offsets_match_the_last_fitting_window() {
        let text = "alpha beta gamma target delta epsilon zeta eta theta iota";
        let clipped = clip_query_aware(text, "target", 5, &ExactWords);
        let selected = &text[clipped.start.unwrap()..clipped.end.unwrap()];
        assert!(clipped.truncated);
        assert_eq!(
            clipped.snippet.trim().trim_matches('…').trim(),
            selected.trim()
        );
        assert!(ExactWords.count(&clipped.snippet) <= 5);
    }

    #[test]
    fn fallback_counts_non_ascii_per_character_and_respects_budget() {
        assert_eq!(ConservativeTokenCounter.count("你好世界"), 12);
        assert_eq!(ConservativeTokenCounter.count("café"), 5);
        assert_eq!(ConservativeTokenCounter.count("😀"), 4);
        let mut multilingual_options = options();
        multilingual_options.max_total_tokens = 80;
        multilingual_options.max_chunk_tokens = 20;
        let context = build_augment_context_from_hits(
            "搜尋目標".into(),
            SearchScope::Notes,
            None,
            vec![hit(
                "n:cjk",
                0.9,
                "這是一段很長的中文內容，包含搜尋目標以及額外說明文字。",
            )],
            multilingual_options,
            0,
        );
        assert!(context.total_tokens <= 80);
        assert_eq!(context.chunks.len(), 1);
        assert!(context.chunks[0].snippet.contains("搜尋目標"));
        assert!(context.chunks[0].approx_tokens <= 20);
        assert!(std::str::from_utf8(context.chunks[0].snippet.as_bytes()).is_ok());
    }

    #[test]
    fn emoji_heavy_context_stays_within_the_hard_budget() {
        let mut emoji_options = options();
        emoji_options.max_total_tokens = 80;
        emoji_options.max_chunk_tokens = 16;
        let context = build_augment_context_from_hits(
            "😀😀".into(),
            SearchScope::Notes,
            None,
            vec![hit("n:emoji", 0.9, "😀😀😀😀😀😀")],
            emoji_options,
            0,
        );
        assert_eq!(context.chunks.len(), 1);
        assert!(context.total_tokens <= 80);
        assert!(std::str::from_utf8(context.chunks[0].snippet.as_bytes()).is_ok());
    }

    #[test]
    fn whitespace_heavy_context_is_counted_and_stays_within_budget() {
        assert_eq!(ConservativeTokenCounter.count("a          b"), 12);
        let mut whitespace_options = options();
        whitespace_options.max_total_tokens = 100;
        whitespace_options.max_chunk_tokens = 30;
        let context = build_augment_context_from_hits(
            "needle".into(),
            SearchScope::Notes,
            None,
            vec![hit(
                "n:whitespace",
                0.9,
                "needle\n                        deeply indented context that must be clipped",
            )],
            whitespace_options,
            0,
        );
        assert_eq!(context.chunks.len(), 1);
        assert!(context.total_tokens <= 100);
        assert_eq!(
            context.total_tokens,
            ConservativeTokenCounter.count(&context.render_prompt_block())
        );
    }

    #[test]
    fn diversity_never_bypasses_minimum_relevance_and_empty_context_is_safe() {
        let mut options = options();
        options.min_relevance = 0.5;
        options.max_total_tokens = 1;
        let context = build_augment_context_from_hits(
            "q".into(),
            SearchScope::Notes,
            None,
            vec![
                hit("n:1", 0.4, "unrelated"),
                hit("n:2", 0.9, "relevant content"),
            ],
            options,
            0,
        );
        assert!(context.chunks.is_empty());
        assert_eq!(context.total_tokens, 0);
        assert_eq!(context.render_prompt_block(), "");
        assert_eq!(context.diagnostics.dropped_for_relevance, 1);
    }

    #[test]
    fn normalized_relevance_prevents_low_score_novelty_from_overtaking_retrieval() {
        let mut selection_options = options();
        selection_options.max_chunks = 3;
        selection_options.max_total_tokens = 300;
        selection_options.max_chunk_tokens = 80;
        selection_options.novelty_weight = 0.25;
        selection_options.near_duplicate_threshold = 1.0;
        let context = build_augment_context_from_hits(
            "rust".into(),
            SearchScope::Notes,
            None,
            vec![
                hit("n:best", 0.9, "rust ownership borrowing lifetimes"),
                hit("n:relevant", 0.8, "rust ownership borrowing patterns"),
                hit("n:novel", 0.01, "unrelated gardening and recipes"),
            ],
            selection_options,
            0,
        );
        assert_eq!(
            context
                .chunks
                .iter()
                .map(|chunk| chunk.id.as_str())
                .collect::<Vec<_>>(),
            vec!["n:best", "n:relevant", "n:novel"]
        );
    }
}
