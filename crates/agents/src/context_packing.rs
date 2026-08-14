//! Deterministic, budget-safe prompt-context packing.
//!
//! This module intentionally has no network or model-download path. A caller
//! with a locally available provider tokenizer can inject it through
//! [`TokenCounter`]; the default is a conservative, deterministic estimate.

use crate::search::{ScopedSearchResult, SearchHitType, SearchScope};
use chrono::{DateTime, Utc};
use serde::Serialize;
use std::collections::HashSet;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

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
/// Every alphanumeric scalar and punctuation mark is charged separately. This
/// deliberately treats arbitrary ASCII identifiers (hashes, URLs, and opaque
/// model tokens) as a worst case and avoids under-counting scripts that do not
/// use whitespace. This is suitable for enforcing a hard budget in fallback
/// mode.
#[derive(Debug, Default)]
pub struct ConservativeTokenCounter;

impl TokenCounter for ConservativeTokenCounter {
    fn count(&self, text: &str) -> usize {
        let mut count = 0usize;
        for ch in text.chars() {
            if ch.is_alphanumeric() || ch == '_' || !ch.is_whitespace() {
                count += 1;
            }
        }
        count
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
    /// Weight of novelty in greedy MMR-style selection. `0.0` preserves score
    /// ordering; `1.0` is allowed but `min_relevance` still gates candidates.
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
    mut hits: Vec<ScopedSearchResult>,
    options: AugmentOptions,
    dropped_for_entity_filter: usize,
) -> AugmentContext {
    hits.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
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
        if hit.score < options.min_relevance {
            dropped_for_relevance += 1;
            continue;
        }
        let mut candidate = Candidate::from_hit(hit);
        candidate.rank = rank;
        candidates.push(candidate);
    }

    let mut selected = Vec::new();
    let mut dropped_near_duplicates = 0usize;
    let mut dropped_for_budget = 0usize;

    while !candidates.is_empty() && selected.len() < options.max_chunks {
        let selected_tokens = selected
            .iter()
            .map(|selected: &SelectedChunk| selected.full_tokens.clone())
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

        let tokens = normalized_tokens(candidate.content());
        if selected.iter().any(|selected| {
            jaccard_similarity(&tokens, &selected.full_tokens) >= options.near_duplicate_threshold
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
            full_tokens: tokens,
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
}

#[derive(Debug)]
struct SelectedChunk {
    chunk: AugmentChunk,
    /// The full untruncated record token set. Diversity and duplicate checks
    /// deliberately never compare a new record to only a selected snippet.
    full_tokens: HashSet<String>,
}

impl Candidate {
    fn from_hit(hit: ScopedSearchResult) -> Self {
        let content_start = hit.content.len() - hit.content.trim_start().len();
        Self {
            hit,
            content_start,
            rank: 0,
        }
    }

    fn content(&self) -> &str {
        self.hit.content.trim()
    }
}

fn selection_score(
    candidate: &Candidate,
    selected: &[HashSet<String>],
    novelty_weight: f32,
) -> f32 {
    let relevance = candidate.hit.score.clamp(0.0, 1.0);
    let candidate_tokens = normalized_tokens(candidate.content());
    let novelty = selected
        .iter()
        .map(|chosen| 1.0 - jaccard_similarity(&candidate_tokens, chosen))
        .fold(1.0_f32, f32::min);
    let novelty_weight = novelty_weight.clamp(0.0, 1.0);
    relevance * (1.0 - novelty_weight) + novelty * novelty_weight
}

fn fit_candidate(
    candidate: &Candidate,
    query: &str,
    citation: usize,
    chunks: &[AugmentChunk],
    options: &AugmentOptions,
) -> Option<AugmentChunk> {
    let counter = options.token_counter.as_ref();
    let mut cap = options.max_chunk_tokens;
    while cap > 0 {
        let clipped = clip_query_aware(candidate.content(), query, cap, counter);
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
        cap -= 1;
    }
    None
}

#[derive(Debug)]
struct ClippedSpan {
    snippet: String,
    truncated: bool,
    start: Option<usize>,
    end: Option<usize>,
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
        if line.trim_start().starts_with("```") {
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
    let Some(mut selected) = phrase_match_range(segment, query)
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
            // has a better chance of fitting than abandoning the hit.
            let single_character_end = next_character_boundary(segment, selected.start);
            if selected.end == single_character_end {
                break;
            }
            selected.end = single_character_end;
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
        .filter(|line| !line.contains("```"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn lexical_anchor(text: &str, terms: &HashSet<String>) -> Option<usize> {
    lexical_match_range(text, terms).map(|range| range.start)
}

fn lexical_match_range(text: &str, terms: &HashSet<String>) -> Option<Range<usize>> {
    terms
        .iter()
        .filter_map(|term| phrase_match_range(text, term))
        .min_by_key(|range| range.start)
}

/// Find a Unicode case-insensitive phrase without deriving offsets from a
/// case-folded allocation. Returned byte ranges always index `text` itself.
fn phrase_match_range(text: &str, phrase: &str) -> Option<Range<usize>> {
    let wanted = phrase
        .chars()
        .flat_map(char::to_lowercase)
        .collect::<Vec<_>>();
    if wanted.is_empty() {
        return None;
    }
    text.char_indices().find_map(|(start, _)| {
        casefolded_prefix_end(&text[start..], &wanted).map(|end| start..start + end)
    })
}

fn casefolded_prefix_end(text: &str, wanted: &[char]) -> Option<usize> {
    let mut matched = 0usize;
    for (offset, ch) in text.char_indices() {
        for folded in ch.to_lowercase() {
            if wanted.get(matched) != Some(&folded) {
                return None;
            }
            matched += 1;
        }
        if matched == wanted.len() {
            return Some(offset + ch.len_utf8());
        }
    }
    None
}

fn centered_character_range(text: &str) -> Option<Range<usize>> {
    let start = text
        .char_indices()
        .nth(text.chars().count() / 2)
        .map(|(index, _)| index)?;
    Some(start..next_character_boundary(text, start))
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
    text.split(|ch: char| !ch.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .flat_map(|token| {
            // Whitespace-delimited scripts retain word tokens. A run in a
            // script that normally omits spaces (CJK/Japanese/Korean) uses
            // order-sensitive character bigrams so similarity does not treat
            // a reordered character inventory as the same passage.
            if token.chars().any(is_unspaced_script_char) {
                unspaced_script_bigrams(token)
            } else {
                vec![token.to_lowercase()]
            }
        })
        .collect()
}

fn unspaced_script_bigrams(token: &str) -> Vec<String> {
    let characters = token
        .chars()
        .map(|ch| ch.to_lowercase().to_string())
        .collect::<Vec<_>>();
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
    )
}

fn jaccard_similarity(left: &HashSet<String>, right: &HashSet<String>) -> f32 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let intersection = left.intersection(right).count() as f32;
    intersection / (left.len() + right.len() - intersection as usize) as f32
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

    fn hit(id: &str, score: f32, content: &str) -> ScopedSearchResult {
        ScopedSearchResult {
            hit_type: SearchHitType::Note,
            id: id.into(),
            title: Some("title".into()),
            content: content.into(),
            created_at: None,
            source_uri: None,
            score,
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
    fn late_query_span_is_preferred_without_invalid_utf8_or_unbalanced_fences() {
        let text = "irrelevant first sentence. ```rust\nlet early = true;\n```\nmore filler. the unicode café target phrase is here. trailing filler.";
        let clipped = clip_query_aware(text, "café target", 18, &ConservativeTokenCounter);
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
        duplicate_options.max_chunk_tokens = 8;
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
    fn casefolded_matching_returns_original_kelvin_sign_byte_offsets() {
        let text = "prefix Kelvin marker suffix";
        let range = phrase_match_range(text, "kelvin marker").unwrap();
        assert_eq!(&text[range], "Kelvin marker");
    }

    #[test]
    fn full_cjk_query_beats_an_earlier_shared_character() {
        let text = "目甲乙丙丁戊己庚辛壬癸目標片段後續說明";
        let clipped = clip_query_aware(text, "目標片段", 6, &ConservativeTokenCounter);
        assert!(clipped.snippet.contains("目標片段"));
        assert!(ConservativeTokenCounter.count(&clipped.snippet) <= 6);
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
        assert_eq!(ConservativeTokenCounter.count("你好世界"), 4);
        assert_eq!(ConservativeTokenCounter.count("café"), 4);
        let mut multilingual_options = options();
        multilingual_options.max_total_tokens = 40;
        multilingual_options.max_chunk_tokens = 6;
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
        assert!(context.total_tokens <= 40);
        assert_eq!(context.chunks.len(), 1);
        assert!(context.chunks[0].snippet.contains("搜尋目標"));
        assert!(context.chunks[0].approx_tokens <= 6);
        assert!(std::str::from_utf8(context.chunks[0].snippet.as_bytes()).is_ok());
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
}
