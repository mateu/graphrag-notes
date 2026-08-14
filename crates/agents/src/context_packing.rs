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
/// Word-like runs are split into three-character pieces and punctuation is
/// charged separately. This modestly over-counts common BPE tokenizers and is
/// suitable for enforcing a hard budget in fallback mode.
#[derive(Debug, Default)]
pub struct ConservativeTokenCounter;

impl TokenCounter for ConservativeTokenCounter {
    fn count(&self, text: &str) -> usize {
        let mut count = 0usize;
        let mut word_len = 0usize;
        for ch in text.chars() {
            if ch.is_alphanumeric() || ch == '_' {
                word_len += 1;
            } else {
                if word_len > 0 {
                    count += word_len.div_ceil(3);
                    word_len = 0;
                }
                if !ch.is_whitespace() {
                    count += 1;
                }
            }
        }
        count + word_len.div_ceil(3)
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

    let mut chunks = Vec::new();
    let mut dropped_near_duplicates = 0usize;
    let mut dropped_for_budget = 0usize;

    while !candidates.is_empty() && chunks.len() < options.max_chunks {
        let selected_tokens = chunks
            .iter()
            .map(|chunk: &AugmentChunk| normalized_tokens(&chunk.snippet))
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
        if chunks.iter().any(|chunk| {
            jaccard_similarity(&tokens, &normalized_tokens(&chunk.snippet))
                >= options.near_duplicate_threshold
        }) {
            dropped_near_duplicates += 1;
            continue;
        }

        let citation = chunks.len() + 1;
        let Some(chunk) = fit_candidate(&candidate, &query, citation, &chunks, &options) else {
            dropped_for_budget += 1;
            continue;
        };
        chunks.push(chunk);
    }

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
            let matches = query_terms.intersection(&terms).count();
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
        return clip_around_match(text, &spans[best_index], &query_terms, max_tokens, counter);
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
    query_terms: &HashSet<String>,
    max_tokens: usize,
    counter: &dyn TokenCounter,
) -> ClippedSpan {
    let segment = &text[span.clone()];
    let anchor = lexical_anchor(segment, query_terms).unwrap_or(segment.len() / 2);
    let mut start = nearest_boundary_left(segment, anchor);
    let mut end = nearest_boundary_right(segment, anchor);
    if start == end {
        end = segment.len();
    }
    let mut best = String::new();
    loop {
        let candidate =
            render_clipped_segment(&segment[start..end], start > 0, end < segment.len());
        if counter.count(&candidate) > max_tokens {
            break;
        }
        best = candidate;
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
        start: Some(span.start + start),
        end: Some(span.start + end),
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
    let lower = text.to_lowercase();
    terms.iter().filter_map(|term| lower.find(term)).min()
}

fn nearest_boundary_left(text: &str, mut index: usize) -> usize {
    index = index.min(text.len());
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    while index > 0 && !text[..index].ends_with(char::is_whitespace) {
        index -= 1;
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
        .map(|token| token.to_lowercase())
        .collect()
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
        let clipped = clip_query_aware(text, "café target", 10, &ConservativeTokenCounter);
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
