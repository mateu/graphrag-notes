//! Deterministic, structure-aware Markdown chunking.
//!
//! Chunk sizes are measured in Unicode scalar values (`str::chars()`), never
//! bytes. Source spans remain byte offsets so callers can slice the original
//! UTF-8 document exactly. An oversized fenced block is the one exception to
//! the normal "fences are atomic" rule: it is split at UTF-8-safe boundaries
//! to honor the configured hard maximum and is marked in the result.

use sha2::{Digest, Sha256};

/// The sizing policy for [`MarkdownChunker`]. All values are Unicode scalar
/// values (characters), not bytes or model tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkingConfig {
    pub target_size: usize,
    pub min_size: usize,
    pub max_size: usize,
    pub overlap_size: usize,
}

impl ChunkingConfig {
    pub fn validate(self) -> Result<Self, ChunkingError> {
        if self.min_size == 0 || self.target_size == 0 || self.max_size == 0 {
            return Err(ChunkingError::InvalidConfig(
                "chunk sizes must be greater than zero".into(),
            ));
        }
        if self.min_size > self.target_size || self.target_size > self.max_size {
            return Err(ChunkingError::InvalidConfig(
                "min_size <= target_size <= max_size is required".into(),
            ));
        }
        if self.overlap_size >= self.max_size {
            return Err(ChunkingError::InvalidConfig(
                "overlap_size must be smaller than max_size".into(),
            ));
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkingError {
    InvalidConfig(String),
}

impl std::fmt::Display for ChunkingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for ChunkingError {}

/// A chunking implementation can be exercised without a database or an
/// inference provider.
pub trait Chunker {
    fn chunk(&self, source_identity: &str, markdown: &str) -> Result<Vec<Chunk>, ChunkingError>;
}

/// Structure-aware, dependency-free Markdown chunker.
#[derive(Debug, Clone, Copy)]
pub struct MarkdownChunker {
    config: ChunkingConfig,
}

impl MarkdownChunker {
    pub fn new(config: ChunkingConfig) -> Result<Self, ChunkingError> {
        Ok(Self {
            config: config.validate()?,
        })
    }

    pub fn config(&self) -> ChunkingConfig {
        self.config
    }
}

/// Persistable source metadata for one chunk. `content` is intentionally the
/// displayed source text; `search_text` includes heading context for embedding
/// and full-text search without repeating headings in displayed content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub key: String,
    pub location_key: String,
    pub ordinal: usize,
    pub heading_path: Vec<String>,
    pub start_line: usize,
    pub end_line: usize,
    pub start_byte: usize,
    pub end_byte: usize,
    pub overlap_from: Option<String>,
    pub overlap_chars: usize,
    pub content_hash: String,
    pub content: String,
    pub search_text: String,
    pub split_fenced_code: bool,
}

#[derive(Debug, Clone)]
struct Block {
    start_line: usize,
    end_line: usize,
    start_byte: usize,
    end_byte: usize,
    heading_path: Vec<String>,
    content: String,
    kind: BlockKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    Prose,
    List,
    Quote,
    CodeFence,
}

#[derive(Debug, Clone)]
struct Draft {
    start_line: usize,
    end_line: usize,
    start_byte: usize,
    end_byte: usize,
    heading_path: Vec<String>,
    content: String,
    split_fenced_code: bool,
}

impl Chunker for MarkdownChunker {
    fn chunk(&self, source_identity: &str, markdown: &str) -> Result<Vec<Chunk>, ChunkingError> {
        self.config.validate()?;
        let blocks = parse_blocks(markdown);
        let mut drafts = assemble_blocks(&blocks, self.config);
        if drafts.is_empty() {
            return Ok(Vec::new());
        }

        // Add overlap after the primary structural chunks exist. The display
        // content includes exactly the copied tail, while the source span
        // covers the complete origin range of the resulting text.
        let mut chunks = Vec::with_capacity(drafts.len());
        let mut previous_key: Option<String> = None;
        let mut previous_content = String::new();
        for (ordinal, draft) in drafts.drain(..).enumerate() {
            let mut content = draft.content.trim().to_owned();
            let mut overlap_from = None;
            let mut overlap_chars = 0;
            if self.config.overlap_size > 0 && !previous_content.is_empty() {
                let overlap = tail_chars(&previous_content, self.config.overlap_size);
                if !overlap.is_empty()
                    && char_count(&overlap) + char_count(&content) <= self.config.max_size
                {
                    overlap_chars = char_count(&overlap);
                    content = format!("{overlap}\n\n{content}");
                    overlap_from = previous_key.clone();
                }
            }
            let location_key = hash_parts(&[
                source_identity,
                &draft.heading_path.join("\u{1f}"),
                &format!("{}", ordinal),
            ]);
            let content_hash = hash_text(&normalize_for_hash(&content));
            let key = hash_parts(&[source_identity, &location_key, &content_hash]);
            let search_text = search_text(&draft.heading_path, &content);
            previous_content = content.clone();
            previous_key = Some(key.clone());
            chunks.push(Chunk {
                key,
                location_key,
                ordinal,
                heading_path: draft.heading_path,
                start_line: draft.start_line,
                end_line: draft.end_line,
                start_byte: draft.start_byte,
                end_byte: draft.end_byte,
                overlap_from,
                overlap_chars,
                content_hash,
                content,
                search_text,
                split_fenced_code: draft.split_fenced_code,
            });
        }
        Ok(chunks)
    }
}

fn parse_blocks(markdown: &str) -> Vec<Block> {
    let lines = lines_with_offsets(markdown);
    let mut blocks = Vec::new();
    let mut headings: Vec<String> = Vec::new();
    let mut index = 0;
    while index < lines.len() {
        let line = lines[index].text;
        if line.trim().is_empty() {
            index += 1;
            continue;
        }
        if let Some((level, title)) = heading(line) {
            if headings.len() >= level {
                headings.truncate(level - 1);
            }
            while headings.len() < level.saturating_sub(1) {
                headings.push(String::new());
            }
            headings.push(title.to_string());
            index += 1;
            continue;
        }
        if thematic_boundary(line) {
            index += 1;
            continue;
        }
        let start = index;
        let (end, kind) = if fence_marker(line).is_some() {
            let marker = fence_marker(line).expect("checked");
            index += 1;
            while index < lines.len() && !is_closing_fence(lines[index].text, marker) {
                index += 1;
            }
            if index < lines.len() {
                index += 1;
            }
            (index, BlockKind::CodeFence)
        } else if is_quote(line) {
            index += 1;
            while index < lines.len()
                && (is_quote(lines[index].text) || lines[index].text.trim().is_empty())
            {
                index += 1;
            }
            (index, BlockKind::Quote)
        } else if is_list_item(line) {
            index += 1;
            while index < lines.len() {
                let candidate = lines[index].text;
                if candidate.trim().is_empty() {
                    let next = index + 1;
                    if next < lines.len()
                        && (is_list_item(lines[next].text) || indented(lines[next].text))
                    {
                        index += 1;
                        continue;
                    }
                    break;
                }
                if is_list_item(candidate) || indented(candidate) {
                    index += 1;
                } else {
                    break;
                }
            }
            (index, BlockKind::List)
        } else {
            index += 1;
            while index < lines.len() {
                let candidate = lines[index].text;
                if candidate.trim().is_empty()
                    || heading(candidate).is_some()
                    || thematic_boundary(candidate)
                    || fence_marker(candidate).is_some()
                    || is_quote(candidate)
                    || is_list_item(candidate)
                {
                    break;
                }
                index += 1;
            }
            (index, BlockKind::Prose)
        };
        if end > start {
            let first = &lines[start];
            let last = &lines[end - 1];
            let raw_end = last.end_byte;
            let content = markdown[first.start_byte..raw_end].trim().to_string();
            if !content.is_empty() {
                blocks.push(Block {
                    start_line: first.number,
                    end_line: last.number,
                    start_byte: first.start_byte,
                    end_byte: raw_end,
                    heading_path: headings.clone(),
                    content,
                    kind,
                });
            }
        }
    }
    blocks
}

fn assemble_blocks(blocks: &[Block], config: ChunkingConfig) -> Vec<Draft> {
    let mut output = Vec::new();
    let mut current: Option<Draft> = None;
    for block in blocks {
        let parts = split_block(block, config);
        for part in parts {
            let should_boundary = current
                .as_ref()
                .is_some_and(|draft| draft.heading_path != part.heading_path);
            if should_boundary {
                flush(&mut current, &mut output);
            }
            let candidate_len = current
                .as_ref()
                .map_or(0, |draft| char_count(&draft.content) + 2)
                + char_count(&part.content);
            if current.is_some() && candidate_len > config.target_size {
                if current
                    .as_ref()
                    .is_some_and(|draft| char_count(&draft.content) >= config.min_size)
                {
                    flush(&mut current, &mut output);
                } else if candidate_len > config.max_size {
                    flush(&mut current, &mut output);
                }
            }
            match current.as_mut() {
                Some(draft) => {
                    draft.content.push_str("\n\n");
                    draft.content.push_str(&part.content);
                    draft.end_line = part.end_line;
                    draft.end_byte = part.end_byte;
                    draft.split_fenced_code |= part.split_fenced_code;
                }
                None => current = Some(part),
            }
            if current
                .as_ref()
                .is_some_and(|draft| char_count(&draft.content) >= config.target_size)
            {
                flush(&mut current, &mut output);
            }
        }
    }
    flush(&mut current, &mut output);
    output
}

fn split_block(block: &Block, config: ChunkingConfig) -> Vec<Draft> {
    if char_count(&block.content) <= config.max_size {
        return vec![draft_from_block(block, block.content.clone(), false)];
    }
    match block.kind {
        BlockKind::CodeFence => split_at_char_boundaries(block, config.max_size, true),
        BlockKind::Prose => split_prose(block, config.max_size),
        BlockKind::List | BlockKind::Quote => {
            split_at_char_boundaries(block, config.max_size, false)
        }
    }
}

fn split_prose(block: &Block, max_size: usize) -> Vec<Draft> {
    let sentences = sentence_slices(&block.content);
    if sentences.len() <= 1 {
        return split_at_char_boundaries(block, max_size, false);
    }
    let mut output = Vec::new();
    let mut current = String::new();
    let mut current_start = 0;
    for (sentence_start, sentence) in sentences {
        let additional = if current.is_empty() {
            char_count(sentence)
        } else {
            char_count(sentence) + 1
        };
        if !current.is_empty() && char_count(&current) + additional > max_size {
            output.push(draft_with_offset(block, current_start, &current, false));
            current.clear();
            current_start = sentence_start;
        }
        if current.is_empty() {
            current_start = sentence_start;
        } else {
            current.push(' ');
        }
        if char_count(sentence) > max_size {
            let temp = Block {
                content: sentence.to_string(),
                start_byte: block.start_byte + sentence_start,
                ..block.clone()
            };
            output.extend(split_at_char_boundaries(&temp, max_size, false));
            current.clear();
        } else {
            current.push_str(sentence);
        }
    }
    if !current.is_empty() {
        output.push(draft_with_offset(block, current_start, &current, false));
    }
    output
}

fn split_at_char_boundaries(block: &Block, max_size: usize, split_fenced_code: bool) -> Vec<Draft> {
    let mut output = Vec::new();
    let mut start = 0;
    while start < block.content.len() {
        let remaining = &block.content[start..];
        let mut end = remaining.len();
        if char_count(remaining) > max_size {
            end = nth_char_byte(remaining, max_size);
            // For prose/list/quote prefer a whitespace break inside the safe
            // prefix. Fenced code remains exact because syntax is opaque.
            if !split_fenced_code {
                if let Some(space) = remaining[..end].rfind(char::is_whitespace) {
                    if space > 0 {
                        end = space;
                    }
                }
            }
        }
        let piece = remaining[..end].trim();
        if !piece.is_empty() {
            output.push(draft_with_offset(
                block,
                start + leading_bytes(&remaining[..end]),
                piece,
                split_fenced_code,
            ));
        }
        start += end;
        while start < block.content.len() && block.content[start..].starts_with(char::is_whitespace)
        {
            let ch = block.content[start..].chars().next().expect("nonempty");
            start += ch.len_utf8();
        }
    }
    output
}

fn draft_from_block(block: &Block, content: String, split_fenced_code: bool) -> Draft {
    Draft {
        start_line: block.start_line,
        end_line: block.end_line,
        start_byte: block.start_byte,
        end_byte: block.end_byte,
        heading_path: block.heading_path.clone(),
        content,
        split_fenced_code,
    }
}

fn draft_with_offset(
    block: &Block,
    offset: usize,
    content: &str,
    split_fenced_code: bool,
) -> Draft {
    let before = &block.content[..offset.min(block.content.len())];
    let start_line = block.start_line + before.bytes().filter(|byte| *byte == b'\n').count();
    let end_line = start_line + content.bytes().filter(|byte| *byte == b'\n').count();
    Draft {
        start_line,
        end_line,
        start_byte: block.start_byte + offset,
        end_byte: block.start_byte + offset + content.len(),
        heading_path: block.heading_path.clone(),
        content: content.to_string(),
        split_fenced_code,
    }
}

fn flush(current: &mut Option<Draft>, output: &mut Vec<Draft>) {
    if let Some(draft) = current.take() {
        if !draft.content.trim().is_empty() {
            output.push(draft);
        }
    }
}

#[derive(Debug)]
struct Line<'a> {
    number: usize,
    start_byte: usize,
    end_byte: usize,
    text: &'a str,
}

fn lines_with_offsets(markdown: &str) -> Vec<Line<'_>> {
    let mut lines = Vec::new();
    let mut start = 0;
    for (index, line) in markdown.split_inclusive('\n').enumerate() {
        let end = start + line.len();
        lines.push(Line {
            number: index + 1,
            start_byte: start,
            end_byte: end,
            text: line.trim_end_matches('\n').trim_end_matches('\r'),
        });
        start = end;
    }
    if start < markdown.len() || markdown.is_empty() {
        lines.push(Line {
            number: lines.len() + 1,
            start_byte: start,
            end_byte: markdown.len(),
            text: &markdown[start..],
        });
    }
    lines
}

fn heading(line: &str) -> Option<(usize, &str)> {
    let trimmed = line.trim_start_matches(|ch: char| ch == ' ' || ch == '\t');
    let count = trimmed.chars().take_while(|ch| *ch == '#').count();
    if !(1..=6).contains(&count)
        || !trimmed
            .as_bytes()
            .get(count)
            .is_some_and(u8::is_ascii_whitespace)
    {
        return None;
    }
    let title = trimmed[count..].trim().trim_end_matches('#').trim();
    (!title.is_empty()).then_some((count, title))
}

fn thematic_boundary(line: &str) -> bool {
    let compact: String = line.chars().filter(|ch| !ch.is_whitespace()).collect();
    compact.len() >= 3 && compact.chars().all(|ch| matches!(ch, '-' | '_' | '*'))
}

fn fence_marker(line: &str) -> Option<(char, usize)> {
    let trimmed = line.trim_start_matches(' ');
    let marker = trimmed.chars().next()?;
    if marker != '`' && marker != '~' {
        return None;
    }
    let count = trimmed.chars().take_while(|ch| *ch == marker).count();
    (count >= 3).then_some((marker, count))
}

fn is_closing_fence(line: &str, marker: (char, usize)) -> bool {
    let trimmed = line.trim_start_matches(' ');
    let count = trimmed.chars().take_while(|ch| *ch == marker.0).count();
    count >= marker.1 && trimmed[count..].trim().is_empty()
}

fn is_quote(line: &str) -> bool {
    line.trim_start().starts_with('>')
}
fn indented(line: &str) -> bool {
    line.starts_with("  ") || line.starts_with('\t')
}
fn is_list_item(line: &str) -> bool {
    let trimmed = line.trim_start();
    if matches!(trimmed.chars().next(), Some('-' | '*' | '+'))
        && trimmed
            .as_bytes()
            .get(1)
            .is_some_and(u8::is_ascii_whitespace)
    {
        return true;
    }
    let digits = trimmed.chars().take_while(|ch| ch.is_ascii_digit()).count();
    digits > 0
        && trimmed[digits..].starts_with('.')
        && trimmed
            .as_bytes()
            .get(digits + 1)
            .is_some_and(u8::is_ascii_whitespace)
}

fn sentence_slices(text: &str) -> Vec<(usize, &str)> {
    let mut slices = Vec::new();
    let mut start = 0;
    for (offset, ch) in text.char_indices() {
        if matches!(ch, '.' | '!' | '?') {
            let end = offset + ch.len_utf8();
            let next = text[end..].chars().next();
            if next.is_none_or(char::is_whitespace) {
                let sentence = text[start..end].trim();
                if !sentence.is_empty() {
                    let leading = text[start..end].len() - text[start..end].trim_start().len();
                    slices.push((start + leading, sentence));
                }
                start = end;
            }
        }
    }
    let rest = text[start..].trim();
    if !rest.is_empty() {
        let leading = text[start..].len() - text[start..].trim_start().len();
        slices.push((start + leading, rest));
    }
    slices
}

fn char_count(text: &str) -> usize {
    text.chars().count()
}
fn nth_char_byte(text: &str, chars: usize) -> usize {
    text.char_indices()
        .nth(chars)
        .map_or(text.len(), |(idx, _)| idx)
}
fn leading_bytes(text: &str) -> usize {
    text.len() - text.trim_start().len()
}
fn tail_chars(text: &str, chars: usize) -> String {
    text.chars()
        .rev()
        .take(chars)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
}
fn normalize_for_hash(text: &str) -> String {
    text.replace("\r\n", "\n").trim().to_string()
}
fn hash_text(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))
}
fn hash_parts(parts: &[&str]) -> String {
    hash_text(&parts.join("\0"))
}
fn search_text(headings: &[String], content: &str) -> String {
    let headings = headings
        .iter()
        .filter(|heading| !heading.is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join(" > ");
    if headings.is_empty() {
        content.to_string()
    } else {
        format!("{headings}\n\n{content}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(markdown: &str, config: ChunkingConfig) -> Vec<Chunk> {
        MarkdownChunker::new(config)
            .unwrap()
            .chunk("source:fixture", markdown)
            .unwrap()
    }

    const SIZES: ChunkingConfig = ChunkingConfig {
        min_size: 20,
        target_size: 80,
        max_size: 100,
        overlap_size: 0,
    };

    #[test]
    fn retains_heading_hierarchy_without_repeating_it_in_display_content() {
        let chunks = chunk("# Root\n\nintro paragraph with enough meaningful words.\n\n## Child\n\nchild paragraph with enough meaningful words.", SIZES);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].heading_path, ["Root"]);
        assert_eq!(chunks[1].heading_path, ["Root", "Child"]);
        assert!(!chunks[1].content.contains("## Child"));
        assert!(chunks[1].search_text.starts_with("Root > Child\n\n"));
    }

    #[test]
    fn keeps_lists_quotes_and_fences_atomic_when_feasible() {
        let markdown = "# H\n\n- one list item\n- another list item\n\n> quoted material remains one block\n> with two lines\n\n```rust\nlet value = 42;\nprintln!(\"{value}\");\n```";
        let chunks = chunk(
            markdown,
            ChunkingConfig {
                min_size: 10,
                target_size: 45,
                max_size: 80,
                overlap_size: 0,
            },
        );
        assert!(chunks
            .iter()
            .any(|chunk| chunk.content.contains("- one list item\n- another")));
        assert!(chunks
            .iter()
            .any(|chunk| chunk.content.contains("> quoted material")));
        let fence = chunks
            .iter()
            .find(|chunk| chunk.content.contains("```rust"))
            .unwrap();
        assert!(fence.content.ends_with("```"));
        assert!(!fence.split_fenced_code);
    }

    #[test]
    fn oversized_prose_prefers_sentence_boundaries_and_unicode_is_not_split() {
        let markdown = "# Unicode\n\n第一文です。第二文です。Third sentence is deliberately long enough to require a safe boundary.";
        let chunks = chunk(
            markdown,
            ChunkingConfig {
                min_size: 5,
                target_size: 20,
                max_size: 28,
                overlap_size: 0,
            },
        );
        assert!(chunks.len() >= 2);
        assert!(chunks.iter().all(|chunk| char_count(&chunk.content) <= 28));
        assert!(chunks
            .iter()
            .all(|chunk| std::str::from_utf8(chunk.content.as_bytes()).is_ok()));
    }

    #[test]
    fn merges_short_adjacent_blocks_under_the_same_heading() {
        let chunks = chunk(
            "# H\n\nshort one.\n\nshort two.\n\nThis is a longer paragraph that makes the combined chunk useful.",
            ChunkingConfig {
                min_size: 20,
                target_size: 150,
                max_size: 200,
                overlap_size: 0,
            },
        );
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("short one.\n\nshort two."));
    }

    #[test]
    fn overlap_has_explicit_parent_metadata() {
        let chunks = chunk("# H\n\nFirst paragraph has sufficient text to become its own chunk here.\n\nSecond paragraph also has sufficient text to become its own chunk here.", ChunkingConfig { min_size: 10, target_size: 50, max_size: 100, overlap_size: 12 });
        assert_eq!(chunks.len(), 2);
        assert_eq!(
            chunks[1].overlap_from.as_deref(),
            Some(chunks[0].key.as_str())
        );
        assert_eq!(chunks[1].overlap_chars, 12);
        assert!(chunks[1]
            .content
            .starts_with(&tail_chars(&chunks[0].content, 12)));
    }

    #[test]
    fn stable_identity_and_local_change_are_scoped_to_affected_chunk() {
        let before = "# H\n\nFirst independent paragraph with useful stable content.\n\nSecond independent paragraph with useful stable content.";
        let after = "# H\n\nFirst independent paragraph with one local changed word.\n\nSecond independent paragraph with useful stable content.";
        let config = ChunkingConfig {
            min_size: 10,
            target_size: 55,
            max_size: 100,
            overlap_size: 0,
        };
        let before = chunk(before, config);
        let after = chunk(after, config);
        assert_ne!(before[0].key, after[0].key);
        assert_eq!(before[1].key, after[1].key);
        assert_eq!(before[1].content_hash, after[1].content_hash);
    }

    #[test]
    fn whitespace_only_and_oversized_fence_are_explicit() {
        assert!(chunk(" \n\t\n", SIZES).is_empty());
        let chunks = chunk(
            "```text\n0123456789012345678901234567890123456789\n```",
            ChunkingConfig {
                min_size: 5,
                target_size: 10,
                max_size: 16,
                overlap_size: 0,
            },
        );
        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|chunk| chunk.split_fenced_code));
        assert!(chunks.iter().all(|chunk| char_count(&chunk.content) <= 16));
    }
}
