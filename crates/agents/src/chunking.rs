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
    /// Exact source bytes between the preceding content block and this block.
    /// It is empty for the first block and for fragments split from one block.
    separator_before: String,
    /// A thematic break ended the preceding assembly region. The delimiter is
    /// structural rather than displayed content, but chunks may not span it.
    assembly_boundary_before: bool,
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
    separator_before: String,
    assembly_boundary_before: bool,
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
            let mut content = draft.content;
            let mut overlap_from = None;
            let mut overlap_chars = 0;
            if self.config.overlap_size > 0 && !previous_content.is_empty() {
                let overlap = tail_chars(&previous_content, self.config.overlap_size);
                // The display representation includes the two newline
                // separator characters, which count toward the hard limit.
                const OVERLAP_SEPARATOR: &str = "\n\n";
                if !overlap.is_empty()
                    && char_count(&overlap) + char_count(OVERLAP_SEPARATOR) + char_count(&content)
                        <= self.config.max_size
                {
                    overlap_chars = char_count(&overlap);
                    content = format!("{overlap}{OVERLAP_SEPARATOR}{content}");
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
    let mut previous_end_byte = None;
    let mut assembly_boundary_before = false;
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
            assembly_boundary_before = true;
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
            let raw = &markdown[first.start_byte..raw_end];
            let (trim_start, trim_end) = trim_byte_bounds(raw);
            let content = raw[trim_start..trim_end].to_string();
            if !content.is_empty() {
                let start_byte = first.start_byte + trim_start;
                let end_byte = first.start_byte + trim_end;
                let start_line = first.number
                    + raw[..trim_start]
                        .bytes()
                        .filter(|byte| *byte == b'\n')
                        .count();
                let end_line = first.number
                    + raw[..trim_end]
                        .bytes()
                        .filter(|byte| *byte == b'\n')
                        .count();
                let separator_before = previous_end_byte
                    .map(|previous_end| markdown[previous_end..start_byte].to_string())
                    .unwrap_or_default();
                blocks.push(Block {
                    start_line,
                    end_line,
                    start_byte,
                    end_byte,
                    heading_path: headings.clone(),
                    separator_before,
                    assembly_boundary_before,
                    content,
                    kind,
                });
                assembly_boundary_before = false;
                previous_end_byte = Some(end_byte);
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
            let should_boundary = current.as_ref().is_some_and(|draft| {
                part.assembly_boundary_before || draft.heading_path != part.heading_path
            });
            if should_boundary {
                flush(&mut current, &mut output);
            }
            let candidate_len = current
                .as_ref()
                .map_or(0, |draft| char_count(&draft.content))
                + char_count(&part.separator_before)
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
                    draft.content.push_str(&part.separator_before);
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
    let mut output = Vec::new();
    let mut start = 0;
    while start < block.content.len() {
        let remaining = &block.content[start..];
        if char_count(remaining) <= max_size {
            output.push(draft_with_offset(block, start, remaining, false));
            break;
        }
        let limit = nth_char_byte(remaining, max_size);
        let end = preferred_prose_break(remaining, limit).unwrap_or(limit);
        output.push(draft_with_offset(block, start, &remaining[..end], false));
        start += end;
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
                end = whitespace_break(remaining, end).unwrap_or(end);
            }
        }
        let piece = &remaining[..end];
        output.push(draft_with_offset(block, start, piece, split_fenced_code));
        start += end;
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
        separator_before: block.separator_before.clone(),
        assembly_boundary_before: block.assembly_boundary_before,
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
        separator_before: if offset == 0 {
            block.separator_before.clone()
        } else {
            String::new()
        },
        assembly_boundary_before: offset == 0 && block.assembly_boundary_before,
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
    // CommonMark permits at most three leading spaces for an ATX heading.
    // Four spaces or a tab are indented code/content, not document structure.
    let leading_spaces = line.bytes().take_while(|byte| *byte == b' ').count();
    if leading_spaces > 3 {
        return None;
    }
    let trimmed = &line[leading_spaces..];
    let count = trimmed.chars().take_while(|ch| *ch == '#').count();
    if !(1..=6).contains(&count)
        || !trimmed
            .as_bytes()
            .get(count)
            .is_some_and(u8::is_ascii_whitespace)
    {
        return None;
    }
    let title = trimmed[count..].trim();
    // A closing ATX sequence is syntactic only when separated from the title
    // by whitespace. `# C#` therefore has the literal title `C#`, whereas
    // `# C #` has the title `C`.
    let without_closing = title
        .char_indices()
        .rev()
        .take_while(|(_, ch)| *ch == '#')
        .last()
        .and_then(|(start, _)| {
            title[..start]
                .chars()
                .next_back()
                .is_some_and(char::is_whitespace)
                .then_some(title[..start].trim_end())
        })
        .unwrap_or(title);
    Some((count, without_closing))
}

fn thematic_boundary(line: &str) -> bool {
    // A thematic break has at most three leading spaces, then three or more
    // instances of one marker. Tabs in the indentation position represent an
    // indented code block, and mixed markers are ordinary prose.
    let leading_spaces = line.bytes().take_while(|byte| *byte == b' ').count();
    if leading_spaces > 3 {
        return false;
    }
    let body = &line[leading_spaces..];
    if body.chars().next().is_some_and(char::is_whitespace) {
        return false;
    }

    let mut marker = None;
    let mut count = 0;
    for character in body.chars() {
        if character.is_whitespace() {
            continue;
        }
        if !matches!(character, '-' | '_' | '*') {
            return false;
        }
        match marker {
            Some(expected) if expected != character => return false,
            Some(_) => {}
            None => marker = Some(character),
        }
        count += 1;
    }
    count >= 3
}

fn fence_marker(line: &str) -> Option<(char, usize)> {
    // Fenced-code delimiters, like ATX headings, may be indented by at most
    // three spaces. Four spaces or a tab are literal indented-code content.
    let leading_spaces = line.bytes().take_while(|byte| *byte == b' ').count();
    if leading_spaces > 3 {
        return None;
    }
    let trimmed = &line[leading_spaces..];
    let marker = trimmed.chars().next()?;
    if marker != '`' && marker != '~' {
        return None;
    }
    let count = trimmed.chars().take_while(|ch| *ch == marker).count();
    if count < 3 {
        return None;
    }
    // CommonMark forbids backticks in the info string of a backtick fence.
    // Treat an invalid opener as ordinary content so it cannot hide later
    // document structure behind an unterminated code block.
    let info_start = nth_char_byte(trimmed, count);
    if marker == '`' && trimmed[info_start..].contains('`') {
        return None;
    }
    Some((marker, count))
}

fn is_closing_fence(line: &str, marker: (char, usize)) -> bool {
    let leading_spaces = line.bytes().take_while(|byte| *byte == b' ').count();
    if leading_spaces > 3 {
        return false;
    }
    let trimmed = &line[leading_spaces..];
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

fn trim_byte_bounds(text: &str) -> (usize, usize) {
    let start = text.len() - text.trim_start().len();
    let end = text.trim_end().len();
    (start, end)
}

/// Return a source-preserving sentence boundary inside `text[..limit]`. The
/// returned index is immediately after punctuation, leaving following source
/// whitespace at the beginning of the next fragment instead of normalizing it.
fn preferred_prose_break(text: &str, limit: usize) -> Option<usize> {
    let mut boundary = None;
    for (offset, ch) in text[..limit].char_indices() {
        if matches!(ch, '.' | '!' | '?') {
            let end = offset + ch.len_utf8();
            if text[end..].chars().next().is_none_or(char::is_whitespace) {
                boundary = Some(end);
            }
        }
    }
    boundary.or_else(|| whitespace_break(text, limit))
}

/// Prefer splitting after whitespace so every character remains in exactly one
/// fragment. Returning the byte after whitespace avoids the old trim-and-skip
/// behavior that silently dropped separators.
fn whitespace_break(text: &str, limit: usize) -> Option<usize> {
    text[..limit]
        .char_indices()
        .filter_map(|(offset, ch)| ch.is_whitespace().then_some(offset + ch.len_utf8()))
        .last()
        .filter(|offset| *offset > 0)
}

fn char_count(text: &str) -> usize {
    text.chars().count()
}
fn nth_char_byte(text: &str, chars: usize) -> usize {
    text.char_indices()
        .nth(chars)
        .map_or(text.len(), |(idx, _)| idx)
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
    fn indented_hashes_remain_content_not_headings() {
        let markdown = "    # Four-space code\n\n\t# Tab code\n\n# Actual Heading\n\nBody under the actual heading.";
        let chunks = chunk(
            markdown,
            ChunkingConfig {
                min_size: 1,
                target_size: 200,
                max_size: 240,
                overlap_size: 0,
            },
        );

        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].heading_path.is_empty());
        assert!(chunks[0].content.contains("# Four-space code"));
        assert!(chunks[0].content.contains("# Tab code"));
        assert_eq!(chunks[1].heading_path, ["Actual Heading"]);
    }

    #[test]
    fn atx_headings_allow_empty_titles_and_only_strip_spaced_closing_hashes() {
        assert_eq!(heading("# "), Some((1, "")));
        assert_eq!(heading("# C#"), Some((1, "C#")));
        assert_eq!(heading("# C ###"), Some((1, "C")));

        let chunks = chunk(
            "# \n\nContent belongs below an empty ATX heading.",
            ChunkingConfig {
                min_size: 1,
                target_size: 200,
                max_size: 240,
                overlap_size: 0,
            },
        );
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].heading_path, [""]);
        assert!(!chunks[0].content.contains("# "));
    }

    #[test]
    fn indented_fences_remain_literal_content_and_do_not_hide_headings() {
        let markdown = "    ```rust\nlet four_space = true;\n    ```\n\n\t~~~text\ntab fence content\n\t~~~\n\n# Actual Heading\n\nBody remains structurally visible.";
        let chunks = chunk(
            markdown,
            ChunkingConfig {
                min_size: 1,
                target_size: 240,
                max_size: 280,
                overlap_size: 0,
            },
        );

        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].heading_path.is_empty());
        assert!(chunks[0].content.contains("```rust"));
        assert!(chunks[0].content.contains("~~~text"));
        assert_eq!(chunks[1].heading_path, ["Actual Heading"]);
    }

    #[test]
    fn backtick_fence_info_strings_cannot_contain_backticks() {
        assert!(fence_marker("```rust").is_some());
        assert!(fence_marker("```rust`invalid").is_none());

        let chunks = chunk(
            "```rust`invalid\nthis remains prose\n\n# Actual Heading\n\nThe heading remains visible.",
            ChunkingConfig {
                min_size: 1,
                target_size: 240,
                max_size: 280,
                overlap_size: 0,
            },
        );
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].content.contains("```rust`invalid"));
        assert_eq!(chunks[1].heading_path, ["Actual Heading"]);
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
    fn oversized_prose_preserves_original_whitespace_across_fragments() {
        let markdown = "  Alpha sentence. \t Beta sentence.  Gamma sentence.   ";
        let chunks = chunk(
            markdown,
            ChunkingConfig {
                min_size: 1,
                target_size: 18,
                max_size: 20,
                overlap_size: 0,
            },
        );
        assert!(chunks.len() >= 2);
        let trimmed = markdown.trim();
        assert_eq!(
            chunks
                .iter()
                .map(|chunk| chunk.content.as_str())
                .collect::<String>(),
            trimmed
        );
        assert!(chunks
            .iter()
            .all(|chunk| { markdown[chunk.start_byte..chunk.end_byte] == chunk.content }));
    }

    #[test]
    fn source_byte_spans_reproduce_trimmed_chunk_content() {
        let markdown = "# Heading\n\n  First sentence is intentionally long.\nSecond sentence keeps its exact newline.  ";
        let chunks = chunk(
            markdown,
            ChunkingConfig {
                min_size: 1,
                target_size: 24,
                max_size: 28,
                overlap_size: 0,
            },
        );
        assert!(chunks.len() >= 2);
        for chunk in chunks {
            assert_eq!(
                &markdown[chunk.start_byte..chunk.end_byte],
                chunk.content,
                "chunk source span must reproduce displayed primary content"
            );
        }
    }

    #[test]
    fn merges_short_adjacent_blocks_under_the_same_heading() {
        let markdown = "# H\n\nshort one.\n\nshort two.\n\nThis is a longer paragraph that makes the combined chunk useful.";
        let chunks = chunk(
            markdown,
            ChunkingConfig {
                min_size: 20,
                target_size: 150,
                max_size: 200,
                overlap_size: 0,
            },
        );
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("short one.\n\nshort two."));
        assert_eq!(
            &markdown[chunks[0].start_byte..chunks[0].end_byte],
            chunks[0].content
        );
    }

    #[test]
    fn thematic_breaks_are_assembly_boundaries() {
        let markdown =
            "# H\n\nAlpha stays on the first side.\n\n---\n\nBeta stays on the second side.";
        let chunks = chunk(
            markdown,
            ChunkingConfig {
                min_size: 1,
                target_size: 200,
                max_size: 240,
                overlap_size: 0,
            },
        );

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].content, "Alpha stays on the first side.");
        assert_eq!(chunks[1].content, "Beta stays on the second side.");
        assert!(chunks.iter().all(|chunk| !chunk.content.contains("---")));
        for chunk in chunks {
            assert_eq!(
                &markdown[chunk.start_byte..chunk.end_byte],
                chunk.content,
                "thematic-break boundaries must retain exact source spans"
            );
        }
    }

    #[test]
    fn only_valid_unindented_thematic_breaks_are_structural() {
        assert!(thematic_boundary("---"));
        assert!(thematic_boundary("  * * *"));
        assert!(thematic_boundary("___"));
        assert!(!thematic_boundary("-_*"));
        assert!(!thematic_boundary("    ---"));
        assert!(!thematic_boundary("\t---"));

        let markdown = "Alpha prose stays together.\n\n-_*\n\nBeta prose stays together.\n\n    ---\n\nGamma prose stays together.";
        let chunks = chunk(
            markdown,
            ChunkingConfig {
                min_size: 1,
                target_size: 240,
                max_size: 280,
                overlap_size: 0,
            },
        );
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("-_*"));
        assert!(chunks[0].content.contains("    ---"));
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
    fn overlap_separator_is_included_in_the_hard_maximum() {
        let chunks = chunk(
            "abcdefghij\n\nklmnopqrst",
            ChunkingConfig {
                min_size: 1,
                target_size: 10,
                max_size: 20,
                overlap_size: 10,
            },
        );
        assert_eq!(chunks.len(), 2);
        assert!(chunks.iter().all(|chunk| char_count(&chunk.content) <= 20));
        // 10 overlap chars + 10 content chars would have fit before the
        // separator was counted, but `\n\n` would have made a 22-char chunk.
        assert!(chunks[1].overlap_from.is_none());
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
