//! Text and lineage heuristics for opt-in candidate quality gates (#230).
//!
//! Everything here is deterministic, local and bounded by the two notes'
//! own text: no model call, no extra read. Each heuristic only *removes*
//! pairs (near-duplicates, machine dumps, lineage templates) or adds one
//! explicit, explainable second signal (a textual cue).

use chrono::NaiveDate;
use std::collections::BTreeSet;

/// Lowercased alphanumeric word runs (Unicode letters and digits).
pub(crate) fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// A shared run shorter than this many words is a phrase, never a copy.
pub(crate) const MIN_SHARED_SPAN_WORDS: usize = 12;

/// Longest run of consecutive words two notes share, as a share of the
/// shorter note's word count; zero when the run is shorter than
/// [`MIN_SHARED_SPAN_WORDS`]. Overlapping chunks of one file, copies and
/// repeated templated output share long runs; independent notes about one
/// topic rarely do.
pub(crate) fn shared_span_share(left: &str, right: &str) -> f32 {
    let (left, right) = (words(left), words(right));
    let (short, long) = if left.len() <= right.len() {
        (left, right)
    } else {
        (right, left)
    };
    if short.is_empty() {
        return 0.0;
    }
    let mut best = 0usize;
    let mut previous = vec![0usize; long.len() + 1];
    let mut current = vec![0usize; long.len() + 1];
    for word in &short {
        for (index, other) in long.iter().enumerate() {
            current[index + 1] = if word == other {
                previous[index] + 1
            } else {
                0
            };
            best = best.max(current[index + 1]);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    if best < MIN_SHARED_SPAN_WORDS {
        return 0.0;
    }
    best as f32 / short.len() as f32
}

fn key_like(text: &str) -> bool {
    !text.is_empty()
        && text
            .chars()
            .all(|c| c.is_alphanumeric() || "_.$/'\"-".contains(c))
}

/// One line of structured or machine output rather than prose: JSON/YAML
/// structure, table rows, stack frames and addresses, or mostly symbols and
/// digits.
pub(crate) fn machine_line(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.starts_with(['{', '}', '[', ']']) {
        return true;
    }
    // "key": value
    if let Some(rest) = trimmed.strip_prefix('"') {
        if let Some((key, after)) = rest.split_once('"') {
            if !key.is_empty() && after.trim_start().starts_with(':') {
                return true;
            }
        }
    }
    // Indented YAML `key:` or `- key:`.
    let indent = line.len() - line.trim_start().len();
    if indent >= 2 {
        let item = trimmed
            .strip_prefix('-')
            .filter(|rest| rest.starts_with(char::is_whitespace))
            .map_or(trimmed, str::trim_start);
        if let Some((key, after)) = item.split_once(':') {
            if key_like(key) && (after.is_empty() || after.starts_with(char::is_whitespace)) {
                return true;
            }
        }
    }
    if trimmed.len() >= 2 && trimmed.starts_with('|') && trimmed.ends_with('|') {
        return true;
    }
    if stack_frame(trimmed) {
        return true;
    }
    let visible = trimmed.chars().filter(|c| !c.is_whitespace()).count();
    let letters = trimmed.chars().filter(|c| c.is_alphabetic()).count();
    visible > 0 && (letters as f32) < 0.5 * visible as f32
}

fn stack_frame(line: &str) -> bool {
    // `(in libsystem_kernel.dylib)`
    let library_frame = line.match_indices("(in ").any(|(start, _)| {
        let rest = &line[start + 4..];
        rest.split_once(')').is_some_and(|(library, _)| {
            !library.is_empty()
                && library
                    .chars()
                    .all(|c| c.is_alphanumeric() || "_.+-".contains(c))
        })
    });
    // A hexadecimal address of six or more digits.
    let address = line.match_indices("0x").any(|(start, _)| {
        line[start + 2..]
            .bytes()
            .take_while(u8::is_ascii_hexdigit)
            .count()
            >= 6
    });
    // `at module.function(`
    let call_frame = line.strip_prefix("at ").is_some_and(|rest| {
        rest.split_once('(').is_some_and(|(name, _)| {
            !name.is_empty()
                && name
                    .chars()
                    .all(|c| c.is_alphanumeric() || "_.$<>".contains(c))
        })
    });
    // `File "worker.py", line 42`
    let python_frame = line.match_indices("File \"").any(|(start, _)| {
        line[start + 6..]
            .split_once('"')
            .is_some_and(|(file, after)| {
                !file.is_empty()
                    && after
                        .strip_prefix(", line ")
                        .is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_digit()))
            })
    });
    library_frame || address || call_frame || python_frame
}

/// Share of a note's non-blank lines that are machine output.
pub(crate) fn machine_line_share(text: &str) -> f32 {
    let lines = text
        .split('\n')
        .filter(|line| !line.trim().is_empty())
        .collect::<Vec<_>>();
    if lines.is_empty() {
        return 0.0;
    }
    lines.iter().filter(|line| machine_line(line)).count() as f32 / lines.len() as f32
}

/// Recurring-document family of an origin URI: digit runs collapse, so the
/// daily logs `memory/2026-07-01.md` and `memory/2026-07-02.md` (or the
/// heartbeat runs `cron/run-0041.md` and `cron/run-0042.md`) are one family.
pub(crate) fn lineage_family(origin: &str) -> String {
    let mut family = String::with_capacity(origin.len());
    let mut in_digits = false;
    for c in origin.chars() {
        if c.is_ascii_digit() {
            if !in_digits {
                family.push('#');
            }
            in_digits = true;
        } else {
            family.push(c);
            in_digits = false;
        }
    }
    family
}

/// The ISO date in an origin document's file name, if any.
pub(crate) fn document_date(origin: &str) -> Option<NaiveDate> {
    let name = origin.rsplit(['/', '\\']).next()?;
    let bytes = name.as_bytes();
    (0..bytes.len().saturating_sub(9)).find_map(|start| {
        let candidate = name.get(start..start + 10)?;
        let shape = candidate.bytes().enumerate().all(|(index, byte)| {
            if index == 4 || index == 7 {
                byte == b'-'
            } else {
                byte.is_ascii_digit()
            }
        });
        let bounded = start == 0 || !bytes[start - 1].is_ascii_digit();
        let closed = bytes
            .get(start + 10)
            .is_none_or(|byte| !byte.is_ascii_digit());
        if shape && bounded && closed {
            NaiveDate::parse_from_str(candidate, "%Y-%m-%d").ok()
        } else {
            None
        }
    })
}

/// An explicit textual pointer from one note to the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TextualCueKind {
    /// "yesterday's …" in a note whose document is dated the day after the
    /// other note's document.
    RelativeDate,
    /// "compared with 2026-04-28", "as noted in 2026-04-28 …": a
    /// back-reference phrase naming the other document's date.
    DatedReference,
    /// An open question in the earlier note answered by a statement in a
    /// note of the same day or later.
    AnsweredQuestion,
}

impl TextualCueKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RelativeDate => "relative_date",
            Self::DatedReference => "dated_reference",
            Self::AnsweredQuestion => "answered_question",
        }
    }
}

fn whole_word(haystack: &str, needle: &str) -> bool {
    haystack.match_indices(needle).any(|(start, _)| {
        let before = haystack[..start].chars().next_back();
        let after = haystack[start + needle.len()..].chars().next();
        before.is_none_or(|c| !c.is_alphanumeric() && c != '_')
            && after.is_none_or(|c| !c.is_alphanumeric() && c != '_')
    })
}

const BACK_REFERENCES: &[&str] = &[
    "compared with",
    "compared to",
    "as noted in",
    "as noted on",
    "as recorded in",
    "as recorded on",
    "as logged in",
    "as logged on",
    "as reported in",
    "as reported on",
    "carried over from",
    "follow-up to",
    "follow-up on",
    "follow-up from",
    "following up on",
];

/// `text` (dated `own`) points back at a document dated `other`.
fn back_reference(text: &str, own: NaiveDate, other: NaiveDate) -> Option<TextualCueKind> {
    if other >= own {
        return None;
    }
    let lower = text.to_lowercase();
    if other.succ_opt() == Some(own) && whole_word(&lower, "yesterday") {
        return Some(TextualCueKind::RelativeDate);
    }
    let date = other.format("%Y-%m-%d").to_string();
    lower.match_indices(&date).find_map(|(start, _)| {
        let window_start = lower.floor_char_boundary(start.saturating_sub(40));
        let before = lower[window_start..start].trim_end();
        let before = before
            .strip_suffix("the")
            .filter(|rest| rest.ends_with(char::is_whitespace))
            .map_or(before, str::trim_end);
        let whitespace_separated = lower[..start].ends_with(char::is_whitespace);
        (whitespace_separated
            && BACK_REFERENCES.iter().any(|phrase| {
                before.ends_with(phrase)
                    && before[..before.len() - phrase.len()]
                        .chars()
                        .next_back()
                        .is_none_or(|c| !c.is_alphanumeric())
            }))
        .then_some(TextualCueKind::DatedReference)
    })
}

const STOP_WORDS: &[&str] = &[
    "what", "which", "when", "where", "who", "whom", "whose", "why", "how", "should", "would",
    "could", "can", "will", "does", "did", "is", "are", "was", "were", "the", "and", "for", "with",
    "that", "this", "from", "into", "onto", "your", "you", "have", "has", "had", "any", "each",
    "also", "then", "than", "them", "they", "their", "there", "here", "about", "after", "before",
    "need", "needs", "want", "me", "my", "our", "we", "us", "it", "its", "to", "of", "on", "in",
    "at", "by", "or", "an", "be", "do", "not", "yes", "no",
];

/// Content words of a sentence: lowercased tokens of three or more
/// characters (letters, digits, `_`, `-`) minus stop words, with a plural
/// `s` dropped.
fn content_words(text: &str) -> BTreeSet<String> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '-'))
        .map(|token| token.trim_start_matches(['_', '-']).to_lowercase())
        .filter(|token| token.chars().count() >= 3 && !STOP_WORDS.contains(&token.as_str()))
        .map(|token| {
            if token.chars().count() > 3 && token.ends_with('s') && !token.ends_with("ss") {
                token[..token.len() - 1].to_owned()
            } else {
                token
            }
        })
        .collect()
}

/// Questions: a run of at least eight characters without a sentence end or
/// line break, closed by `?`.
fn questions(text: &str) -> Vec<&str> {
    let mut found = Vec::new();
    let mut start = 0;
    for (index, c) in text.char_indices() {
        match c {
            '?' => {
                let question = &text[start..index];
                if question.chars().count() >= 8 {
                    found.push(question);
                }
                start = index + c.len_utf8();
            }
            '.' | '!' | '\n' => start = index + c.len_utf8(),
            _ => {}
        }
    }
    found
}

/// Statements (sentences without a question mark) of a note.
fn statements(text: &str) -> Vec<&str> {
    let mut found = Vec::new();
    for line in text.split('\n') {
        let mut start = 0;
        let chars = line.char_indices().collect::<Vec<_>>();
        for (position, (index, c)) in chars.iter().enumerate() {
            let ends = matches!(c, '.' | '!')
                && chars
                    .get(position + 1)
                    .is_some_and(|(_, next)| next.is_whitespace());
            if ends {
                found.push(&line[start..index + c.len_utf8()]);
                start = index + c.len_utf8();
            }
        }
        found.push(&line[start..]);
    }
    found.retain(|sentence| !sentence.contains('?'));
    found
}

/// A question in `asked` (dated `asked_on`) answered by one statement in
/// `answer` (dated `answered_on`, the same day or later): the statement
/// carries at least three, and at least three quarters, of the question's
/// content words.
fn answered_question(
    asked: &str,
    asked_on: NaiveDate,
    answer: &str,
    answered_on: NaiveDate,
) -> bool {
    if answered_on < asked_on {
        return false;
    }
    let answers = statements(answer)
        .into_iter()
        .map(content_words)
        .collect::<Vec<_>>();
    questions(asked).into_iter().any(|question| {
        let wanted = content_words(question);
        if wanted.len() < 3 {
            return false;
        }
        let needed = (3 * wanted.len()).div_ceil(4).max(3);
        answers
            .iter()
            .any(|statement| wanted.intersection(statement).count() >= needed)
    })
}

/// Which note carries the cue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CueDirection {
    /// The `from` note points at, or answers, the `to` note.
    FromMentionsTo,
    ToMentionsFrom,
}

/// The first explicit textual cue between two dated notes, strongest kind
/// first. Notes whose documents carry no date never cue.
pub(crate) fn textual_cue(
    from_text: &str,
    from_origin: Option<&str>,
    to_text: &str,
    to_origin: Option<&str>,
) -> Option<(TextualCueKind, CueDirection)> {
    let from_date = from_origin.and_then(document_date)?;
    let to_date = to_origin.and_then(document_date)?;
    if let Some(kind) = back_reference(from_text, from_date, to_date) {
        return Some((kind, CueDirection::FromMentionsTo));
    }
    if let Some(kind) = back_reference(to_text, to_date, from_date) {
        return Some((kind, CueDirection::ToMentionsFrom));
    }
    // The answering note is the one that carries the answer.
    if answered_question(from_text, from_date, to_text, to_date) {
        return Some((
            TextualCueKind::AnsweredQuestion,
            CueDirection::ToMentionsFrom,
        ));
    }
    if answered_question(to_text, to_date, from_text, from_date) {
        return Some((
            TextualCueKind::AnsweredQuestion,
            CueDirection::FromMentionsTo,
        ));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_spans_measure_copied_runs_not_topic_overlap() {
        let original = "Deploy notes: the relay restarted at noon and the queue drained in four minutes after the cache warmed";
        let overlapping_chunk = "the relay restarted at noon and the queue drained in four minutes after the cache warmed; next the audit";
        assert!(shared_span_share(original, overlapping_chunk) > 0.8);
        let same_topic = "Relay check: queue normal, cache warm, nothing restarted today and no drain was needed at all";
        assert!(shared_span_share(original, same_topic) < 0.2);
        assert_eq!(shared_span_share("", original), 0.0);
        // A short shared phrase is never a copy, even in a short note.
        assert_eq!(
            shared_span_share(
                "Gateway status: warning shown.",
                "Gateway status: warning gone."
            ),
            0.0
        );
    }

    #[test]
    fn machine_lines_cover_json_yaml_tables_and_stack_frames() {
        for line in [
            "{\"queue\": 3}",
            "  \"depth\": 4,",
            "    responses:",
            "      - name: ready",
            "| route | status |",
            "    + 1 start_wqthread  (in libsystem_pthread.dylib) + 8  [0x1899dcc10]",
            "at app.main(Main.java:10)",
            "  File \"worker.py\", line 42, in run",
            "12.4 | 33.1 | 0.002",
        ] {
            assert!(machine_line(line), "{line}");
        }
        for line in [
            "The relay restarted at noon: nothing else changed.",
            "Decision: keep the nightly window.",
            "- Gateway status: running, probe OK.",
            "Why did the queue drain so slowly (in my view)?",
        ] {
            assert!(!machine_line(line), "{line}");
        }
        let dump = "{\n  \"a\": 1,\n  \"b\": [1, 2],\n  \"c\": {\"d\": 0.5}\n}";
        assert!(machine_line_share(dump) >= 0.99);
        let prose = "Morning check.\nThe relay is healthy.\n  key: value";
        assert!(machine_line_share(prose) < 0.5);
        assert_eq!(machine_line_share("\n \n"), 0.0);
    }

    #[test]
    fn lineage_families_collapse_dated_and_numbered_documents() {
        assert_eq!(
            lineage_family("vault://ops/memory/2026-07-01.md"),
            lineage_family("vault://ops/memory/2026-07-02.md")
        );
        assert_eq!(
            lineage_family("vault://ops/cron/run-0041.md"),
            "vault://ops/cron/run-#.md"
        );
        assert_ne!(
            lineage_family("vault://ops/memory/2026-07-01.md"),
            lineage_family("vault://ops/memory/2026-07-01-release.md")
        );
        assert_eq!(
            document_date("vault://ops/memory/2026-07-01-release.md"),
            NaiveDate::from_ymd_opt(2026, 7, 1)
        );
        assert_eq!(document_date("vault://ops/memory/notes.md"), None);
        assert_eq!(document_date("vault://2026-07-01/notes.md"), None);
        assert_eq!(document_date("vault://ops/12026-07-01.md"), None);
    }

    #[test]
    fn textual_cues_need_dated_documents_and_an_explicit_pointer() {
        let monday = Some("vault://ops/memory/2026-04-27.md");
        let tuesday = Some("vault://ops/memory/2026-04-28.md");
        let thursday = Some("vault://ops/memory/2026-04-30.md");
        // Relative date: only the day after, only the later note.
        assert_eq!(
            textual_cue(
                "Gateway warning seen.",
                monday,
                "Yesterday's gateway warning is gone.",
                tuesday
            ),
            Some((TextualCueKind::RelativeDate, CueDirection::ToMentionsFrom))
        );
        assert_eq!(
            textual_cue(
                "Gateway warning seen.",
                monday,
                "Yesterday's gateway warning is gone.",
                thursday
            ),
            None
        );
        assert_eq!(
            textual_cue(
                "Yesterday was quiet.",
                tuesday,
                "Gateway warning seen.",
                thursday
            ),
            None
        );
        // Dated back-reference phrase, earlier document only.
        assert_eq!(
            textual_cue(
                "Compared with 2026-04-27: the warning moved.",
                tuesday,
                "Warning seen.",
                monday
            ),
            Some((TextualCueKind::DatedReference, CueDirection::FromMentionsTo))
        );
        assert_eq!(
            textual_cue(
                "Recorded on 2026-04-27 and later.",
                tuesday,
                "Warning seen.",
                monday
            ),
            None
        );
        assert_eq!(
            textual_cue(
                "Compared with 2026-04-27 here.",
                monday,
                "Warning seen.",
                tuesday
            ),
            None
        );
        // Answered question, same day or later.
        let asked = "Which yard should each heron truck start on?";
        let answer = "- Both heron trucks start on yard A (provisional).";
        assert_eq!(
            textual_cue(asked, monday, answer, tuesday),
            Some((
                TextualCueKind::AnsweredQuestion,
                CueDirection::ToMentionsFrom
            ))
        );
        assert_eq!(textual_cue(asked, tuesday, answer, monday), None);
        assert_eq!(
            textual_cue(asked, monday, "Heron trucks are fine.", tuesday),
            None
        );
        // Undated documents never cue.
        assert_eq!(textual_cue(asked, None, answer, tuesday), None);
    }
}
