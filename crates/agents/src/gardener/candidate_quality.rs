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

/// Stable small-integer identities for words, shared across calls.
fn word_identities<'a>(
    words: &'a [String],
    vocabulary: &mut std::collections::HashMap<&'a str, u32>,
) -> Vec<u32> {
    words
        .iter()
        .map(|word| {
            let next = vocabulary.len() as u32;
            *vocabulary.entry(word.as_str()).or_insert(next)
        })
        .collect()
}

/// Longest run of consecutive words two notes share, as a share of the
/// shorter note's word count; zero when the run is shorter than
/// [`MIN_SHARED_SPAN_WORDS`]. Overlapping chunks of one file, copies and
/// repeated templated output share long runs; independent notes about one
/// topic rarely do.
///
/// Linear in the notes' lengths: every run of [`MIN_SHARED_SPAN_WORDS`]
/// words of the longer note is indexed by its first occurrence, and the
/// shorter note is walked once, extending each verified match greedily and
/// skipping past it, so each word is compared a bounded number of times.
/// With repeated text this can under-count the longest run (a lower bound),
/// which only ever keeps a pair; a verbatim copy of most of the shorter note
/// is always found.
pub(crate) fn shared_span_share(left: &str, right: &str) -> f32 {
    use std::collections::HashMap;
    let (left, right) = (words(left), words(right));
    let (short, long) = if left.len() <= right.len() {
        (left, right)
    } else {
        (right, left)
    };
    let window = MIN_SHARED_SPAN_WORDS;
    if short.len() < window {
        return 0.0;
    }
    // Word identities, so windows compare as integer slices.
    let mut vocabulary = HashMap::<&str, u32>::new();
    let long_ids = word_identities(&long, &mut vocabulary);
    let short_ids = word_identities(&short, &mut vocabulary);
    let mut starts = HashMap::<&[u32], usize>::new();
    for (start, run) in long_ids.windows(window).enumerate() {
        starts.entry(run).or_insert(start);
    }
    let mut best = 0usize;
    let mut position = 0usize;
    while position + window <= short_ids.len() {
        let Some(&start) = starts.get(&short_ids[position..position + window]) else {
            position += 1;
            continue;
        };
        let mut length = window;
        while position + length < short_ids.len()
            && start + length < long_ids.len()
            && short_ids[position + length] == long_ids[start + length]
        {
            length += 1;
        }
        best = best.max(length);
        position += length.saturating_sub(window - 1);
    }
    if best < window {
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

/// `text` (dated `own`) points back at a document dated `other`. A
/// "yesterday" that `relative_ok` declines is skipped, and reported through
/// `declined`, so a dated back-reference can still cue.
fn back_reference(
    text: &str,
    own: NaiveDate,
    other: NaiveDate,
    relative_ok: impl FnOnce() -> bool,
    declined: &mut bool,
) -> Option<TextualCueKind> {
    if other >= own {
        return None;
    }
    let lower = text.to_lowercase();
    if other.succ_opt() == Some(own) && whole_word(&lower, "yesterday") {
        if relative_ok() {
            return Some(TextualCueKind::RelativeDate);
        }
        *declined = true;
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
#[cfg(test)]
pub(crate) fn textual_cue(
    from_text: &str,
    from_origin: Option<&str>,
    to_text: &str,
    to_origin: Option<&str>,
) -> Option<(TextualCueKind, CueDirection)> {
    textual_cue_checked(from_text, from_origin, to_text, to_origin, |_| true).cue
}

/// A [`textual_cue_checked`] result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct CueOutcome {
    pub cue: Option<(TextualCueKind, CueDirection)>,
    /// A "yesterday" cue was found but `relative_ok` declined it (and no
    /// other cue was found in its place).
    pub declined_relative_date: Option<CueDirection>,
}

/// [`textual_cue`], where `relative_ok(direction)` decides whether a
/// "yesterday" cue carried in `direction` counts. A declined relative date
/// falls through to the weaker kinds.
pub(crate) fn textual_cue_checked(
    from_text: &str,
    from_origin: Option<&str>,
    to_text: &str,
    to_origin: Option<&str>,
    relative_ok: impl Fn(CueDirection) -> bool,
) -> CueOutcome {
    let mut outcome = CueOutcome::default();
    let (Some(from_date), Some(to_date)) = (
        from_origin.and_then(document_date),
        to_origin.and_then(document_date),
    ) else {
        return outcome;
    };
    let mut declined = [false; 2];
    outcome.cue = textual_cue_dated(
        (from_text, from_date),
        (to_text, to_date),
        &relative_ok,
        &mut declined,
    );
    if outcome.cue.is_none() {
        outcome.declined_relative_date = if declined[0] {
            Some(CueDirection::FromMentionsTo)
        } else if declined[1] {
            Some(CueDirection::ToMentionsFrom)
        } else {
            None
        };
    }
    outcome
}

fn textual_cue_dated(
    (from_text, from_date): (&str, NaiveDate),
    (to_text, to_date): (&str, NaiveDate),
    relative_ok: &impl Fn(CueDirection) -> bool,
    declined: &mut [bool; 2],
) -> Option<(TextualCueKind, CueDirection)> {
    let from_ok = || relative_ok(CueDirection::FromMentionsTo);
    if let Some(kind) = back_reference(from_text, from_date, to_date, from_ok, &mut declined[0]) {
        return Some((kind, CueDirection::FromMentionsTo));
    }
    let to_ok = || relative_ok(CueDirection::ToMentionsFrom);
    if let Some(kind) = back_reference(to_text, to_date, from_date, to_ok, &mut declined[1]) {
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

/// Clock times (`7:05`, `22:37`) written in `text`, as minutes after
/// midnight. A run of three colon-separated numbers (`04:54:39`) or a time
/// glued to other digits is not a clock time here.
pub(crate) fn clock_times(text: &str) -> BTreeSet<u16> {
    let bytes = text.as_bytes();
    let mut found = BTreeSet::new();
    for (colon, _) in text.match_indices(':') {
        let hours_start = (0..colon)
            .rev()
            .take_while(|&index| bytes[index].is_ascii_digit())
            .last();
        let Some(hours_start) = hours_start else {
            continue;
        };
        let hours = &text[hours_start..colon];
        let minutes = text[colon + 1..]
            .bytes()
            .take_while(u8::is_ascii_digit)
            .count();
        let after = colon + 1 + minutes;
        let bounded_before = hours_start == 0 || bytes[hours_start - 1] != b':';
        let bounded_after = bytes.get(after).is_none_or(|byte| *byte != b':');
        if hours.len() > 2 || minutes != 2 || !bounded_before || !bounded_after {
            continue;
        }
        let (Ok(hours), Ok(minutes)) =
            (hours.parse::<u16>(), text[colon + 1..after].parse::<u16>())
        else {
            continue;
        };
        if hours < 24 && minutes < 60 {
            found.insert(hours * 60 + minutes);
        }
    }
    found
}

/// Words that place a statement in time; they never make a topic span.
const TEMPORAL_WORDS: &[&str] = &[
    "yesterday",
    "today",
    "tomorrow",
    "tonight",
    "morning",
    "afternoon",
    "evening",
    "night",
    "last",
    "earlier",
    "later",
    "since",
    "prior",
    "previou",
    "next",
    "day",
    "week",
];

/// Adjacent content-word pairs of `text`: a two-word topic span such as
/// "service path". Temporal words, stop words and number-only tokens are
/// gaps, so a span never bridges them.
fn topic_spans(text: &str) -> BTreeSet<(String, String)> {
    let tokens = text
        .split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '-'))
        .filter(|token| !token.trim_matches(['_', '-']).is_empty())
        .map(|token| {
            // Stop and temporal words are checked before and after the plural
            // `s` is dropped ("this", "days"), so neither form makes a span.
            let token = token.trim_matches(['_', '-']).to_lowercase();
            let stem =
                if token.chars().count() > 3 && token.ends_with('s') && !token.ends_with("ss") {
                    token[..token.len() - 1].to_owned()
                } else {
                    token.clone()
                };
            let excluded =
                |word: &str| STOP_WORDS.contains(&word) || TEMPORAL_WORDS.contains(&word);
            let content = stem.chars().count() >= 3
                && stem.chars().any(char::is_alphabetic)
                && !excluded(&token)
                && !excluded(&stem);
            content.then_some(stem)
        })
        .collect::<Vec<_>>();
    tokens
        .windows(2)
        .filter_map(|pair| match pair {
            [Some(left), Some(right)] => Some((left.clone(), right.clone())),
            _ => None,
        })
        .collect()
}

/// Sentences of `text` (split at line breaks and at `.`, `!`, `?` or `;`
/// followed by whitespace).
fn sentences(text: &str) -> Vec<&str> {
    let mut found = Vec::new();
    for line in text.split('\n') {
        let mut start = 0;
        let chars = line.char_indices().collect::<Vec<_>>();
        for (position, (index, c)) in chars.iter().enumerate() {
            let ends = matches!(c, '.' | '!' | '?' | ';')
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
    found.retain(|sentence| !sentence.trim().is_empty());
    found
}

/// A "yesterday" in `cue_text` resolves to the counterpart (`other_text`,
/// whose entry heading is `other_heading`): one sentence saying "yesterday"
/// names no clock time, or a clock time of the counterpart's heading, and
/// shares a two-word topic span with the counterpart. A monitoring window
/// ("since 22:37 yesterday") or an aggregate ("yesterday's notes") that
/// names no time and nothing the counterpart says does not resolve.
pub(crate) fn relative_date_resolves(
    cue_text: &str,
    other_text: &str,
    other_heading: &[String],
) -> bool {
    let heading = other_heading.join("\n");
    let other_times = clock_times(&heading);
    let other_spans = topic_spans(&format!("{other_text}\n{heading}"));
    sentences(cue_text).into_iter().any(|sentence| {
        if !whole_word(&sentence.to_lowercase(), "yesterday") {
            return false;
        }
        let times = clock_times(sentence);
        (times.is_empty() || !times.is_disjoint(&other_times))
            && !topic_spans(sentence).is_disjoint(&other_spans)
    })
}

/// The digit-collapsed, lowercased form of a note's entry heading (the last
/// heading-path element) when that heading carries a clock time: the
/// template of a timestamped log entry ("#:# mdt — heartbeat recheck").
pub(crate) fn timestamped_entry_template(heading_path: &[String]) -> Option<String> {
    let heading = heading_path.last()?;
    if clock_times(heading).is_empty() {
        return None;
    }
    Some(lineage_family(&heading.trim().to_lowercase()))
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
        // Long notes stay linear: two ~65 KB repetitive texts finish fast.
        let unit = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu ";
        let huge = unit.repeat(5_000);
        let started = std::time::Instant::now();
        assert!(shared_span_share(&huge, &huge) > 0.99);
        assert!(started.elapsed() < std::time::Duration::from_secs(2));
        let shifted = format!("omega {huge}");
        assert!(shared_span_share(&huge, &shifted) > 0.99);
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
    fn clock_times_are_hours_and_minutes_only() {
        assert_eq!(
            clock_times("since 22:37 yesterday, and 7:05 today")
                .into_iter()
                .collect::<Vec<_>>(),
            vec![7 * 60 + 5, 22 * 60 + 37]
        );
        // Seconds-bearing stamps, ratios and out-of-range values are not times.
        assert!(clock_times("Session 04:54:39 UTC, ratio 3:1, 25:00, 123:45").is_empty());
        assert!(clock_times("no times here").is_empty());
    }

    #[test]
    fn relative_dates_resolve_only_to_a_matching_counterpart() {
        let heading = |parts: &[&str]| {
            parts
                .iter()
                .map(|part| (*part).to_owned())
                .collect::<Vec<_>>()
        };
        let counterpart = "Gateway running. The service PATH warning is shown.";
        let lane = heading(&["2026-04-27", "Maintenance lane — 08:20 MDT"]);
        // Shares "service path" and names no other time: resolves.
        assert!(relative_date_resolves(
            "Yesterday's service PATH warning is not present today.",
            counterpart,
            &lane
        ));
        // Names the counterpart's own time: still resolves.
        assert!(relative_date_resolves(
            "The service PATH warning from 08:20 yesterday is gone.",
            counterpart,
            &lane
        ));
        // A monitoring window at another time does not.
        assert!(!relative_date_resolves(
            "The service PATH warning is unchanged since 22:37 yesterday.",
            counterpart,
            &lane
        ));
        // An aggregate pointer sharing no topic span does not.
        assert!(!relative_date_resolves(
            "Consistent with yesterday's notes. Probes 200.",
            counterpart,
            &lane
        ));
        // A topic span elsewhere in the note, not in the cue sentence, does not.
        assert!(!relative_date_resolves(
            "Service PATH warning shown. Yesterday was quiet.",
            counterpart,
            &lane
        ));
        // Stop words never make a span, before or after a plural `s` is
        // dropped ("this does" would otherwise stem to "thi doe").
        assert!(!relative_date_resolves(
            "Yesterday this does not work.",
            "This does something else.",
            &[]
        ));
        // Temporal words never make a span ("yesterday morning").
        assert!(!relative_date_resolves(
            "Same as yesterday morning.",
            "Checked this morning.",
            &lane
        ));
    }

    #[test]
    fn timestamped_entry_templates_collapse_digits() {
        let entry = |text: &str| vec!["2026-06-27".to_owned(), text.to_owned()];
        assert_eq!(
            timestamped_entry_template(&entry("19:07 MDT — Heartbeat recheck")),
            timestamped_entry_template(&entry("07:37 MDT — Heartbeat Recheck"))
        );
        assert_eq!(
            timestamped_entry_template(&entry("19:07 MDT — Heartbeat recheck")).as_deref(),
            Some("#:# mdt — heartbeat recheck")
        );
        assert_eq!(timestamped_entry_template(&entry("Owner decisions")), None);
        assert_eq!(timestamped_entry_template(&[]), None);
    }

    #[test]
    fn a_declined_relative_date_falls_through_and_is_reported() {
        let monday = Some("vault://ops/memory/2026-04-27.md");
        let tuesday = Some("vault://ops/memory/2026-04-28.md");
        let outcome = textual_cue_checked(
            "Gateway warning seen.",
            monday,
            "Yesterday's gateway warning is gone.",
            tuesday,
            |_| false,
        );
        assert_eq!(outcome.cue, None);
        assert_eq!(
            outcome.declined_relative_date,
            Some(CueDirection::ToMentionsFrom)
        );
        // A dated back-reference in the same note still cues.
        let outcome = textual_cue_checked(
            "Gateway warning seen.",
            monday,
            "Yesterday was quiet. Compared with 2026-04-27: the warning moved.",
            tuesday,
            |_| false,
        );
        assert_eq!(
            outcome.cue,
            Some((TextualCueKind::DatedReference, CueDirection::ToMentionsFrom))
        );
        assert_eq!(outcome.declined_relative_date, None);
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
