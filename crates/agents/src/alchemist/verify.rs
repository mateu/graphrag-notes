//! The fail-closed citation verifier. Nothing reaches the synthesis unless
//! each citation's span appears verbatim in the cited chunk at the revision
//! the model was shown, and that chunk is still at that revision.

use super::gather::EvidenceChunk;
use super::{AlchemistError, CitationPolicy, VerificationReport, MIN_QUOTE_CHARS};
use serde::Deserialize;

/// Upper bound on the raw model response the parser will consider.
pub(crate) const MAX_DRAFT_BYTES: usize = 256 * 1024;
pub(crate) const MAX_CLAIMS: usize = 32;
pub(crate) const MAX_CITATIONS_PER_CLAIM: usize = 8;
pub(crate) const MAX_CLAIM_CHARS: usize = 2_000;
pub(crate) const MAX_QUOTE_CHARS: usize = 1_024;
pub(crate) const MAX_CONFLICTS: usize = 16;
pub(crate) const MAX_TOPIC_CHARS: usize = 300;
pub(crate) const MAX_OPEN_QUESTIONS: usize = 8;
pub(crate) const MAX_QUESTION_CHARS: usize = 500;

/// Model output. Unknown fields (for example an attempted "proposal" or
/// "edge") are ignored: the Alchemist has no write path to give them.
#[derive(Debug, Default, Deserialize)]
pub(crate) struct DraftSynthesis {
    #[serde(default)]
    pub claims: Vec<DraftClaim>,
    #[serde(default)]
    pub conflicts: Vec<DraftConflict>,
    #[serde(default)]
    pub open_questions: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct DraftClaim {
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub citations: Vec<DraftCitation>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct DraftCitation {
    /// Prompt label such as `C2`, or the full canonical chunk ID.
    #[serde(default)]
    pub chunk: String,
    /// Optional; when present it must equal the pinned revision.
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub quote: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct DraftConflict {
    #[serde(default)]
    pub topic: String,
    /// Indexes into the draft's `claims`.
    #[serde(default)]
    pub claims: Vec<usize>,
}

/// Why a citation was rejected. Serialized names are receipt keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Rejection {
    UnknownChunk,
    StaleRevision,
    QuoteNotFound,
    QuoteTooShort,
    QuoteTooLong,
    OverCitationLimit,
}

impl Rejection {
    fn key(self) -> &'static str {
        match self {
            Self::UnknownChunk => "unknown_chunk",
            Self::StaleRevision => "stale_revision",
            Self::QuoteNotFound => "quote_not_found",
            Self::QuoteTooShort => "quote_too_short",
            Self::QuoteTooLong => "quote_too_long",
            Self::OverCitationLimit => "over_citation_limit",
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct VerifiedCitation {
    pub label: String,
    pub chunk_id: String,
    pub revision: String,
    pub span: String,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct VerifiedClaim {
    pub text: String,
    pub citations: Vec<VerifiedCitation>,
}

#[derive(Debug, Default)]
pub(crate) struct Verified {
    pub claims: Vec<VerifiedClaim>,
    /// Topic and indexes into `claims`, each with at least two kept claims.
    pub conflicts: Vec<(String, Vec<usize>)>,
    pub open_questions: Vec<String>,
    pub report: VerificationReport,
}

fn unverifiable(reason: impl Into<String>, report: VerificationReport) -> AlchemistError {
    AlchemistError::Unverifiable {
        reason: reason.into(),
        report: Box::new(report),
    }
}

/// Parse a bounded JSON draft, tolerating a surrounding Markdown fence.
pub(crate) fn parse_draft(raw: &str) -> Result<DraftSynthesis, AlchemistError> {
    if raw.len() > MAX_DRAFT_BYTES {
        return Err(unverifiable(
            "the synthesis provider returned an oversized draft",
            VerificationReport::default(),
        ));
    }
    let trimmed = raw.trim();
    let body = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .and_then(|inner| inner.trim_end().strip_suffix("```"))
        .unwrap_or(trimmed);
    serde_json::from_str(body.trim()).map_err(|_| {
        unverifiable(
            "the synthesis provider did not return the structured draft JSON; nothing was verified",
            VerificationReport::default(),
        )
    })
}

/// Resolve `C3`, `[C3]` or a full canonical ID to an evidence index.
pub(crate) fn resolve_label(reference: &str, chunks: &[EvidenceChunk]) -> Option<usize> {
    let reference = reference
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']');
    chunks
        .iter()
        .position(|chunk| chunk.label.eq_ignore_ascii_case(reference) || chunk.id == reference)
}

fn check(
    citation: &DraftCitation,
    chunks: &[EvidenceChunk],
) -> Result<VerifiedCitation, Rejection> {
    let index = resolve_label(&citation.chunk, chunks).ok_or(Rejection::UnknownChunk)?;
    let chunk = &chunks[index];
    if !chunk.current
        || citation
            .revision
            .as_deref()
            .is_some_and(|revision| revision.trim() != chunk.revision)
    {
        return Err(Rejection::StaleRevision);
    }
    let quote = citation.quote.trim();
    if quote.chars().count() < MIN_QUOTE_CHARS {
        return Err(Rejection::QuoteTooShort);
    }
    if quote.chars().count() > MAX_QUOTE_CHARS {
        return Err(Rejection::QuoteTooLong);
    }
    let start = chunk.content.find(quote).ok_or(Rejection::QuoteNotFound)?;
    Ok(VerifiedCitation {
        label: chunk.label.clone(),
        chunk_id: chunk.id.clone(),
        revision: chunk.revision.clone(),
        span: quote.to_string(),
        start,
        end: start + quote.len(),
    })
}

fn reject(report: &mut VerificationReport, reason: Rejection) {
    report.citations_rejected += 1;
    *report.rejections.entry(reason.key().into()).or_default() += 1;
}

fn bounded_text(text: &str, max: usize) -> Option<String> {
    let text = text.trim();
    (!text.is_empty() && text.chars().count() <= max).then(|| text.to_string())
}

/// Verify every citation against the pinned evidence. Under `Drop`, claims
/// without a verified citation are dropped and counted; under `Strict`, any
/// rejection fails the request.
pub(crate) fn verify(
    draft: &DraftSynthesis,
    chunks: &[EvidenceChunk],
    policy: CitationPolicy,
    rechecked: usize,
) -> Result<Verified, AlchemistError> {
    let mut report = VerificationReport {
        claims_proposed: draft.claims.len(),
        conflicts_proposed: draft.conflicts.len(),
        chunks_rechecked: rechecked,
        ..Default::default()
    };
    let mut claims = Vec::new();
    // Draft claim index → kept claim index.
    let mut kept_index = vec![None; draft.claims.len()];
    for (position, claim) in draft.claims.iter().enumerate() {
        report.citations_checked += claim.citations.len();
        let text = if position < MAX_CLAIMS {
            bounded_text(&claim.text, MAX_CLAIM_CHARS)
        } else {
            None
        };
        if claim.citations.is_empty() {
            report.uncited_claims += 1;
        }
        let mut citations: Vec<VerifiedCitation> = Vec::new();
        for (number, citation) in claim.citations.iter().enumerate() {
            if number >= MAX_CITATIONS_PER_CLAIM {
                reject(&mut report, Rejection::OverCitationLimit);
                continue;
            }
            match check(citation, chunks) {
                Ok(verified) => {
                    if !citations.iter().any(|existing| {
                        existing.chunk_id == verified.chunk_id && existing.start == verified.start
                    }) {
                        citations.push(verified);
                    }
                }
                Err(reason) => reject(&mut report, reason),
            }
        }
        match text {
            Some(text) if !citations.is_empty() => {
                report.citations_kept += citations.len();
                kept_index[position] = Some(claims.len());
                claims.push(VerifiedClaim { text, citations });
            }
            _ => report.claims_dropped += 1,
        }
    }
    report.claims_kept = claims.len();

    let mut conflicts = Vec::new();
    for (position, conflict) in draft.conflicts.iter().enumerate() {
        let mut members: Vec<usize> = conflict
            .claims
            .iter()
            .filter_map(|index| kept_index.get(*index).copied().flatten())
            .collect();
        members.sort_unstable();
        members.dedup();
        match bounded_text(&conflict.topic, MAX_TOPIC_CHARS) {
            Some(topic) if position < MAX_CONFLICTS && members.len() >= 2 => {
                conflicts.push((topic, members))
            }
            _ => report.conflicts_dropped += 1,
        }
    }

    let mut open_questions = Vec::new();
    for question in &draft.open_questions {
        match bounded_text(question, MAX_QUESTION_CHARS) {
            Some(question) if open_questions.len() < MAX_OPEN_QUESTIONS => {
                open_questions.push(question)
            }
            _ => report.open_questions_dropped += 1,
        }
    }

    if policy == CitationPolicy::Strict
        && (report.citations_rejected > 0 || report.claims_dropped > 0)
    {
        return Err(unverifiable(
            format!(
                "strict citation policy: {} citation(s) rejected and {} claim(s) unverifiable; no synthesis was returned",
                report.citations_rejected, report.claims_dropped
            ),
            report,
        ));
    }
    Ok(Verified {
        claims,
        conflicts,
        open_questions,
        report,
    })
}
