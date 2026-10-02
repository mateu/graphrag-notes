//! Provider-free proposal cards and deliberate, audited connection review.

use super::navigation::{readable_title, shell_quote};
use crate::output::{self, OutputFormat};
use anyhow::{Context, Result};
use graphrag_core::{record_id_to_string, ProposedEdge, ProposedEdgeStatus};
use graphrag_db::{
    repository::{parse_record_id, InspectionProvenance},
    DbError, Repository,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::io::{self, BufRead, Write};
use std::path::Path;

pub(crate) struct ReviewOptions {
    pub id: Option<String>,
    pub status: Option<ProposedEdgeStatus>,
    pub all_statuses: bool,
    pub limit: usize,
    pub interactive: bool,
    pub format: OutputFormat,
}

#[derive(Debug, Serialize)]
struct ReviewEndpoint {
    id: String,
    available: bool,
    title: Option<String>,
    excerpt: Option<String>,
    revision: Option<String>,
    provenance: Option<InspectionProvenance>,
    warnings: Vec<String>,
    inspect_command: Option<String>,
}

#[derive(Debug, Serialize)]
struct ReviewCard {
    id: String,
    status: ProposedEdgeStatus,
    edge_type: String,
    confidence: f32,
    reason: String,
    generator: String,
    generator_version: Option<String>,
    model: Option<String>,
    updated_at: String,
    reviewed_at: Option<String>,
    reviewer: Option<String>,
    action_reason: Option<String>,
    acceptance_is_manual: Option<bool>,
    supersession_reason: Option<String>,
    resulting_edge_id: Option<String>,
    from: ReviewEndpoint,
    to: ReviewEndpoint,
    accept_allowed: bool,
    accept_blocked_reason: Option<String>,
    undo_command: Option<String>,
    review_command: String,
    /// Snapshot of proposal metadata and both inspected endpoint revisions.
    revision: String,
}

fn base_command(database: &Path, config: Option<&Path>) -> Result<String> {
    let absolute = |path: &Path| -> Result<String> {
        let path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()?.join(path)
        };
        Ok(path
            .to_str()
            .context("review follow-up paths must be valid UTF-8")?
            .to_string())
    };
    let mut command = "graphrag".to_string();
    if let Some(path) = config {
        command.push_str(&format!(" --config {}", shell_quote(&absolute(path)?)));
    }
    command.push_str(&format!(" --db-path {}", shell_quote(&absolute(database)?)));
    Ok(command)
}

async fn endpoint(repo: &Repository, id: String, base: &str) -> Result<ReviewEndpoint> {
    match repo.inspect_record(&id, 0).await {
        Ok(record) => {
            let mut characters = record.content.chars();
            let mut excerpt = characters.by_ref().take(500).collect::<String>();
            if characters.next().is_some() {
                excerpt.pop();
                excerpt.push('…');
            }
            Ok(ReviewEndpoint {
                inspect_command: Some(format!(
                    "{base} inspect {} --revision {}",
                    shell_quote(&id),
                    shell_quote(&record.revision)
                )),
                id,
                available: true,
                title: Some(readable_title(record.title.as_deref(), &record.content)),
                excerpt: Some(excerpt),
                revision: Some(record.revision),
                provenance: Some(record.provenance),
                warnings: record.warnings,
            })
        }
        Err(DbError::NotFound(_, _)) => Ok(ReviewEndpoint {
            id,
            available: false,
            title: None,
            excerpt: None,
            revision: None,
            provenance: None,
            warnings: vec![
                "Note is missing or belongs to an unavailable source generation.".into(),
            ],
            inspect_command: None,
        }),
        Err(error) => Err(error.into()),
    }
}

async fn card(repo: &Repository, proposal: ProposedEdge, base: &str) -> Result<ReviewCard> {
    let id = proposal.id.as_ref().context("stored proposal has no ID")?;
    let from = endpoint(repo, record_id_to_string(&proposal.from_id), base).await?;
    let to = endpoint(repo, record_id_to_string(&proposal.to_id), base).await?;
    let blocked = if proposal.status != ProposedEdgeStatus::Pending {
        Some(format!("Proposal is {}, not pending.", proposal.status))
    } else if !from.available || !to.available {
        Some("Both notes must be visible before this proposal can be accepted.".into())
    } else {
        None
    };
    let resulting_edge_id = proposal.resulting_edge_id.as_ref().map(record_id_to_string);
    let undo_command = resulting_edge_id
        .as_ref()
        .filter(|_| proposal.status == ProposedEdgeStatus::Accepted)
        .map(|edge| format!("{base} edges undo {} --yes", shell_quote(edge)));
    let revision = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&(
            &proposal,
            &from.revision,
            &to.revision,
        ))?)
    );
    Ok(ReviewCard {
        id: record_id_to_string(id),
        status: proposal.status,
        edge_type: proposal.edge_type.to_string(),
        confidence: proposal.confidence,
        reason: proposal.reason,
        generator: proposal.generator,
        generator_version: proposal.generator_version,
        model: proposal.model,
        updated_at: proposal.updated_at.to_rfc3339(),
        reviewed_at: proposal.reviewed_at.map(|date| date.to_rfc3339()),
        reviewer: proposal.reviewer,
        action_reason: proposal.action_reason,
        acceptance_is_manual: proposal.acceptance_is_manual,
        supersession_reason: proposal.supersession_reason,
        resulting_edge_id,
        from,
        to,
        accept_allowed: blocked.is_none(),
        accept_blocked_reason: blocked,
        undo_command,
        review_command: format!(
            "{base} garden review --id {} --interactive",
            shell_quote(&record_id_to_string(id))
        ),
        revision,
    })
}

fn render_endpoint(writer: &mut dyn Write, label: &str, note: &ReviewEndpoint) -> io::Result<()> {
    writeln!(
        writer,
        "  {label}: {} [{}]",
        note.title.as_deref().unwrap_or("Unavailable note"),
        note.id
    )?;
    if let Some(excerpt) = &note.excerpt {
        for line in excerpt.lines() {
            writeln!(writer, "    {line}")?;
        }
    }
    if let Some(provenance) = &note.provenance {
        if let Some(id) = &provenance.source_id {
            writeln!(writer, "    Source ID: {id}")?;
        }
        if let Some(source_type) = &provenance.source_type {
            writeln!(writer, "    Source type: {source_type}")?;
        }
        if let Some(generation) = provenance.source_generation {
            writeln!(writer, "    Source generation: {generation}")?;
        }
        if let Some(uri) = &provenance.source_uri {
            writeln!(writer, "    Source: {uri}")?;
        } else if provenance.source_id.is_none() && provenance.conversation_id.is_none() {
            writeln!(writer, "    Source: manual/local note")?;
        } else {
            writeln!(
                writer,
                "    Source URI: unavailable; stored provenance retained"
            )?;
        }
        if !provenance.heading_path.is_empty() {
            writeln!(
                writer,
                "    Heading: {}",
                provenance.heading_path.join(" > ")
            )?;
        }
        if let Some(line) = provenance.start_line {
            writeln!(
                writer,
                "    Lines: {line}-{} (at last import)",
                provenance.end_line.unwrap_or(line)
            )?;
        }
        if let Some(uuid) = &provenance.conversation_uuid {
            writeln!(writer, "    Conversation UUID: {uuid}")?;
        }
        if let Some(uuid) = &provenance.message_uuid {
            writeln!(writer, "    Message UUID: {uuid}")?;
        } else if let Some(key) = &provenance.message_key {
            writeln!(writer, "    Message key: {key}")?;
        }
    }
    for warning in &note.warnings {
        writeln!(writer, "    Warning: {warning}")?;
    }
    if let Some(command) = &note.inspect_command {
        writeln!(writer, "    Inspect full note: {command}")?;
    }
    Ok(())
}

fn render_card(writer: &mut dyn Write, card: &ReviewCard) -> io::Result<()> {
    writeln!(writer, "Proposal: {} [{}]", card.id, card.status)?;
    writeln!(
        writer,
        "Relationship: {} | confidence {:.1}%",
        card.edge_type,
        card.confidence * 100.0
    )?;
    writeln!(writer, "Reason: {}", card.reason)?;
    writeln!(
        writer,
        "Generator: {}{}",
        card.generator,
        card.generator_version
            .as_ref()
            .map(|version| format!(" ({version})"))
            .unwrap_or_default()
    )?;
    render_endpoint(writer, "From", &card.from)?;
    render_endpoint(writer, "To", &card.to)?;
    if let Some(reason) = &card.accept_blocked_reason {
        writeln!(writer, "Accept unavailable: {reason}")?;
    }
    if let Some(reviewer) = &card.reviewer {
        writeln!(writer, "Reviewed by: {reviewer}")?;
    }
    if let Some(reason) = &card.action_reason {
        writeln!(writer, "Decision reason: {reason}")?;
    }
    if let Some(reason) = &card.supersession_reason {
        writeln!(writer, "Supersession reason: {reason}")?;
    }
    if let Some(command) = &card.undo_command {
        writeln!(writer, "Undo accepted edge: {command}")?;
    }
    writeln!(writer, "Review interactively: {}", card.review_command)?;
    writeln!(writer)
}

pub(crate) async fn review(
    repo: &Repository,
    options: ReviewOptions,
    database: &Path,
    config: Option<&Path>,
) -> Result<()> {
    if !(1..=200).contains(&options.limit) {
        anyhow::bail!("review --limit must be between 1 and 200");
    }
    if options.interactive && options.format != OutputFormat::Human {
        anyhow::bail!(
            "interactive review requires --format human; JSON/JSONL inbox output is read-only"
        );
    }
    let proposals = if let Some(id) = &options.id {
        if !id.starts_with("proposed_edge:") || id.chars().any(char::is_control) {
            anyhow::bail!("review --id requires a full proposed_edge:ID");
        }
        let id = parse_record_id(id, Some("proposed_edge"))
            .map_err(|error| anyhow::anyhow!("invalid review proposal ID: {error}"))?;
        vec![repo
            .get_edge_proposal(&id)
            .await?
            .ok_or_else(|| DbError::NotFound("proposed_edge".into(), record_id_to_string(&id)))?]
    } else {
        let status = if options.all_statuses {
            None
        } else {
            Some(options.status.unwrap_or(ProposedEdgeStatus::Pending))
        };
        repo.list_edge_proposals(status, options.limit).await?
    };
    let base = base_command(database, config)?;
    if options.interactive {
        return interactive(repo, proposals, &base, &mut io::stdin().lock()).await;
    }
    let mut cards = Vec::with_capacity(proposals.len());
    for proposal in proposals {
        cards.push(card(repo, proposal, &base).await?);
    }
    if options.format == OutputFormat::Jsonl {
        return output::print_jsonl("garden.review", cards);
    }
    output::print(
        options.format,
        "garden.review",
        serde_json::json!({"proposals": cards}),
        |writer| {
            if cards.is_empty() {
                writeln!(writer, "No matching proposals. Use garden scan to generate proposals, or --all-statuses to review history.")?;
            }
            for card in &cards {
                render_card(writer, card)?;
            }
            if !cards.is_empty() {
                writeln!(
                    writer,
                    "Use a printed 'Review interactively' command to accept, reject, or skip that proposal."
                )?;
            }
            Ok(())
        },
    )
}

fn prompt(reader: &mut impl BufRead, text: &str) -> Result<Option<String>> {
    let mut stderr = io::stderr().lock();
    write!(stderr, "{text}")?;
    stderr.flush()?;
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    Ok(Some(line.trim().to_string()))
}

#[derive(Clone, Copy)]
enum Decision {
    Accept,
    Reject,
    Undo,
}

impl Decision {
    fn word(self) -> &'static str {
        match self {
            Self::Accept => "accept",
            Self::Reject => "reject",
            Self::Undo => "undo",
        }
    }
}

async fn decide(
    repo: &Repository,
    shown: &ReviewCard,
    decision: Decision,
    reason: Option<String>,
) -> Result<ReviewCard> {
    let id = parse_record_id(&shown.id, Some("proposed_edge"))?;
    let current = repo
        .get_edge_proposal(&id)
        .await?
        .ok_or_else(|| DbError::NotFound("proposed_edge".into(), shown.id.clone()))?;
    let fresh = card(repo, current, "graphrag").await?;
    if shown.revision != fresh.revision {
        anyhow::bail!("proposal or a note changed during review; refusing this decision. Review the updated card before deciding again");
    }
    match decision {
        Decision::Accept => {
            if !fresh.accept_allowed {
                anyhow::bail!(
                    "refusing acceptance: {}",
                    fresh
                        .accept_blocked_reason
                        .as_deref()
                        .unwrap_or("proposal unavailable")
                );
            }
            repo.accept_edge_proposal(&id, Some("cli interactive review".into()), reason, true)
                .await?;
        }
        Decision::Reject => {
            if fresh.status != ProposedEdgeStatus::Pending {
                anyhow::bail!(
                    "refusing rejection: proposal is {}, not pending",
                    fresh.status
                );
            }
            repo.reject_edge_proposal(&id, Some("cli interactive review".into()), reason)
                .await?;
        }
        Decision::Undo => {
            if fresh.status != ProposedEdgeStatus::Accepted {
                anyhow::bail!("refusing undo: proposal is {}, not accepted", fresh.status);
            }
            let edge = fresh.resulting_edge_id.as_deref().context("accepted proposal has no edge ID; use the existing proposals accept command to recover acceptance first")?;
            let edge = parse_record_id(edge, None)?;
            repo.undo_edge(
                &edge,
                Some(
                    reason.unwrap_or_else(|| {
                        "accepted edge undone through interactive review".into()
                    }),
                ),
            )
            .await?;
        }
    }
    let updated = repo
        .get_edge_proposal(&id)
        .await?
        .context("reviewed proposal disappeared")?;
    card(repo, updated, "graphrag").await
}

async fn interactive(
    repo: &Repository,
    proposals: Vec<ProposedEdge>,
    base: &str,
    reader: &mut impl BufRead,
) -> Result<()> {
    if proposals.is_empty() {
        println!("No matching proposals. Use garden scan or --all-statuses to review history.");
        return Ok(());
    }
    let mut decisions = 0;
    let mut skipped = 0;
    for proposal in proposals {
        let id = proposal.id.context("stored proposal has no ID")?;
        loop {
            let current = repo.get_edge_proposal(&id).await?.ok_or_else(|| {
                DbError::NotFound("proposed_edge".into(), record_id_to_string(&id))
            })?;
            let shown = card(repo, current, base).await?;
            render_card(&mut io::stdout().lock(), &shown)?;
            let Some(action) = prompt(
                reader,
                "Action [a]ccept / [r]eject / [s]kip / [v]iew full notes / [u]ndo / [q]uit: ",
            )?
            else {
                return Ok(());
            };
            let decision = match action.as_str() {
                "s" | "skip" => {
                    skipped += 1;
                    eprintln!("Skipped; proposal and audit data unchanged.");
                    break;
                }
                "q" | "quit" => return Ok(()),
                "v" | "view" => {
                    for endpoint in [&shown.from, &shown.to] {
                        if endpoint.available {
                            let record = repo.inspect_record(&endpoint.id, 0).await?;
                            println!("Full note {}:\n{}\n", endpoint.id, record.content);
                        }
                    }
                    continue;
                }
                "a" | "accept" => Decision::Accept,
                "r" | "reject" => Decision::Reject,
                "u" | "undo" => Decision::Undo,
                _ => {
                    eprintln!("Choose an action; no decision was recorded.");
                    continue;
                }
            };
            if matches!(decision, Decision::Accept) && !shown.accept_allowed {
                eprintln!(
                    "Accept unavailable: {}",
                    shown
                        .accept_blocked_reason
                        .as_deref()
                        .unwrap_or("proposal unavailable")
                );
                continue;
            }
            if matches!(decision, Decision::Reject) && shown.status != ProposedEdgeStatus::Pending {
                eprintln!("Reject unavailable: proposal is {}.", shown.status);
                continue;
            }
            if matches!(decision, Decision::Undo) && shown.undo_command.is_none() {
                eprintln!("Undo unavailable: this proposal has no accepted edge.");
                continue;
            }
            let word = decision.word();
            let Some(confirmation) = prompt(
                reader,
                &format!("Type '{word}' to confirm this {word} decision, or Enter to cancel: "),
            )?
            else {
                return Ok(());
            };
            if confirmation != word {
                eprintln!("Decision cancelled; no changes made.");
                continue;
            }
            let Some(reason) = prompt(
                reader,
                "Decision reason (optional; Enter uses a review default): ",
            )?
            else {
                return Ok(());
            };
            let reason = Some(if reason.is_empty() {
                format!("explicit interactive review {word}")
            } else {
                reason
            });
            match decide(repo, &shown, decision, reason).await {
                Ok(_) => {
                    let current = repo
                        .get_edge_proposal(&id)
                        .await?
                        .context("reviewed proposal disappeared")?;
                    let reviewed = card(repo, current, base).await?;
                    println!("Decision recorded: {} is {}.", reviewed.id, reviewed.status);
                    if let Some(command) = reviewed.undo_command {
                        println!("Undo accepted edge: {command}");
                    }
                    decisions += 1;
                    break;
                }
                Err(error) => eprintln!("Decision refused: {error}"),
            }
        }
    }
    println!("Review complete: {decisions} decision(s), {skipped} skipped.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use graphrag_core::Note;
    use graphrag_db::init_memory;

    async fn pending() -> (Repository, ProposedEdge) {
        let repo = Repository::new(init_memory().await.unwrap());
        let left = repo
            .create_note(Note::new("Pilot evidence before review"))
            .await
            .unwrap();
        let right = repo
            .create_note(Note::new("Related context before review"))
            .await
            .unwrap();
        let proposal = repo
            .upsert_gardener_proposal(
                left.id.as_ref().unwrap(),
                right.id.as_ref().unwrap(),
                0.9,
                "Original evidence".into(),
                None,
                None,
            )
            .await
            .unwrap();
        (repo, proposal)
    }

    #[tokio::test]
    async fn changing_a_displayed_note_refuses_acceptance_without_audit_writes() {
        let (repo, proposal) = pending().await;
        let shown = card(&repo, proposal.clone(), "graphrag").await.unwrap();
        let endpoint = record_id_to_string(&proposal.from_id);
        let mut note = repo.get_note(&endpoint).await.unwrap().unwrap();
        note.content = "Changed evidence after the card was shown".into();
        repo.update_note(&endpoint, note).await.unwrap();
        let error = decide(
            &repo,
            &shown,
            Decision::Accept,
            Some("obsolete review".into()),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("changed during review"));
        let after = repo
            .get_edge_proposal(proposal.id.as_ref().unwrap())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(after.status, ProposedEdgeStatus::Pending);
        assert_eq!(after.updated_at, proposal.updated_at);
        assert!(after.reviewer.is_none());
        assert!(after.reviewed_at.is_none());
        assert!(after.action_reason.is_none());
        assert!(after.resulting_edge_id.is_none());
    }

    #[tokio::test]
    async fn rescanning_a_displayed_proposal_requires_a_new_review() {
        let (repo, proposal) = pending().await;
        let shown = card(&repo, proposal.clone(), "graphrag").await.unwrap();
        let updated = repo
            .upsert_gardener_proposal(
                &proposal.from_id,
                &proposal.to_id,
                0.75,
                "Revised evidence from a later scan".into(),
                None,
                None,
            )
            .await
            .unwrap();
        let error = decide(
            &repo,
            &shown,
            Decision::Reject,
            Some("obsolete reason".into()),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("changed during review"));
        let after = repo
            .get_edge_proposal(proposal.id.as_ref().unwrap())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(after.status, ProposedEdgeStatus::Pending);
        assert_eq!(after.updated_at, updated.updated_at);
        assert_eq!(after.reason, updated.reason);
        assert!(after.reviewed_at.is_none());
        assert!(after.action_reason.is_none());
    }

    #[tokio::test]
    async fn endpoint_deletion_cancels_a_displayed_decision_and_keeps_retirement() {
        let (repo, proposal) = pending().await;
        let shown = card(&repo, proposal.clone(), "graphrag").await.unwrap();
        repo.delete_note(&record_id_to_string(&proposal.from_id))
            .await
            .unwrap();
        assert!(decide(&repo, &shown, Decision::Accept, None).await.is_err());
        let after = repo
            .get_edge_proposal(proposal.id.as_ref().unwrap())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(after.status, ProposedEdgeStatus::Superseded);
        let unavailable = card(&repo, after, "graphrag").await.unwrap();
        assert!(!unavailable.accept_allowed);
        assert!(!unavailable.from.available || !unavailable.to.available);
    }
}
