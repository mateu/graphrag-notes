//! `garden scan --max-candidates`: the bounded, high-precision candidate scan.
//!
//! The same wire request runs in-process against an owned database or through
//! `--server` against the owning service. Budgets are always sent explicitly
//! and the receipt must echo them before anything is rendered.
use crate::cli::GardenCommand;
use crate::output::{self, OutputFormat};
use anyhow::Result;
use graphrag_agents::gardener::CandidatePolicy;
use graphrag_application::{
    ApplicationError, RelationshipCandidateScanRequest, RelationshipCandidateScanResponse,
    RelationshipCandidateScope,
};
use graphrag_config::GardenerCandidateConfig;
use serde_json::Value;
use std::io::{self, Write};

pub(crate) const TOOL: &str = "scan_relationship_candidates";
pub(crate) const DEFAULT_MAX_ENDPOINTS: usize = 25;
pub(crate) const DEFAULT_DEADLINE_MS: u64 = 10_000;

/// Operator policy from `[gardener.candidates]`.
pub(crate) fn policy(config: &GardenerCandidateConfig) -> CandidatePolicy {
    CandidatePolicy {
        similarity_floor: config.similarity_floor,
        neighbors_per_endpoint: config.neighbors_per_endpoint,
        max_per_endpoint: config.max_per_endpoint,
        max_entity_note_frequency: config.max_entity_note_frequency,
    }
}

/// The explicit request for a candidate-mode scan; `None` selects the legacy
/// host-local scan (no `--max-candidates`).
pub(crate) fn request(
    command: &GardenCommand,
) -> Option<(RelationshipCandidateScanRequest, OutputFormat)> {
    let GardenCommand::Scan {
        dry_run,
        max_candidates,
        max_endpoints,
        deadline_ms,
        notes,
        sources,
        counterpart_sources,
        after,
        min_similarity,
        max_per_endpoint,
        format,
    } = command
    else {
        return None;
    };
    let scope = if !notes.is_empty() {
        RelationshipCandidateScope::Notes { ids: notes.clone() }
    } else if !sources.is_empty() {
        RelationshipCandidateScope::Sources {
            ids: sources.clone(),
        }
    } else {
        RelationshipCandidateScope::Orphans {}
    };
    Some((
        RelationshipCandidateScanRequest {
            scope,
            counterpart_source_ids: counterpart_sources.clone(),
            after_note_id: after.clone(),
            max_candidates: (*max_candidates)?,
            max_endpoints: max_endpoints.unwrap_or(DEFAULT_MAX_ENDPOINTS),
            deadline_ms: deadline_ms.unwrap_or(DEFAULT_DEADLINE_MS),
            min_similarity: *min_similarity,
            max_per_endpoint: *max_per_endpoint,
            dry_run: *dry_run,
        },
        format.unwrap_or(OutputFormat::Human),
    ))
}

/// A receipt is trusted only when it echoes exactly what was requested and
/// stays within those bounds.
pub(crate) fn verify_receipt(
    request: &RelationshipCandidateScanRequest,
    receipt: &RelationshipCandidateScanResponse,
) -> Result<()> {
    let policy = &receipt.policy;
    let consistent = receipt.limits.max_candidates == request.max_candidates
        && receipt.limits.max_endpoints == request.max_endpoints
        && receipt.limits.deadline_ms == request.deadline_ms
        && receipt.dry_run == request.dry_run
        && receipt.scope == request.scope
        && receipt.counterpart_source_ids == request.counterpart_source_ids
        && receipt.after_note_id == request.after_note_id
        && receipt.candidates.len() <= request.max_candidates
        && receipt.endpoints_examined <= request.max_endpoints
        && request
            .min_similarity
            .is_none_or(|floor| policy.similarity_floor >= floor)
        && request
            .max_per_endpoint
            .is_none_or(|cap| policy.max_per_endpoint <= cap)
        && receipt.candidates.iter().all(|candidate| {
            let proposal_consistent = if request.dry_run {
                candidate.proposal.is_none()
            } else {
                candidate
                    .proposal
                    .as_ref()
                    .is_none_or(|proposal| proposal.id.starts_with("proposed_edge:"))
            };
            candidate.relationship == "related_to"
                && candidate.similarity >= policy.similarity_floor
                && !candidate.signals.is_empty()
                && proposal_consistent
        });
    if !consistent {
        return Err(ApplicationError::Compatibility(
            "candidate scan receipt does not echo its request budgets and scope; list pending proposals before retrying".into(),
        )
        .into());
    }
    Ok(())
}

/// Parse and verify a remote envelope's receipt against the sent request.
pub(crate) fn remote_receipt(
    arguments: &Value,
    value: &Value,
) -> Result<RelationshipCandidateScanResponse> {
    let incompatible =
        || ApplicationError::Compatibility("incompatible candidate scan result contract".into());
    let request: RelationshipCandidateScanRequest =
        serde_json::from_value(arguments.clone()).map_err(|_| incompatible())?;
    let receipt: RelationshipCandidateScanResponse =
        serde_json::from_value(value["data"].clone()).map_err(|_| incompatible())?;
    verify_receipt(&request, &receipt)?;
    Ok(receipt)
}

pub(crate) fn print(
    format: OutputFormat,
    receipt: &RelationshipCandidateScanResponse,
) -> Result<()> {
    output::print(format, "garden scan", receipt, |writer| {
        render(writer, receipt)
    })
}

fn text(value: &str) -> String {
    output::safe_text(value, false)
}

pub(crate) fn render(
    writer: &mut dyn Write,
    receipt: &RelationshipCandidateScanResponse,
) -> io::Result<()> {
    let policy = &receipt.policy;
    let limits = &receipt.limits;
    if receipt.dry_run {
        writeln!(
            writer,
            "Candidate scan (dry run): no proposals or accepted edges were changed."
        )?;
    } else {
        writeln!(
            writer,
            "Candidate scan recorded {} pending proposal(s); accepted edges were not changed.",
            receipt.proposals_created
        )?;
    }
    writeln!(
        writer,
        "  Limits: max_candidates {}, max_endpoints {}, deadline {} ms",
        limits.max_candidates, limits.max_endpoints, limits.deadline_ms
    )?;
    writeln!(
        writer,
        "  Policy: similarity floor {:.3}, {} neighbour(s)/endpoint, at most {} candidate(s)/endpoint, entities in more than {} notes are generic ({} v{})",
        policy.similarity_floor,
        policy.neighbors_per_endpoint,
        policy.max_per_endpoint,
        policy.max_entity_note_frequency,
        text(&policy.generator),
        text(&policy.generator_version),
    )?;
    let scope = match &receipt.scope {
        RelationshipCandidateScope::Notes { ids } => format!("{} note(s)", ids.len()),
        RelationshipCandidateScope::Sources { ids } => format!("{} source(s)", ids.len()),
        RelationshipCandidateScope::Orphans {} => "orphans".into(),
    };
    writeln!(
        writer,
        "  Scope: {scope}; examined {} endpoint(s) in {} ms; stopped: {}",
        receipt.endpoints_examined,
        receipt.elapsed_ms,
        serde_json::to_value(receipt.stop_reason)
            .ok()
            .and_then(|value| value.as_str().map(str::to_owned))
            .unwrap_or_default()
    )?;
    if let Some(next) = &receipt.next_after_note_id {
        writeln!(writer, "  Resume with: --after {}", text(next))?;
    }
    let rejected = &receipt.rejections;
    writeln!(
        writer,
        "  Nominated {} pair(s); selected {}. Rejected: same origin {}, duplicate {}, no second signal {}, generic entities only {}, connected {}, already proposed {}, prior decision {}, endpoint cap {}, over budget {}",
        receipt.nominations,
        receipt.candidates.len(),
        rejected.same_origin,
        rejected.duplicate_content,
        rejected.no_second_signal,
        rejected.generic_entities_only,
        rejected.already_connected,
        rejected.already_proposed,
        rejected.prior_decision,
        rejected.endpoint_cap,
        rejected.over_budget,
    )?;
    if receipt.candidates.is_empty() {
        writeln!(writer, "  No candidate met the policy (abstained).")?;
    }
    for (index, candidate) in receipt.candidates.iter().enumerate() {
        let signals = candidate
            .signals
            .iter()
            .filter_map(|signal| serde_json::to_value(signal).ok())
            .filter_map(|value| value.as_str().map(str::to_owned))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            writer,
            "{:>3}. {} <-> {}  cosine {:.3}  [{}]",
            index + 1,
            text(&candidate.from.id),
            text(&candidate.to.id),
            candidate.similarity,
            signals
        )?;
        if !candidate.shared_entities.is_empty() {
            let entities = candidate
                .shared_entities
                .iter()
                .map(|entity| format!("{} ({} notes)", text(&entity.name), entity.note_frequency))
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(writer, "     shared entities: {entities}")?;
        }
        if let Some(reference) = &candidate.cross_reference {
            let label = |value: Value| value.as_str().map(str::to_owned).unwrap_or_default();
            writeln!(
                writer,
                "     cross-reference: {} ({})",
                serde_json::to_value(reference.kind)
                    .map(label)
                    .unwrap_or_default(),
                serde_json::to_value(reference.direction)
                    .map(label)
                    .unwrap_or_default()
            )?;
        }
        if let Some(proposal) = &candidate.proposal {
            writeln!(
                writer,
                "     proposal: {} ({}); review with: garden review --id {}",
                text(&proposal.id),
                text(&proposal.status),
                text(&proposal.id)
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn parse(args: &[&str]) -> Result<crate::cli::Cli, clap::Error> {
        crate::cli::Cli::try_parse_from(args)
    }

    fn scan(cli: crate::cli::Cli) -> GardenCommand {
        match cli.command {
            crate::cli::Commands::Garden { command } => command,
            _ => panic!("expected garden"),
        }
    }

    #[test]
    fn legacy_scan_is_unchanged_and_candidate_flags_require_max_candidates() {
        let legacy = scan(parse(&["graphrag", "garden", "scan", "--dry-run"]).unwrap());
        assert!(request(&legacy).is_none());
        for flag in [
            &["--max-endpoints", "5"][..],
            &["--deadline-ms", "500"],
            &["--note", "note:alpha"],
            &["--source", "source:alpha"],
            &["--counterpart-source", "source:beta"],
            &["--after", "note:alpha"],
            &["--min-similarity", "0.9"],
            &["--max-per-endpoint", "1"],
            &["--format", "json"],
        ] {
            let mut args = vec!["graphrag", "garden", "scan"];
            args.extend_from_slice(flag);
            assert!(parse(&args).is_err(), "{flag:?} needs --max-candidates");
        }
        assert!(parse(&[
            "graphrag",
            "garden",
            "scan",
            "--max-candidates",
            "1",
            "--note",
            "note:a",
            "--source",
            "source:b"
        ])
        .is_err());
    }

    #[test]
    fn candidate_requests_send_every_budget_explicitly() {
        let command = scan(
            parse(&[
                "graphrag",
                "garden",
                "scan",
                "--max-candidates",
                "1",
                "--source",
                "source:alpha",
                "--counterpart-source",
                "source:beta",
                "--format",
                "json",
            ])
            .unwrap(),
        );
        let (sent, format) = request(&command).unwrap();
        assert_eq!(format, OutputFormat::Json);
        let wire = serde_json::to_value(&sent).unwrap();
        assert_eq!(
            wire,
            serde_json::json!({
                "scope": {"kind": "sources", "ids": ["source:alpha"]},
                "counterpart_source_ids": ["source:beta"],
                "after_note_id": null,
                "max_candidates": 1,
                "max_endpoints": DEFAULT_MAX_ENDPOINTS,
                "deadline_ms": DEFAULT_DEADLINE_MS,
                "min_similarity": null,
                "max_per_endpoint": null,
                "dry_run": false
            })
        );
        let orphans =
            scan(parse(&["graphrag", "garden", "scan", "--max-candidates", "3"]).unwrap());
        assert_eq!(
            request(&orphans).unwrap().0.scope,
            RelationshipCandidateScope::Orphans {}
        );
    }

    fn receipt(request: &RelationshipCandidateScanRequest) -> RelationshipCandidateScanResponse {
        serde_json::from_value(serde_json::json!({
            "contract": 1,
            "actor": "mcp:fixture",
            "dry_run": request.dry_run,
            "limits": {
                "max_candidates": request.max_candidates,
                "max_endpoints": request.max_endpoints,
                "deadline_ms": request.deadline_ms
            },
            "policy": {
                "relationship": "related_to",
                "generator": "gardener-candidate",
                "generator_version": "1",
                "similarity_floor": 0.8,
                "neighbors_per_endpoint": 8,
                "max_per_endpoint": 2,
                "max_entity_note_frequency": 10
            },
            "scope": request.scope,
            "counterpart_source_ids": request.counterpart_source_ids,
            "after_note_id": request.after_note_id,
            "stop_reason": "completed",
            "endpoints_examined": 1,
            "scope_exhausted": true,
            "next_after_note_id": null,
            "nominations": 0,
            "rejections": {
                "same_origin": 0, "duplicate_content": 0, "no_second_signal": 0,
                "generic_entities_only": 0, "already_connected": 0, "already_proposed": 0,
                "prior_decision": 0, "endpoint_cap": 0, "over_budget": 0
            },
            "unevaluated_names": 0,
            "elapsed_ms": 3,
            "embedding_model": null,
            "proposals_created": 0,
            "candidates": []
        }))
        .unwrap()
    }

    #[test]
    fn receipts_must_echo_the_requested_budgets() {
        let command = scan(
            parse(&[
                "graphrag",
                "garden",
                "scan",
                "--max-candidates",
                "1",
                "--dry-run",
            ])
            .unwrap(),
        );
        let (sent, _) = request(&command).unwrap();
        let valid = receipt(&sent);
        verify_receipt(&sent, &valid).unwrap();
        let mut rendered = Vec::new();
        render(&mut rendered, &valid).unwrap();
        let rendered = String::from_utf8(rendered).unwrap();
        assert!(rendered.contains("max_candidates 1, max_endpoints 25, deadline 10000 ms"));
        assert!(rendered.contains("abstained"));
        let mutations: [fn(&mut RelationshipCandidateScanResponse); 6] = [
            |r| r.limits.max_candidates = 50,
            |r| r.limits.max_endpoints = 1,
            |r| r.limits.deadline_ms = 60_000,
            |r| r.dry_run = false,
            |r| {
                r.scope = RelationshipCandidateScope::Notes {
                    ids: vec!["note:a".into()],
                }
            },
            |r| r.endpoints_examined = 26,
        ];
        for (index, mutate) in mutations.into_iter().enumerate() {
            let mut changed = valid.clone();
            mutate(&mut changed);
            assert!(verify_receipt(&sent, &changed).is_err(), "mutation {index}");
        }
    }
}
