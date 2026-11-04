//! Validation, policy resolution and receipts for bounded candidate scans.
use crate::*;
use graphrag_agents::gardener::{
    CandidatePolicy, CandidateScanReport, CandidateScanRequest, CandidateScope, CandidateSignal,
    CandidateStopReason, CrossReferenceDirection, CrossReferenceKind, TextualCueKind,
    CANDIDATE_CONTRACT, CANDIDATE_GENERATOR,
};
use graphrag_agents::GardenerAgent;
use graphrag_core::{record_id_to_string, ProposedEdgeStatus};
use graphrag_db::Repository;
use std::collections::BTreeSet;
use std::time::Duration;
use surrealdb::types::RecordId;

fn canonical_ids(ids: &[String], table: &str, max: usize) -> ApplicationResult<Vec<RecordId>> {
    if ids.len() > max {
        return Err(ApplicationError::Validation(format!(
            "At most {max} canonical {table}: IDs are allowed."
        )));
    }
    let mut seen = BTreeSet::new();
    ids.iter()
        .map(|id| {
            let parsed = canonical_id(id, table)?;
            if !seen.insert(id.as_str()) {
                return Err(ApplicationError::Validation(format!(
                    "Duplicate {table}: ID in a candidate scan."
                )));
            }
            Ok(parsed)
        })
        .collect()
}

fn canonical_id(id: &str, table: &str) -> ApplicationResult<RecordId> {
    let invalid = || {
        ApplicationError::Validation(format!(
            "Use canonical {table}:ID values without controls or surrounding whitespace."
        ))
    };
    if id.len() > 512 || id.trim() != id || id.chars().any(char::is_control) {
        return Err(invalid());
    }
    let parsed = graphrag_db::parse_record_id(id, Some(table)).map_err(|_| invalid())?;
    if record_id_to_string(&parsed) != id {
        return Err(invalid());
    }
    Ok(parsed)
}

/// Reject anything outside the advertised bounds; nothing is defaulted.
pub(crate) fn validate_candidate_scan(
    request: &RelationshipCandidateScanRequest,
) -> ApplicationResult<CandidateScanRequest> {
    if !(1..=MAX_SCAN_CANDIDATES).contains(&request.max_candidates)
        || !(1..=MAX_SCAN_ENDPOINTS).contains(&request.max_endpoints)
        || !(MIN_SCAN_DEADLINE_MS..=MAX_SCAN_DEADLINE_MS).contains(&request.deadline_ms)
        || request
            .min_similarity
            .is_some_and(|floor| !floor.is_finite() || !(0.0..=1.0).contains(&floor))
        || request
            .max_per_endpoint
            .is_some_and(|cap| !(1..=MAX_SCAN_PER_ENDPOINT).contains(&cap))
    {
        return Err(ApplicationError::Validation(format!(
            "Candidate scans require max_candidates 1–{MAX_SCAN_CANDIDATES}, max_endpoints 1–{MAX_SCAN_ENDPOINTS}, deadline_ms {MIN_SCAN_DEADLINE_MS}–{MAX_SCAN_DEADLINE_MS}, min_similarity in [0, 1] and max_per_endpoint 1–{MAX_SCAN_PER_ENDPOINT}."
        )));
    }
    let scope = match &request.scope {
        RelationshipCandidateScope::Notes { ids } | RelationshipCandidateScope::Sources { ids }
            if ids.is_empty() =>
        {
            return Err(ApplicationError::Validation(
                "A notes or sources scope needs at least one canonical ID.".into(),
            ));
        }
        RelationshipCandidateScope::Notes { ids } => {
            CandidateScope::Notes(canonical_ids(ids, "note", MAX_SCAN_SCOPE_IDS)?)
        }
        RelationshipCandidateScope::Sources { ids } => {
            CandidateScope::Sources(canonical_ids(ids, "source", MAX_SCAN_SCOPE_IDS)?)
        }
        RelationshipCandidateScope::Orphans {} => CandidateScope::Orphans,
    };
    let counterparts = canonical_ids(
        &request.counterpart_source_ids,
        "source",
        MAX_SCAN_SCOPE_IDS,
    )?;
    let after = request
        .after_note_id
        .as_deref()
        .map(|id| canonical_id(id, "note"))
        .transpose()?;
    Ok(CandidateScanRequest {
        scope,
        counterpart_sources: (!counterparts.is_empty()).then_some(counterparts),
        after,
        max_candidates: request.max_candidates,
        max_endpoints: request.max_endpoints,
        deadline: Duration::from_millis(request.deadline_ms),
        min_similarity: request.min_similarity,
        max_per_endpoint: request.max_per_endpoint,
        dry_run: request.dry_run,
        requested_by: String::new(),
    })
}

fn stop_reason(reason: CandidateStopReason) -> RelationshipCandidateStopReason {
    match reason {
        CandidateStopReason::Completed => RelationshipCandidateStopReason::Completed,
        CandidateStopReason::EndpointBudget => RelationshipCandidateStopReason::EndpointBudget,
        CandidateStopReason::CandidateBudget => RelationshipCandidateStopReason::CandidateBudget,
        CandidateStopReason::Deadline => RelationshipCandidateStopReason::Deadline,
    }
}

fn signal(signal: CandidateSignal) -> RelationshipCandidateSignal {
    match signal {
        CandidateSignal::ExplicitCrossReference => {
            RelationshipCandidateSignal::ExplicitCrossReference
        }
        CandidateSignal::TextualCue => RelationshipCandidateSignal::TextualCue,
        CandidateSignal::SharedSpecificEntity => RelationshipCandidateSignal::SharedSpecificEntity,
    }
}

fn cue_kind(kind: TextualCueKind) -> RelationshipTextualCueKind {
    match kind {
        TextualCueKind::RelativeDate => RelationshipTextualCueKind::RelativeDate,
        TextualCueKind::DatedReference => RelationshipTextualCueKind::DatedReference,
        TextualCueKind::AnsweredQuestion => RelationshipTextualCueKind::AnsweredQuestion,
    }
}

fn reference_kind(kind: CrossReferenceKind) -> RelationshipCrossReferenceKind {
    match kind {
        CrossReferenceKind::RecordId => RelationshipCrossReferenceKind::RecordId,
        CrossReferenceKind::SourceUri => RelationshipCrossReferenceKind::SourceUri,
        CrossReferenceKind::Wikilink => RelationshipCrossReferenceKind::Wikilink,
        CrossReferenceKind::FileName => RelationshipCrossReferenceKind::FileName,
    }
}

fn reference_direction(direction: CrossReferenceDirection) -> RelationshipCrossReferenceDirection {
    match direction {
        CrossReferenceDirection::FromMentionsTo => {
            RelationshipCrossReferenceDirection::FromMentionsTo
        }
        CrossReferenceDirection::ToMentionsFrom => {
            RelationshipCrossReferenceDirection::ToMentionsFrom
        }
        CrossReferenceDirection::Both => RelationshipCrossReferenceDirection::Both,
    }
}

fn receipt(
    actor: String,
    request: RelationshipCandidateScanRequest,
    report: CandidateScanReport,
) -> RelationshipCandidateScanResponse {
    let rejections = report.rejections;
    let dry_run_cut = report
        .dry_run_cut()
        .map(|cut| RelationshipCandidateDryRunCut {
            candidates_cut: cut,
            note: dry_run_cut_note(cut),
        });
    let candidates = report
        .candidates
        .into_iter()
        .map(|candidate| RelationshipCandidateOutput {
            from: RelationshipCandidateEndpoint {
                id: record_id_to_string(&candidate.from),
                source_id: candidate.from_source.as_ref().map(record_id_to_string),
            },
            to: RelationshipCandidateEndpoint {
                id: record_id_to_string(&candidate.to),
                source_id: candidate.to_source.as_ref().map(record_id_to_string),
            },
            relationship: "related_to".into(),
            similarity: candidate.similarity,
            signals: candidate.signals.into_iter().map(signal).collect(),
            shared_entities: candidate
                .shared_entities
                .into_iter()
                .map(|entity| RelationshipCandidateEntity {
                    name: entity.name,
                    note_frequency: entity.note_frequency,
                })
                .collect(),
            cross_reference: candidate.cross_reference.map(|reference| {
                RelationshipCandidateCrossReference {
                    kind: reference_kind(reference.kind),
                    direction: reference_direction(reference.direction),
                }
            }),
            textual_cue: candidate
                .textual_cue
                .map(|cue| RelationshipCandidateTextualCue {
                    kind: cue_kind(cue.kind),
                    direction: reference_direction(cue.direction),
                }),
            proposal: candidate.proposal.and_then(|proposal| {
                let status = proposal.status;
                proposal.id.map(|id| RelationshipCandidateProposal {
                    id: record_id_to_string(&id),
                    status: status.to_string(),
                })
            }),
        })
        .collect::<Vec<_>>();
    let pending = ProposedEdgeStatus::Pending.to_string();
    RelationshipCandidateScanResponse {
        contract: RELATIONSHIP_CANDIDATE_SCAN_CONTRACT,
        actor,
        dry_run: report.dry_run,
        limits: RelationshipCandidateScanLimits {
            max_candidates: request.max_candidates,
            max_endpoints: request.max_endpoints,
            deadline_ms: request.deadline_ms,
        },
        policy: RelationshipCandidatePolicy {
            relationship: "related_to".into(),
            generator: CANDIDATE_GENERATOR.into(),
            generator_version: CANDIDATE_CONTRACT.to_string(),
            similarity_floor: report.policy.similarity_floor,
            neighbors_per_endpoint: report.policy.neighbors_per_endpoint,
            max_per_endpoint: report.policy.max_per_endpoint,
            max_entity_note_frequency: report.policy.max_entity_note_frequency,
            near_duplicate_ceiling: report.policy.near_duplicate_ceiling,
            max_shared_span: report.policy.max_shared_span,
            max_machine_line_share: report.policy.max_machine_line_share,
            max_lineage_documents: report.policy.max_lineage_documents,
            min_specific_entities: report.policy.min_specific_entities,
            textual_cues: report.policy.textual_cues,
            similarity_batch_anchors: report.policy.similarity_batch_anchors,
        },
        scope: request.scope,
        counterpart_source_ids: request.counterpart_source_ids,
        after_note_id: request.after_note_id,
        stop_reason: stop_reason(report.stop_reason),
        endpoints_examined: report.endpoints_examined,
        machine_dump_anchors: report.machine_dump_anchors,
        scope_exhausted: report.scope_exhausted,
        next_after_note_id: report.next_after.as_ref().map(record_id_to_string),
        dry_run_cut,
        nominations: report.nominations,
        rejections: RelationshipCandidateRejections {
            same_origin: rejections.same_origin,
            duplicate_content: rejections.duplicate_content,
            no_second_signal: rejections.no_second_signal,
            generic_entities_only: rejections.generic_entities_only,
            already_connected: rejections.already_connected,
            already_proposed: rejections.already_proposed,
            prior_decision: rejections.prior_decision,
            endpoint_cap: rejections.endpoint_cap,
            over_budget: rejections.over_budget,
            vanished: rejections.vanished,
            same_document: rejections.same_document,
            machine_dump: rejections.machine_dump,
            near_duplicate: rejections.near_duplicate,
            lineage_template: rejections.lineage_template,
            insufficient_specific_entities: rejections.insufficient_specific_entities,
        },
        unevaluated_names: report.unevaluated_names,
        elapsed_ms: u64::try_from(report.elapsed.as_millis()).unwrap_or(u64::MAX),
        embedding_model: report.embedding_model,
        proposals_created: candidates
            .iter()
            .filter(|candidate| {
                candidate
                    .proposal
                    .as_ref()
                    .is_some_and(|proposal| proposal.status == pending)
            })
            .count(),
        candidates,
    }
}

/// Guidance for a dry run that cut candidates: it has no continuation.
pub(crate) fn dry_run_cut_note(cut: usize) -> String {
    format!(
        "This dry run cut {cut} candidate(s) at max_candidates and recorded nothing, so there is no continuation cursor: repeating it returns the same candidates. Narrow the scope, raise max_candidates (at most {MAX_SCAN_CANDIDATES}), or run with dry_run=false so the recorded pending proposals let later scans progress."
    )
}

/// Run one bounded candidate scan against an owned repository. Shared by
/// the service tool and the host-local CLI so both enforce identical
/// validation, policy tightening and receipts. `actor` is recorded in every
/// persisted proposal's candidate features.
pub async fn scan_relationship_candidates(
    repo: &Repository,
    policy: &CandidatePolicy,
    actor: &str,
    request: RelationshipCandidateScanRequest,
) -> ApplicationResult<RelationshipCandidateScanResponse> {
    let mut scan = validate_candidate_scan(&request)?;
    scan.requested_by = actor.to_owned();
    let report = GardenerAgent::new(repo.clone())
        .scan_candidates(policy, scan)
        .await?;
    Ok(receipt(actor.to_owned(), request, report))
}

impl EmbeddedApplication {
    pub(crate) async fn scan_relationship_candidates_impl(
        &self,
        caller: CallerIdentity,
        request: RelationshipCandidateScanRequest,
    ) -> ApplicationResult<RelationshipCandidateScanResponse> {
        let instance = &caller.instance_id;
        if instance.is_empty()
            || instance.len() > 256
            || instance.trim() != instance
            || instance.chars().any(char::is_control)
        {
            return Err(ApplicationError::Validation(
                "A trusted caller identity is required.".into(),
            ));
        }
        scan_relationship_candidates(
            &self.repo,
            &self.candidate_policy,
            &format!("mcp:{instance}"),
            request,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request() -> serde_json::Value {
        json!({
            "scope": {"kind": "notes", "ids": ["note:alpha"]},
            "max_candidates": 3,
            "max_endpoints": 10,
            "deadline_ms": 5000,
            "dry_run": true
        })
    }

    fn parse(value: serde_json::Value) -> Result<RelationshipCandidateScanRequest, String> {
        serde_json::from_value(value).map_err(|error| error.to_string())
    }

    #[test]
    fn budgets_and_dry_run_are_required_and_unknown_fields_rejected() {
        parse(request()).unwrap();
        for field in [
            "scope",
            "max_candidates",
            "max_endpoints",
            "deadline_ms",
            "dry_run",
        ] {
            let mut value = request();
            value.as_object_mut().unwrap().remove(field);
            assert!(parse(value).is_err(), "{field} must be required");
        }
        let mut injected = request();
        injected["accept"] = json!(true);
        assert!(parse(injected).is_err());
        let mut scope = request();
        scope["scope"]["limit"] = json!(5);
        assert!(parse(scope).is_err());
        let orphans = json!({"kind": "orphans", "ids": ["note:alpha"]});
        let mut value = request();
        value["scope"] = orphans;
        assert!(parse(value).is_err(), "orphans takes no IDs");
        let mut value = request();
        value["scope"] = json!({"kind": "orphans"});
        assert_eq!(
            parse(value).unwrap().scope,
            RelationshipCandidateScope::Orphans {}
        );
    }

    #[test]
    fn budgets_policy_overrides_and_ids_are_bounded() {
        let valid = parse(request()).unwrap();
        let resolved = validate_candidate_scan(&valid).unwrap();
        assert_eq!(resolved.max_candidates, 3);
        assert_eq!(resolved.deadline, Duration::from_millis(5000));
        let cases: Vec<fn(&mut RelationshipCandidateScanRequest)> = vec![
            |r| r.max_candidates = 0,
            |r| r.max_candidates = MAX_SCAN_CANDIDATES + 1,
            |r| r.max_endpoints = 0,
            |r| r.max_endpoints = MAX_SCAN_ENDPOINTS + 1,
            |r| r.deadline_ms = MIN_SCAN_DEADLINE_MS - 1,
            |r| r.deadline_ms = MAX_SCAN_DEADLINE_MS + 1,
            |r| r.min_similarity = Some(1.5),
            |r| r.min_similarity = Some(f32::NAN),
            |r| r.max_per_endpoint = Some(0),
            |r| r.scope = RelationshipCandidateScope::Notes { ids: vec![] },
            |r| {
                r.scope = RelationshipCandidateScope::Notes {
                    ids: vec!["alpha".into()],
                }
            },
            |r| {
                r.scope = RelationshipCandidateScope::Notes {
                    ids: vec!["source:alpha".into()],
                }
            },
            |r| {
                r.scope = RelationshipCandidateScope::Notes {
                    ids: vec![" note:alpha".into()],
                }
            },
            |r| {
                r.scope = RelationshipCandidateScope::Notes {
                    ids: vec!["note:alpha".into(), "note:alpha".into()],
                }
            },
            |r| {
                r.scope = RelationshipCandidateScope::Sources {
                    ids: vec!["note:alpha".into()],
                }
            },
            |r| {
                r.scope = RelationshipCandidateScope::Notes {
                    ids: (0..=MAX_SCAN_SCOPE_IDS)
                        .map(|i| format!("note:n{i}"))
                        .collect(),
                }
            },
            |r| r.counterpart_source_ids = vec!["note:alpha".into()],
            |r| r.after_note_id = Some("source:alpha".into()),
            |r| r.after_note_id = Some("note:alpha\n".into()),
        ];
        for (index, mutate) in cases.into_iter().enumerate() {
            let mut invalid = valid.clone();
            mutate(&mut invalid);
            assert!(
                matches!(
                    validate_candidate_scan(&invalid),
                    Err(ApplicationError::Validation(_))
                ),
                "case {index} must be rejected"
            );
        }
        let mut scoped = valid;
        scoped.scope = RelationshipCandidateScope::Sources {
            ids: vec!["source:alpha".into()],
        };
        scoped.counterpart_source_ids = vec!["source:beta".into()];
        scoped.after_note_id = Some("note:alpha".into());
        let resolved = validate_candidate_scan(&scoped).unwrap();
        assert!(matches!(resolved.scope, CandidateScope::Sources(ref ids) if ids.len() == 1));
        assert_eq!(resolved.counterpart_sources.map(|ids| ids.len()), Some(1));
        assert!(resolved.after.is_some());
    }
}
