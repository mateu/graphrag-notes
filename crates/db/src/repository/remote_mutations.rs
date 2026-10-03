//! Atomic authenticated mutations. Provider work happens before this gate;
//! effect and original successful response always share one transaction.
use super::*;
use sha2::{Digest, Sha256};
use tokio::sync::OwnedMutexGuard;

#[cfg(test)]
#[path = "remote_mutations_tests.rs"]
mod tests;

pub struct MutationGuard {
    identity: Arc<Mutex<()>>,
    _guard: OwnedMutexGuard<()>,
}

#[derive(Debug, Clone)]
pub struct RemoteMutationInput {
    pub instance_id: String,
    pub request_id: String,
    pub operation: String,
    pub payload_fingerprint: String,
    pub payload: serde_json::Value,
    pub result: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct MutationNoteSnapshot {
    pub note: Note,
    pub inspection: RecordInspection,
    pub manual: bool,
}

#[derive(Debug, Clone)]
pub enum RemoteMutationEffect {
    Edit {
        expected: Box<Note>,
        replacement: Box<Note>,
        entities: Option<Vec<Entity>>,
    },
    Delete {
        expected: Box<Note>,
    },
    Decision {
        expected: Box<ProposedEdge>,
        endpoints: Vec<Note>,
        action: String,
        reason: Option<String>,
    },
}

#[derive(Debug, Deserialize, SurrealValue)]
struct MutationReceiptRow {
    operation: String,
    payload_fingerprint: String,
    payload: serde_json::Value,
    result: serde_json::Value,
}

fn conflict() -> DbError {
    DbError::MutationRevisionConflict("The record or request changed. Refresh its snapshot and retain the draft; use a new request ID only for a new decision.".into())
}

fn validate_input(input: &RemoteMutationInput) -> Result<()> {
    for value in [&input.instance_id, &input.request_id] {
        if value.is_empty()
            || value.len() > 256
            || value.chars().count() > 128
            || value.trim() != value
            || value.chars().any(char::is_control)
        {
            return Err(DbError::InvalidMutationRequest(
                "invalid bounded request identity".into(),
            ));
        }
    }
    if !matches!(
        input.operation.as_str(),
        "edit" | "delete" | "accept" | "reject" | "undo"
    ) || input.payload_fingerprint.len() != 64
        || !input
            .payload_fingerprint
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        || !input.payload.is_object()
        || !input.result.is_object()
        || serde_json::to_vec(&input.result)
            .map_err(|e| DbError::InvalidMutationRequest(e.to_string()))?
            .len()
            > 16 * 1024
    {
        return Err(DbError::InvalidMutationRequest(
            "invalid bounded mutation payload/result".into(),
        ));
    }
    Ok(())
}

impl Repository {
    /// Caller must hold this gate from public revision validation through commit.
    /// Never perform provider work or call a separately locking writer under it.
    pub async fn mutation_guard(&self) -> MutationGuard {
        MutationGuard {
            identity: self.proposal_acceptance_lock.clone(),
            _guard: self.proposal_acceptance_lock.clone().lock_owned().await,
        }
    }

    fn check_mutation_guard(&self, guard: &MutationGuard) -> Result<()> {
        if !Arc::ptr_eq(&guard.identity, &self.proposal_acceptance_lock) {
            return Err(DbError::InvalidMutationRequest(
                "mutation guard belongs to another database".into(),
            ));
        }
        Ok(())
    }

    pub async fn mutation_note_snapshot(
        &self,
        guard: &MutationGuard,
        id: &str,
    ) -> Result<MutationNoteSnapshot> {
        self.check_mutation_guard(guard)?;
        let inspection = self.inspect_record(id, 0).await?;
        if inspection.hit_type != "note" {
            return Err(DbError::InvalidMutationRequest(
                "only notes can be edited or deleted".into(),
            ));
        }
        let literal = normalize_note_id(id);
        let native = parse_record_id(id, Some("note"))?;
        let note = match self.db.select::<Option<Note>>(literal.clone()).await? {
            Some(note) => note,
            None => self
                .db
                .select::<Option<Note>>(native)
                .await?
                .ok_or_else(|| DbError::NotFound("note".into(), id.into()))?,
        };
        let note_id = note.id.clone().ok_or_else(conflict)?;
        let mut links_response = self.db.query("SELECT VALUE id FROM note_from_conversation WHERE in = $id LIMIT 1; SELECT VALUE id FROM note_from_message WHERE in = $id LIMIT 1;")
            .bind(("id", note_id)).await?;
        let mut links: Vec<RecordId> = links_response.take(0)?;
        links.extend(links_response.take::<Vec<RecordId>>(1)?);
        let manual = note.source_generation.is_none() && links.is_empty();
        Ok(MutationNoteSnapshot {
            note,
            inspection,
            manual,
        })
    }

    pub async fn mutation_proposal_snapshot(
        &self,
        guard: &MutationGuard,
        id: &str,
    ) -> Result<ProposedEdge> {
        self.check_mutation_guard(guard)?;
        let native = parse_record_id(id, Some("proposed_edge"))?;
        let key = id.split_once(':').ok_or_else(conflict)?.1;
        let literal = RecordId::new("proposed_edge", key);
        let string = self.get_edge_proposal(&literal).await?;
        let typed = if native != literal {
            self.get_edge_proposal(&native).await?
        } else {
            None
        };
        if string.is_some() && typed.is_some() {
            return Err(DbError::InvalidMutationRequest(
                "ambiguous proposal key; use distinct string/native keys".into(),
            ));
        }
        string
            .or(typed)
            .ok_or_else(|| DbError::NotFound("proposal".into(), id.into()))
    }

    async fn mutation_receipt_row(
        &self,
        instance: &str,
        request: &str,
    ) -> Result<Option<MutationReceiptRow>> {
        Ok(self.db.query("SELECT operation,payload_fingerprint,payload,result FROM remote_mutation_receipt WHERE instance_id=$instance AND request_id=$request LIMIT 1")
            .bind(("instance", instance.to_owned())).bind(("request", request.to_owned())).await?.take(0)?)
    }

    pub async fn find_remote_mutation_receipt(
        &self,
        input: &RemoteMutationInput,
    ) -> Result<Option<RemoteCaptureReceipt>> {
        validate_input(input)?;
        match self
            .mutation_receipt_row(&input.instance_id, &input.request_id)
            .await?
        {
            Some(row)
                if row.operation == input.operation
                    && row.payload_fingerprint == input.payload_fingerprint
                    && row.payload == input.payload =>
            {
                Ok(Some(RemoteCaptureReceipt {
                    result: row.result,
                    replayed: true,
                }))
            }
            Some(_) => Err(conflict()),
            None => Ok(None),
        }
    }

    pub async fn mutation_delete_preview(
        &self,
        guard: &MutationGuard,
        note: &Note,
    ) -> Result<SourceDeleteSummary> {
        self.check_mutation_guard(guard)?;
        self.delete_summary_for_notes(&[note.id.clone().ok_or_else(conflict)?])
            .await
    }

    pub async fn apply_remote_mutation(
        &self,
        guard: &MutationGuard,
        input: RemoteMutationInput,
        effect: RemoteMutationEffect,
    ) -> Result<RemoteCaptureReceipt> {
        self.check_mutation_guard(guard)?;
        validate_input(&input)?;
        if let Some(receipt) = self.find_remote_mutation_receipt(&input).await? {
            return Ok(receipt);
        }
        let key = format!(
            "{:x}",
            Sha256::digest(
                serde_json::to_vec(&(&input.instance_id, &input.request_id))
                    .map_err(|e| DbError::InvalidMutationRequest(e.to_string()))?
            )
        );
        let mut expected_note = None;
        let mut replacement = None;
        let mut entities = Vec::new();
        let mut expected_proposal = None;
        let mut endpoint_notes = Vec::new();
        let mut edge_id = None;
        let mut reason = None;
        let target;
        let effects;
        match effect {
            RemoteMutationEffect::Edit {
                expected,
                replacement: note,
                entities: replacement_entities,
            } => {
                if input.operation != "edit" {
                    return Err(conflict());
                }
                let mut note = *note;
                // Match local updates: retain intentional aliases, while
                // rebuilding body/heading-derived text for changed content.
                note.search_content = Some(search_content_for_note_update(&expected, &note));
                target = expected.id.clone().ok_or_else(conflict)?;
                let mut sql = super::notes::editor_snapshot_guard(true);
                if let Some(new_entities) = replacement_entities {
                    entities = new_entities;
                    sql.push_str(super::notes::replacement_entities_transaction());
                    sql.push_str("DELETE mentions WHERE in=$target; FOR $entity_id IN $entity_ids { CREATE mentions SET in=$target,out=$entity_id; }; ");
                }
                sql.push_str("UPDATE $target SET content=$replacement.content,title=$replacement.title,tags=$replacement.tags,embedding=$replacement_embedding,search_content=$replacement.search_content,updated_at=$replacement.updated_at; ");
                expected_note = Some(*expected);
                replacement = Some(note);
                effects = sql;
            }
            RemoteMutationEffect::Delete { expected } => {
                if input.operation != "delete" {
                    return Err(conflict());
                }
                target = expected.id.clone().ok_or_else(conflict)?;
                effects = format!("{} UPDATE proposed_edge SET status='superseded',superseded_at=time::now(),supersession_reason='proposal endpoint removed by remote note deletion',resulting_edge_id=NONE,updated_at=time::now() WHERE status IN ['pending','accepting','accepted'] AND (in=$target OR out=$target); DELETE supports WHERE in=$target OR out=$target; DELETE contradicts WHERE in=$target OR out=$target; DELETE derived_from WHERE in=$target OR out=$target; DELETE related_to WHERE in=$target OR out=$target; DELETE mentions WHERE in=$target; DELETE note_from_message WHERE in=$target; DELETE note_from_conversation WHERE in=$target; DELETE $target; ",super::notes::editor_snapshot_guard(true));
                expected_note = Some(*expected);
            }
            RemoteMutationEffect::Decision {
                expected,
                endpoints,
                action,
                reason: action_reason,
            } => {
                if input.operation != action {
                    return Err(conflict());
                }
                target = expected.id.clone().ok_or_else(conflict)?;
                let mut sql = proposal_guard();
                for (index, _) in endpoints.iter().enumerate() {
                    sql.push_str(
                        &super::notes::editor_snapshot_guard(false)
                            .replace("$editor_source", &format!("$endpoint_{index}"))
                            .replace("$editor_expected", &format!("$expected_endpoint_{index}"))
                            .replace("$editor_matches", &format!("$endpoint_matches_{index}")),
                    );
                }
                match action.as_str() {
                    "accept" if expected.status == ProposedEdgeStatus::Pending && endpoints.len() == 2 => {
                        let table = super::graph::note_edge_table(&expected.edge_type)?;
                        if expected.from_id == expected.to_id { return Err(DbError::InvalidMutationRequest("proposal cannot connect a note to itself".into())); }
                        let id = RecordId::new(table, format!("remote_{}", Sha256::digest(serde_json::to_vec(&target).map_err(|e|DbError::QueryFailed(e.to_string()))?).iter().map(|b|format!("{b:02x}")).collect::<String>()));
                        edge_id = Some(id);
                        sql.push_str(&format!("IF array::len((SELECT VALUE id FROM {table} WHERE dedupe_key=$dedupe_key LIMIT 1)) != 0 {{ THROW 'remote-mutation-revision-conflict'; }}; CREATE $edge SET in=$expected_proposal.from_id,out=$expected_proposal.to_id,confidence=$expected_proposal.confidence,reason=$expected_proposal.reason,provenance='manual',proposal_id=$target,is_manual=true,dedupe_key=$dedupe_key,created_at=time::now(); UPDATE $target SET status='accepted',reviewed_at=time::now(),reviewer=$actor,action_reason=$reason,acceptance_is_manual=true,resulting_edge_id=$edge,updated_at=time::now(); "));
                    }
                    "reject" if expected.status == ProposedEdgeStatus::Pending => sql.push_str("UPDATE $target SET status='rejected',reviewed_at=time::now(),reviewer=$actor,action_reason=$reason,updated_at=time::now(); "),
                    "undo" if expected.status == ProposedEdgeStatus::Accepted => {
                        let edge = expected.resulting_edge_id.clone().ok_or_else(conflict)?;
                        if !matches!(edge.table.as_str(),"supports"|"contradicts"|"derived_from"|"related_to") { return Err(conflict()); }
                        edge_id=Some(edge);
                        sql.push_str("IF array::len((SELECT VALUE id FROM $edge WHERE proposal_id=$target)) != 1 { THROW 'remote-mutation-revision-conflict'; }; DELETE $edge; UPDATE $target SET status='superseded',superseded_at=time::now(),supersession_reason=($reason ?? 'accepted edge undone by authenticated remote reviewer'),resulting_edge_id=NONE,updated_at=time::now(); ");
                    }
                    _ => return Err(conflict()),
                }
                expected_proposal = Some(*expected);
                endpoint_notes = endpoints;
                reason = action_reason;
                effects = sql;
            }
        }
        let receipt = RecordId::new("remote_mutation_receipt", key);
        let query=format!("BEGIN TRANSACTION; {effects} CREATE $receipt SET instance_id=$instance,request_id=$request,operation=$operation,target=$target,payload_fingerprint=$fingerprint,payload=$payload,result=$result,created_at=time::now(),updated_at=time::now(); COMMIT TRANSACTION;");
        let entity_names = entities
            .iter()
            .map(|e| e.canonical_name.clone())
            .collect::<Vec<_>>();
        let expected_status = expected_proposal.as_ref().map(|p| p.status.to_string());
        let expected_edge_type = expected_proposal.as_ref().map(|p| p.edge_type.to_string());
        let replacement_embedding = replacement
            .as_ref()
            .and_then(|note| (!note.embedding.is_empty()).then(|| note.embedding.clone()));
        let dedupe = expected_proposal
            .as_ref()
            .map(|p| super::graph::edge_dedupe_key(&p.from_id, &p.to_id, &p.edge_type));
        let mut query = self
            .db
            .query(query)
            .bind(("target", target.clone()))
            .bind(("editor_source", target))
            .bind(("editor_expected", expected_note))
            .bind(("replacement", replacement))
            .bind(("replacement_embedding", replacement_embedding))
            .bind(("replacement_entities", entities))
            .bind(("replacement_entity_names", entity_names))
            .bind(("expected_proposal", expected_proposal))
            .bind(("expected_status", expected_status))
            .bind(("expected_edge_type", expected_edge_type))
            .bind(("edge", edge_id))
            .bind(("dedupe_key", dedupe))
            .bind(("reason", reason))
            .bind(("actor", format!("mcp:{}", input.instance_id)))
            .bind(("instance", input.instance_id.clone()))
            .bind(("request", input.request_id.clone()))
            .bind(("operation", input.operation.clone()))
            .bind(("fingerprint", input.payload_fingerprint.clone()))
            .bind(("payload", input.payload.clone()))
            .bind(("result", input.result.clone()))
            .bind(("receipt", receipt));
        for (index, note) in endpoint_notes.into_iter().enumerate() {
            query = query
                .bind((format!("endpoint_{index}"), note.id.clone()))
                .bind((format!("expected_endpoint_{index}"), note));
        }
        let mut response = query.await?;
        let errors = response.take_errors();
        if !errors.is_empty() {
            if let Some(receipt) = self.find_remote_mutation_receipt(&input).await? {
                return Ok(receipt);
            }
            if errors.values().any(|e| {
                e.message().contains("revision-conflict")
                    || matches!(
                        e.details(),
                        surrealdb_types::ErrorDetails::Query(Some(
                            surrealdb_types::QueryError::TransactionConflict
                        ))
                    )
            }) {
                return Err(conflict());
            }
            return Err(DbError::QueryFailed(format!(
                "atomic remote mutation failed: {errors:?}"
            )));
        }
        let index = response
            .num_statements()
            .checked_sub(2)
            .ok_or_else(|| DbError::CreateFailed("mutation receipt".into()))?;
        let row: Option<MutationReceiptRow> = response.take(index)?;
        let row = row.ok_or_else(|| DbError::CreateFailed("mutation receipt".into()))?;
        Ok(RemoteCaptureReceipt {
            result: row.result,
            replayed: false,
        })
    }
}

fn proposal_guard() -> String {
    let fields = [
        "dedupe_key",
        "in",
        "out",
        "edge_type",
        "confidence",
        "reason",
        "generator",
        "generator_version",
        "model",
        "status",
        "created_at",
        "updated_at",
        "reviewed_at",
        "reviewer",
        "action_reason",
        "acceptance_is_manual",
        "resulting_edge_id",
        "superseded_at",
        "supersession_reason",
    ];
    let comparison = fields
        .iter()
        .map(|field| {
            if *field == "status" {
                return "status=$expected_status".to_owned();
            }
            if *field == "edge_type" {
                return "edge_type=$expected_edge_type".to_owned();
            }
            format!(
                "{field}=$expected_proposal.{}",
                match *field {
                    "in" => "from_id",
                    "out" => "to_id",
                    other => other,
                }
            )
        })
        .collect::<Vec<_>>()
        .join(" AND ");
    format!("IF array::len((SELECT VALUE id FROM proposed_edge WHERE id=$target AND {comparison} LIMIT 1)) != 1 {{ THROW 'remote-mutation-revision-conflict'; }}; ")
}
