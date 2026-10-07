//! Atomic, quote-grounded endpoint-proposal persistence.
use super::*;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone)]
pub struct RemoteEndpointProposalInput {
    pub mutation: RemoteMutationInput,
    pub from_id: String,
    pub from_revision: String,
    pub from_quote: String,
    pub to_id: String,
    pub to_revision: String,
    pub to_quote: String,
    pub edge_type: EdgeType,
    pub rationale: String,
}

#[derive(Debug, Deserialize, SurrealValue)]
struct ReceiptRow {
    operation: String,
    payload_fingerprint: String,
    payload: serde_json::Value,
    result: serde_json::Value,
}

fn conflict() -> DbError {
    DbError::MutationRevisionConflict("The endpoint proposal changed or a proposal for this pair already exists. Refresh both note snapshots; reuse the same request ID only for an identical retry.".into())
}

async fn receipt(repo: &Repository, input: &RemoteMutationInput) -> Result<Option<ReceiptRow>> {
    repo.db.query("SELECT operation,payload_fingerprint,payload,result FROM remote_mutation_receipt WHERE instance_id=$instance AND request_id=$request LIMIT 1")
        .bind(("instance", input.instance_id.clone())).bind(("request", input.request_id.clone())).await?.take::<Option<ReceiptRow>>(0).map_err(DbError::from)
}

/// Retire only this narrowly reviewed proposal family when an endpoint changes.
/// Independent manual edges and legacy gardener proposals are untouched.
pub(super) fn invalidate_reviewed_endpoint_sql(notes: &str) -> String {
    format!("LET $reviewed_ids = (SELECT VALUE id FROM proposed_edge WHERE generator = 'remote-reviewed-endpoint' AND status IN ['pending','accepting','accepted'] AND (in IN {notes} OR out IN {notes})); DELETE supports WHERE proposal_id IN $reviewed_ids; DELETE contradicts WHERE proposal_id IN $reviewed_ids; DELETE derived_from WHERE proposal_id IN $reviewed_ids; DELETE related_to WHERE proposal_id IN $reviewed_ids; UPDATE proposed_edge SET status='superseded',superseded_at=time::now(),supersession_reason='reviewed endpoint content or policy changed; fresh review required',resulting_edge_id=NONE,updated_at=time::now() WHERE id IN $reviewed_ids; ")
}

pub(super) fn source_snapshot_guards() -> &'static str {
    "FOR $expected IN $endpoint_sources { LET $matches=(SELECT VALUE id FROM source WHERE id=$expected.id AND metadata=$expected.metadata AND generation=$expected.generation AND successful_generation=$expected.successful_generation AND status=$expected.status AND content=$expected.content AND content_hash=$expected.content_hash AND title=$expected.title); IF array::len($matches)!=1 { THROW 'reviewed-endpoint-source-conflict'; }; }; "
}
impl Repository {
    pub(super) async fn endpoint_source_snapshots(&self, notes: &[Note]) -> Result<Vec<Source>> {
        let mut sources = Vec::new();
        let mut seen = HashSet::new();
        for note in notes {
            if let Some(id) = &note.source_id {
                if seen.insert(record_id_to_string(id)) {
                    let source: Source = self.db.select(id.clone()).await?.ok_or_else(conflict)?;
                    sources.push(source);
                }
            }
        }
        Ok(sources)
    }
    /// Called under the proposal lifecycle gate. Unlike legacy acceptance this family
    /// has no recoverable partial accepting state: edge and proposal commit together.
    pub(super) async fn accept_reviewed_endpoint_atomic_locked(
        &self,
        proposal: ProposedEdge,
        reviewer: Option<String>,
        action_reason: Option<String>,
        is_manual: bool,
    ) -> Result<ProposedEdge> {
        if !is_manual || reviewer.as_ref().is_none_or(|r| r.trim().is_empty()) {
            return Err(DbError::InvalidMutationRequest(
                "Reviewed endpoint acceptance requires an explicit manual reviewer".into(),
            ));
        }
        if proposal.status == ProposedEdgeStatus::Accepted {
            return Ok(proposal);
        }
        if proposal.status != ProposedEdgeStatus::Pending {
            return Err(conflict());
        }
        let notes = vec![
            self.get_note(&record_id_to_string(&proposal.from_id))
                .await?
                .ok_or_else(conflict)?,
            self.get_note(&record_id_to_string(&proposal.to_id))
                .await?
                .ok_or_else(conflict)?,
        ];
        let sources = self.endpoint_source_snapshots(&notes).await?;
        if !self.reviewed_endpoint_proposal_current(&proposal).await? {
            return Err(conflict());
        }
        self.commit_reviewed_endpoint_acceptance(proposal, notes, sources, reviewer, action_reason)
            .await
    }
    pub(super) async fn commit_reviewed_endpoint_acceptance(
        &self,
        proposal: ProposedEdge,
        notes: Vec<Note>,
        sources: Vec<Source>,
        reviewer: Option<String>,
        action_reason: Option<String>,
    ) -> Result<ProposedEdge> {
        let id = proposal.id.clone().ok_or_else(conflict)?;
        let table = super::graph::note_edge_table(&proposal.edge_type)?;
        let dedupe =
            super::graph::edge_dedupe_key(&proposal.from_id, &proposal.to_id, &proposal.edge_type);
        let edge = RecordId::new(
            table,
            format!(
                "reviewed_{:x}",
                Sha256::digest(record_id_to_string(&id).as_bytes())
            ),
        );
        let mut guards = String::new();
        for index in 0..2 {
            guards.push_str(
                &super::notes::editor_snapshot_guard(false)
                    .replace("$editor_source", &format!("$endpoints[{index}].id"))
                    .replace("$editor_expected", &format!("$endpoints[{index}]"))
                    .replace("$editor_matches", &format!("$matches_{index}")),
            );
        }
        let mut response=self.db.query(format!("BEGIN TRANSACTION; {guards} {} LET $pending=(SELECT VALUE id FROM proposed_edge WHERE id=$proposal AND status='pending' AND generator='remote-reviewed-endpoint' AND reason=$evidence AND updated_at=$updated); IF array::len($pending)!=1 OR array::len((SELECT VALUE id FROM {table} WHERE dedupe_key=$dedupe LIMIT 1))!=0 {{ THROW 'reviewed-endpoint-accept-conflict'; }}; CREATE $edge SET in=$from,out=$to,confidence=1.0,reason=$evidence,provenance='manual',proposal_id=$proposal,is_manual=true,dedupe_key=$dedupe,created_at=time::now(); UPDATE $proposal SET status='accepted',reviewed_at=time::now(),reviewer=$reviewer,action_reason=$action_reason,acceptance_is_manual=true,resulting_edge_id=$edge,updated_at=time::now(); COMMIT TRANSACTION;",source_snapshot_guards()))
            .bind(("endpoints",notes)).bind(("endpoint_sources",sources)).bind(("proposal",id.clone())).bind(("evidence",proposal.reason.clone())).bind(("updated",proposal.updated_at)).bind(("dedupe",dedupe)).bind(("edge",edge)).bind(("from",proposal.from_id)).bind(("to",proposal.to_id)).bind(("reviewer",reviewer)).bind(("action_reason",action_reason)).await?;
        if !response.take_errors().is_empty() {
            return Err(conflict());
        }
        self.get_edge_proposal(&id).await?.ok_or_else(conflict)
    }

    /// Fail closed for malformed/missing immutable quote evidence or endpoint revisions.
    pub async fn reviewed_endpoint_proposal_current(
        &self,
        proposal: &ProposedEdge,
    ) -> Result<bool> {
        if proposal.generator != "remote-reviewed-endpoint" {
            return Ok(true);
        }
        let Ok(evidence) = serde_json::from_str::<serde_json::Value>(&proposal.reason) else {
            return Ok(false);
        };
        if evidence["contract"] != 1 {
            return Ok(false);
        }
        let mut ids = HashSet::new();
        for name in ["from", "to"] {
            let Some(id) = evidence[name]["id"].as_str() else {
                return Ok(false);
            };
            let Some(revision) = evidence[name]["revision"].as_str() else {
                return Ok(false);
            };
            let Some(quote) = evidence[name]["quote"].as_str() else {
                return Ok(false);
            };
            if quote.trim().is_empty() || !ids.insert(id.to_string()) {
                return Ok(false);
            }
            let record = match self.inspect_record(id, 0).await {
                Ok(v) => v,
                Err(DbError::NotFound(..)) => return Ok(false),
                Err(e) => return Err(e),
            };
            if record.revision != revision || !record.content.contains(quote) {
                return Ok(false);
            }
        }
        Ok(ids
            == [
                record_id_to_string(&proposal.from_id),
                record_id_to_string(&proposal.to_id),
            ]
            .into_iter()
            .collect())
    }

    /// Creates an immutable proposal and its idempotency receipt in one
    /// transaction. Snapshot/quote checks run below the repository mutation
    /// lock and their full note snapshots are fenced again in SQL.
    pub async fn propose_remote_endpoint(
        &self,
        mut request: RemoteEndpointProposalInput,
    ) -> Result<RemoteCaptureReceipt> {
        if request.mutation.operation != "propose_endpoint"
            || request.mutation.payload_fingerprint.len() != 64
            || !request
                .mutation
                .payload_fingerprint
                .bytes()
                .all(|b| b.is_ascii_hexdigit())
            || !request.edge_type.is_note_edge()
            || request.rationale.trim().is_empty()
            || request.rationale.chars().count() > 2048
        {
            return Err(DbError::InvalidMutationRequest(
                "invalid endpoint proposal request".into(),
            ));
        }
        for value in [&request.mutation.instance_id, &request.mutation.request_id] {
            if value.is_empty()
                || value.len() > 256
                || value.chars().count() > 128
                || value.trim() != value
                || value.chars().any(char::is_control)
            {
                return Err(DbError::InvalidMutationRequest(
                    "invalid proposal identity".into(),
                ));
            }
        }
        for (id, revision, quote) in [
            (
                &request.from_id,
                &request.from_revision,
                &request.from_quote,
            ),
            (&request.to_id, &request.to_revision, &request.to_quote),
        ] {
            if id.len() > 512
                || id.trim() != id
                || id.chars().any(char::is_control)
                || record_id_to_string(&parse_record_id(id, Some("note"))?) != *id
                || revision.len() != 64
                || !revision
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                || quote.trim().is_empty()
                || quote.trim() != quote
                || quote.len() > 2048
                || quote.chars().count() > 1024
                || quote.contains('\0')
            {
                return Err(DbError::InvalidMutationRequest(
                    "two canonical bounded revision-pinned quotes required".into(),
                ));
            }
        }
        super::graph::note_edge_table(&request.edge_type)?;
        let payload = &request.mutation.payload;
        if payload["request_id"] != request.mutation.request_id
            || payload["confirmed"] != true
            || payload["from"]["id"] != request.from_id
            || payload["from"]["revision"] != request.from_revision
            || payload["from"]["quote"] != request.from_quote
            || payload["to"]["id"] != request.to_id
            || payload["to"]["revision"] != request.to_revision
            || payload["to"]["quote"] != request.to_quote
            || payload["relationship"] != request.edge_type.to_string()
            || payload["rationale"] != request.rationale
        {
            return Err(DbError::InvalidMutationRequest(
                "journal payload must describe the single intended proposal exactly".into(),
            ));
        }
        let guard = self.mutation_guard().await;
        if let Some(row) = receipt(self, &request.mutation).await? {
            if row.operation == request.mutation.operation
                && row.payload_fingerprint == request.mutation.payload_fingerprint
                && row.payload == request.mutation.payload
            {
                return Ok(RemoteCaptureReceipt {
                    result: row.result,
                    replayed: true,
                });
            }
            return Err(conflict());
        }
        let from = self
            .mutation_note_snapshot(&guard, &request.from_id)
            .await?;
        let to = self.mutation_note_snapshot(&guard, &request.to_id).await?;
        if from.inspection.revision != request.from_revision
            || to.inspection.revision != request.to_revision
            || !from.note.content.contains(&request.from_quote)
            || !to.note.content.contains(&request.to_quote)
        {
            return Err(conflict());
        }
        let from_snapshot = from.note.id.clone().ok_or_else(conflict)?;
        let to_snapshot = to.note.id.clone().ok_or_else(conflict)?;
        let mut from_id = from_snapshot.clone();
        let mut to_id = to_snapshot.clone();
        if from_id == to_id {
            return Err(DbError::InvalidMutationRequest(
                "endpoint proposal requires two distinct notes".into(),
            ));
        }
        let endpoint_sources = self
            .endpoint_source_snapshots(&[from.note.clone(), to.note.clone()])
            .await?;
        if self.inspect_record(&request.from_id, 0).await?.revision != request.from_revision
            || self.inspect_record(&request.to_id, 0).await?.revision != request.to_revision
        {
            return Err(conflict());
        }
        let reason = serde_json::json!({"contract":1,"rationale":request.rationale,"from":{"id":request.from_id,"revision":request.from_revision,"quote":request.from_quote},"to":{"id":request.to_id,"revision":request.to_revision,"quote":request.to_quote}}).to_string();
        super::graph::canonicalize_note_edge(&mut from_id, &mut to_id, &request.edge_type);
        let base = super::graph::edge_dedupe_key(&from_id, &to_id, &request.edge_type);
        let mut endpoints = vec![
            (
                &request.from_id,
                &request.from_revision,
                &request.from_quote,
            ),
            (&request.to_id, &request.to_revision, &request.to_quote),
        ];
        if request.edge_type.is_symmetric() {
            endpoints.sort();
        }
        let dedupe = format!(
            "reviewed-v1:{base}:{:x}",
            Sha256::digest(serde_json::to_vec(&endpoints).map_err(|_| conflict())?)
        );
        let existing: Option<RecordId> = self
            .db
            .query("SELECT VALUE id FROM proposed_edge WHERE dedupe_key=$dedupe LIMIT 1")
            .bind(("dedupe", dedupe.clone()))
            .await?
            .take(0)?;
        if existing.is_some() {
            return Err(conflict());
        }
        let key = format!(
            "endpoint_{:x}",
            Sha256::digest(
                serde_json::to_vec(&(&request.mutation.instance_id, &request.mutation.request_id))
                    .map_err(|e| DbError::InvalidMutationRequest(e.to_string()))?
            )
        );
        let proposal = RecordId::new("proposed_edge", key);
        request.mutation.result = serde_json::json!({
            "id": record_id_to_string(&proposal), "operation": "propose_endpoint",
            "actor": format!("mcp:{}", request.mutation.instance_id), "status": "pending"
        });
        let receipt_id = RecordId::new(
            "remote_mutation_receipt",
            format!(
                "{:x}",
                Sha256::digest(
                    serde_json::to_vec(&(
                        &request.mutation.instance_id,
                        &request.mutation.request_id
                    ))
                    .map_err(|e| DbError::InvalidMutationRequest(e.to_string()))?
                )
            ),
        );
        let from_guard = super::notes::editor_snapshot_guard(false)
            .replace("$editor_source", "$from_snapshot")
            .replace("$editor_expected", "$expected_from")
            .replace("$editor_matches", "$from_matches");
        let to_guard = super::notes::editor_snapshot_guard(false)
            .replace("$editor_source", "$to_snapshot")
            .replace("$editor_expected", "$expected_to")
            .replace("$editor_matches", "$to_matches");
        let query = format!(
            "BEGIN TRANSACTION; {from_guard}{to_guard} {} CREATE $proposal SET dedupe_key=$dedupe,in=$edge_from,out=$edge_to,edge_type=$edge_type,confidence=1.0,reason=$reason,generator='remote-reviewed-endpoint',generator_version='1',model=NONE,status='pending',created_at=time::now(),updated_at=time::now(); CREATE $receipt SET instance_id=$instance,request_id=$request,operation=$operation,target=$proposal,payload_fingerprint=$fingerprint,payload=$payload,result=$result,created_at=time::now(),updated_at=time::now(); COMMIT TRANSACTION;", source_snapshot_guards()
        );
        let mut response = self
            .db
            .query(query)
            .bind(("endpoint_sources", endpoint_sources))
            .bind(("from_snapshot", from_snapshot))
            .bind(("to_snapshot", to_snapshot))
            .bind(("edge_from", from_id))
            .bind(("edge_to", to_id))
            .bind(("expected_from", from.note))
            .bind(("expected_to", to.note))
            .bind(("proposal", proposal))
            .bind(("dedupe", dedupe))
            .bind(("edge_type", request.edge_type.to_string()))
            .bind(("reason", reason))
            .bind(("receipt", receipt_id))
            .bind(("instance", request.mutation.instance_id.clone()))
            .bind(("request", request.mutation.request_id.clone()))
            .bind(("operation", request.mutation.operation.clone()))
            .bind(("fingerprint", request.mutation.payload_fingerprint.clone()))
            .bind(("payload", request.mutation.payload.clone()))
            .bind(("result", request.mutation.result.clone()))
            .await?;
        let errors = response.take_errors();
        if !errors.is_empty() {
            if let Some(row) = receipt(self, &request.mutation).await? {
                if row.operation == request.mutation.operation
                    && row.payload_fingerprint == request.mutation.payload_fingerprint
                    && row.payload == request.mutation.payload
                {
                    return Ok(RemoteCaptureReceipt {
                        result: row.result,
                        replayed: true,
                    });
                }
            }
            return Err(conflict());
        }
        Ok(RemoteCaptureReceipt {
            result: request.mutation.result,
            replayed: false,
        })
    }
}

#[cfg(test)]
#[path = "remote_endpoint_proposals_tests.rs"]
mod tests;
