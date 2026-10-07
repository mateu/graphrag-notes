//! Durable uploaded Markdown admission and fenced worker mutations.
//! Providers never run while the transition gate is held. Running cancellation
//! only sets a durable flag, independently of that gate; an entered mutation
//! phase finishes before the worker acknowledges the next cancellation boundary.
use super::*;
use sha2::{Digest, Sha256};

#[cfg(test)]
#[path = "remote_jobs_tests.rs"]
mod tests;

#[path = "policy_migration.rs"]
mod policy_migration;
pub(super) fn validate_enrichment_entities(
    note: &Note,
    scope: Option<&str>,
    entities: &[Entity],
) -> Result<()> {
    policy_migration::validate_entities(note, scope, entities)
}

pub const MAX_REMOTE_UPLOAD_BYTES: usize = 65_536;
pub const MAX_REMOTE_UPLOAD_CHUNKS: usize = 200;
const MAX_REMOTE_JSON_BYTES: usize = 16 * 1024;
const MAX_REMOTE_CLAIM_SCAN: usize = 100;
const FENCE: &str = "remote-upload-worker-fence";
// Status requests must not materialize private extraction checkpoints (up to
// 16 MiB each) or saved inputs. Project their redaction inside the database,
// before a caller's bounded list can be deserialized in the service process.
const STATUS_FIELDS: &str = "id, job_type, source_generation, scope, item_ids, status, total_count, completed_count, failed_count, checkpoint, last_error, created_at, updated_at, finished_at, remote_instance_id, remote_request_id, remote_payload_fingerprint, {} AS remote_input, (remote_input.enrichment IS NOT NONE AND remote_input.enrichment IS NOT NULL) AS remote_enrichment_job, remote_admission, IF remote_result.policy_migration_stage IS NOT NONE THEN NONE ELSE remote_result END AS remote_result, remote_source_id, remote_source_uri, remote_source_generation, remote_admission_order, remote_phase, remote_migration_contract_version, remote_cancel_requested, remote_service_epoch, remote_worker_token";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RemoteEnrichmentInput {
    pub plan: SourceEnrichmentPlan,
    pub rollback: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RemoteUploadInput {
    pub authenticated_instance_id: String,
    pub request_id: String,
    pub payload_fingerprint: String,
    pub document_key: String,
    pub markdown: String,
    pub title: Option<String>,
    pub source_provenance: serde_json::Value,
    pub extract_entities: bool,
    #[serde(default)]
    pub preserve_unchanged: bool,
    /// Create only if this source is still absent when the worker begins.
    #[serde(default)]
    pub create_only: bool,
    /// Revision of the existing source inspected before this upload.
    #[serde(default)]
    pub expected_source_revision: Option<String>,
    /// Frozen explicit migration intent; old admissions omit this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy_migration: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enrichment: Option<RemoteEnrichmentInput>,
    pub processing_options: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RemoteJobAdmission {
    pub result: serde_json::Value,
    pub replayed: bool,
}

#[derive(Debug, Clone)]
pub struct RemoteJobLease {
    pub job_id: RecordId,
    pub instance_id: String,
    pub service_epoch: String,
    pub worker_token: String,
}

#[derive(Debug, Clone)]
pub struct RemoteUploadJob {
    pub job: ProcessingJob,
    pub instance_id: String,
    pub request_id: String,
    pub source_uri: String,
    pub source_id: Option<RecordId>,
    pub source_generation: Option<u64>,
    pub admission_order: Option<u64>,
    pub phase: String,
    pub cancel_requested: bool,
    pub service_epoch: Option<String>,
    pub worker_token: Option<String>,
    pub input: RemoteUploadInput,
    pub admission: serde_json::Value,
    pub result: Option<serde_json::Value>,
}

/// Owner-scoped progress without decoding the saved execution input. Damaged
/// input must remain inspectable after validation quarantine or recovery.
#[derive(Debug, Clone)]
pub struct RemoteUploadJobStatus {
    pub job: ProcessingJob,
    /// Family classification does not require decoding a potentially damaged plan.
    pub enrichment: bool,
    pub instance_id: String,
    pub source_id: Option<RecordId>,
    pub source_generation: Option<u64>,
    pub phase: String,
    pub cancel_requested: bool,
    pub admission: serde_json::Value,
    pub result: Option<serde_json::Value>,
}

impl From<RemoteUploadJob> for RemoteUploadJobStatus {
    fn from(job: RemoteUploadJob) -> Self {
        Self {
            enrichment: job.input.enrichment.is_some(),
            job: job.job,
            instance_id: job.instance_id,
            source_id: job.source_id,
            source_generation: job.source_generation,
            phase: job.phase,
            cancel_requested: job.cancel_requested,
            admission: job.admission,
            result: job.result,
        }
    }
}

#[derive(Debug, Deserialize, SurrealValue)]
struct JobRow {
    id: RecordId,
    job_type: String,
    source_generation: Option<String>,
    scope: Option<String>,
    item_ids: Vec<String>,
    status: String,
    total_count: i64,
    completed_count: i64,
    failed_count: i64,
    checkpoint: Option<String>,
    last_error: Option<String>,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    finished_at: Option<DateTime<Utc>>,
    remote_instance_id: String,
    remote_request_id: String,
    remote_payload_fingerprint: String,
    remote_input: serde_json::Value,
    remote_enrichment_job: Option<bool>,
    remote_admission: serde_json::Value,
    remote_result: Option<serde_json::Value>,
    remote_source_id: Option<RecordId>,
    remote_source_uri: String,
    remote_source_generation: Option<u64>,
    remote_admission_order: Option<u64>,
    remote_phase: String,
    remote_migration_contract_version: Option<i64>,
    remote_cancel_requested: bool,
    remote_service_epoch: Option<String>,
    remote_worker_token: Option<String>,
}

#[derive(Debug, Deserialize, SurrealValue)]
struct RecoveryFence {
    status: String,
    remote_service_epoch: Option<String>,
    remote_worker_token: Option<String>,
    remote_cancel_requested: bool,
}

#[derive(Debug, Deserialize, SurrealValue)]
struct AdmissionRow {
    remote_payload_fingerprint: String,
    remote_input: serde_json::Value,
    remote_admission: serde_json::Value,
}

#[derive(Debug, Deserialize, SurrealValue)]
struct QueuedJobIdentity {
    id: RecordId,
    remote_instance_id: String,
}

impl JobRow {
    fn processing_job(&self) -> ProcessingJob {
        ProcessingJob {
            id: Some(self.id.clone()),
            job_type: self.job_type.clone(),
            source_generation: self.source_generation.clone(),
            scope: self.scope.clone(),
            item_ids: self.item_ids.clone(),
            status: self.status.clone(),
            total_count: self.total_count,
            completed_count: self.completed_count,
            failed_count: self.failed_count,
            checkpoint: self.checkpoint.clone(),
            last_error: self.last_error.clone(),
            created_at: self.created_at,
            updated_at: self.updated_at,
            finished_at: self.finished_at,
            target_embedding_provider: None,
            target_embedding_model: None,
            target_embedding_dimension: None,
            reindex_item_fingerprints: None,
            reindex_lease_owner: None,
            reindex_lease_expires_at: None,
        }
    }

    fn status(self) -> RemoteUploadJobStatus {
        RemoteUploadJobStatus {
            job: self.processing_job(),
            enrichment: self.remote_enrichment_job.unwrap_or_else(|| {
                self.remote_input
                    .get("enrichment")
                    .is_some_and(|value| !value.is_null())
            }),
            instance_id: self.remote_instance_id,
            source_id: self.remote_source_id,
            source_generation: self.remote_source_generation,
            phase: self.remote_phase,
            cancel_requested: self.remote_cancel_requested,
            admission: self.remote_admission,
            result: self.remote_result,
        }
    }

    fn public(self) -> Result<RemoteUploadJob> {
        let job = self.processing_job();
        let input: RemoteUploadInput =
            serde_json::from_value(self.remote_input).map_err(|error| {
                DbError::InvalidRemoteRequest(format!("stored upload input shape: {error}"))
            })?;
        if input.policy_migration.is_some() && self.remote_migration_contract_version != Some(1) {
            return Err(DbError::InvalidRemoteRequest(
                "migration requires its schema20 durable contract marker".into(),
            ));
        }
        Ok(RemoteUploadJob {
            job,
            instance_id: self.remote_instance_id,
            request_id: self.remote_request_id,
            source_uri: self.remote_source_uri,
            source_id: self.remote_source_id,
            source_generation: self.remote_source_generation,
            admission_order: self.remote_admission_order,
            phase: self.remote_phase,
            cancel_requested: self.remote_cancel_requested,
            service_epoch: self.remote_service_epoch,
            worker_token: self.remote_worker_token,
            input,
            admission: self.remote_admission,
            result: self.remote_result,
        })
    }
}

fn identity(value: &str) -> Result<()> {
    if value.trim().is_empty()
        || value.trim() != value
        || value.chars().count() > 128
        || value.len() > 256
        || value.chars().any(char::is_control)
    {
        return Err(DbError::InvalidRemoteRequest(
            "bounded identities cannot contain controls or surrounding whitespace".into(),
        ));
    }
    Ok(())
}
fn digest(domain: &str, parts: &[&str]) -> String {
    let bytes = serde_json::to_vec(&(domain, parts)).expect("string tuples serialize");
    format!("{:x}", Sha256::digest(bytes))
}
/// Compare applied upload settings; metadata-only registration never invokes
/// extraction, and disabled extraction has no effect on generated vectors.
pub fn uploaded_processing_compatible(
    saved: &serde_json::Value,
    current: &serde_json::Value,
    extract_entities: bool,
    preserve_unchanged: bool,
) -> bool {
    let known = |options: &serde_json::Value| {
        options.is_object()
            && options["runtime"].as_str().is_some_and(|v| !v.is_empty())
            && ["provider", "model", "cache_identity", "endpoint_identity"]
                .iter()
                .all(|field| {
                    options["embedding"][field]
                        .as_str()
                        .is_some_and(|v| !v.is_empty())
                })
            && (!extract_entities
                || ["provider", "model", "cache_identity", "endpoint_identity"]
                    .iter()
                    .all(|field| {
                        options["extraction"][field]
                            .as_str()
                            .is_some_and(|v| !v.is_empty())
                    }))
    };
    if !known(saved) || !known(current) {
        return false;
    }
    let mut saved = saved.clone();
    let mut current = current.clone();
    if !extract_entities || preserve_unchanged {
        saved.as_object_mut().unwrap().remove("extraction");
        current.as_object_mut().unwrap().remove("extraction");
    }
    saved == current
}

pub fn uploaded_processing_policy_sha256(options: &serde_json::Value) -> Result<String> {
    Ok(format!(
        "{:x}",
        Sha256::digest(
            serde_json::to_vec(options)
                .map_err(|error| DbError::InvalidRemoteRequest(error.to_string()))?
        )
    ))
}

/// Provider-free source revision shared by inspection and generation fencing.
pub fn uploaded_source_revision(source: &Source, original_markdown: &str) -> Result<String> {
    let id = source
        .id
        .as_ref()
        .ok_or_else(|| DbError::InvalidRemoteRequest("source identity missing".into()))?;
    let origin = source
        .metadata
        .get("remote_upload_pending")
        .or_else(|| source.metadata.get("remote_upload"))
        .ok_or_else(|| DbError::InvalidRemoteRequest("uploaded source origin missing".into()))?;
    let mut snapshot = serde_json::to_vec(&(
        record_id_to_string(id),
        &source.title,
        &source.content,
        original_markdown,
        &source.content_hash,
        source.generation,
        source.successful_generation,
        &source.status,
        origin,
    ))
    .map_err(|error| DbError::InvalidRemoteRequest(error.to_string()))?;
    // Preserve historical unretired revision bytes; bind the new explicit
    // tombstone when present so old inspected drafts cannot bypass retirement.
    if source.metadata["remote_upload_retired"] == true {
        snapshot.extend_from_slice(b"\0retired");
    }
    if let Some(revision) = source.metadata.get("graph_policy_revision") {
        snapshot.extend_from_slice(b"\0graph_policy_revision\0");
        snapshot.extend_from_slice(
            &serde_json::to_vec(revision).map_err(|_| {
                DbError::InvalidRemoteRequest("graph revision serialization".into())
            })?,
        );
    }
    Ok(format!("{:x}", Sha256::digest(snapshot)))
}

pub fn uploaded_source_id(instance: &str, document_key: &str) -> String {
    format!(
        "source:{}",
        digest(
            "graphrag-remote-upload-source-v1",
            &[instance, document_key]
        )
    )
}
fn job_id(value: &str) -> Result<RecordId> {
    let key = value
        .strip_prefix("processing_job:")
        .ok_or_else(|| DbError::InvalidRemoteRequest("use a canonical processing_job:ID".into()))?;
    if key.len() != 64
        || !key
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(DbError::InvalidRemoteRequest(
            "invalid remote processing-job ID".into(),
        ));
    }
    Ok(RecordId::new("processing_job", key))
}
fn validate_input(input: &RemoteUploadInput) -> Result<()> {
    if let Some(enrichment) = &input.enrichment {
        super::source_enrichment::plan_valid(&enrichment.plan)?;
        if enrichment.plan.instance_id != input.authenticated_instance_id
            || enrichment.plan.document_key != input.document_key
            || enrichment.plan.target_processing_options != input.processing_options
            || (!enrichment.rollback && enrichment.plan.request_id != input.request_id)
            || (enrichment.rollback && enrichment.plan.request_id == input.request_id)
            || input.policy_migration.is_some()
            || !input.extract_entities
            || input.preserve_unchanged
            || input.create_only
            || input.expected_source_revision.as_deref()
                != Some(&enrichment.plan.expected_source_revision)
        {
            return Err(DbError::InvalidRemoteRequest(
                "invalid enrichment job contract".into(),
            ));
        }
    }
    identity(&input.authenticated_instance_id)?;
    identity(&input.request_id)?;
    if let Some(migration) = &input.policy_migration {
        let fields = [
            "original_policy_sha256",
            "target_policy_sha256",
            "plan_sha256",
        ];
        let valid = migration
            .as_object()
            .is_some_and(|value| value.len() == fields.len())
            && fields.iter().all(|field| {
                migration[*field].as_str().is_some_and(|value| {
                    value.len() == 64
                        && value
                            .bytes()
                            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                })
            });
        if !valid
            || !input.extract_entities
            || input.preserve_unchanged
            || input.create_only
            || input.expected_source_revision.is_none()
            || migration["original_policy_sha256"] == migration["target_policy_sha256"]
            || migration["target_policy_sha256"]
                != uploaded_processing_policy_sha256(&input.processing_options)?
        {
            return Err(DbError::InvalidRemoteRequest("explicit source policy migration requires exact old/target/plan digests, extraction, and an existing source revision".into()));
        }
    }
    if input.create_only && (input.preserve_unchanged || input.expected_source_revision.is_some()) {
        return Err(DbError::InvalidRemoteRequest(
            "create_only cannot accompany preserve_unchanged or expected_source_revision".into(),
        ));
    }
    if input.document_key.trim().is_empty()
        || input.document_key.trim() != input.document_key
        || input.document_key.chars().count() > 256
        || input.document_key.len() > 512
        || input.document_key.chars().any(char::is_control)
    {
        return Err(DbError::InvalidRemoteRequest(
            "document key exceeds its bounds or contains controls or surrounding whitespace".into(),
        ));
    }
    if input
        .expected_source_revision
        .as_ref()
        .is_some_and(|revision| {
            revision.len() != 64
                || !revision
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        })
    {
        return Err(DbError::InvalidRemoteRequest(
            "expected source revision must be lowercase SHA-256".into(),
        ));
    }
    if input.payload_fingerprint.len() != 64
        || !input
            .payload_fingerprint
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        || input.markdown.trim().is_empty()
        || input.markdown.len() > MAX_REMOTE_UPLOAD_BYTES
        || input
            .title
            .as_ref()
            .is_some_and(|title| title.chars().count() > 512)
        || !input.processing_options.is_object()
    {
        return Err(DbError::InvalidRemoteRequest(
            "upload content, fingerprint, title or processing snapshot exceeds its bounds".into(),
        ));
    }
    for value in [&input.source_provenance, &input.processing_options] {
        if serde_json::to_vec(value)
            .map_err(|error| DbError::InvalidRemoteRequest(error.to_string()))?
            .len()
            > MAX_REMOTE_JSON_BYTES
        {
            return Err(DbError::InvalidRemoteRequest(
                "upload provenance or processing snapshot exceeds 16 KiB".into(),
            ));
        }
    }
    Ok(())
}
fn guard_sql() -> String {
    format!(
        "LET $owned = (UPDATE $job SET updated_at = time::now() WHERE job_type = 'remote_upload' AND remote_instance_id = $instance AND status = 'running' AND remote_service_epoch = $epoch AND remote_worker_token = $worker AND remote_cancel_requested = false RETURN VALUE id); IF array::len($owned) != 1 {{ THROW '{FENCE}'; }}; "
    )
}
fn check_write(
    errors: HashMap<usize, surrealdb::Error>,
    lease: &RemoteJobLease,
    operation: &'static str,
) -> Result<()> {
    migration_statement_errors(operation, &errors);
    if errors.is_empty() {
        return Ok(());
    }
    if errors
        .values()
        .any(|error| error.to_string().contains(FENCE))
    {
        return Err(DbError::RemoteJobOwnershipLost(record_id_to_string(
            &lease.job_id,
        )));
    }
    Err(DbError::QueryFailed(format!(
        "atomic remote upload mutation failed: {errors:?}"
    )))
}

/// Legacy history predates numbered admissions. Use a total order across
/// mixed histories: legacy timestamp/ID, then durable number/timestamp/ID.
/// Pairwise timestamp fallback when only one number is absent can cycle after
/// clock rollback. Arguments are fixed query expressions, never caller input.
fn admission_precedes_sql(
    left_order: &str,
    left_created: &str,
    left_id: &str,
    right_order: &str,
    right_created: &str,
    right_id: &str,
) -> String {
    format!(
        "(({left_order} = NONE AND {right_order} != NONE) \
         OR ({left_order} != NONE AND {right_order} != NONE AND {left_order} < {right_order}) \
         OR ((({left_order} = NONE AND {right_order} = NONE) \
         OR ({left_order} != NONE AND {right_order} != NONE AND {left_order} = {right_order})) \
         AND ({left_created} < {right_created} OR ({left_created} = {right_created} AND {left_id} < {right_id}))))"
    )
}

/// Share scheduling precedence between the bounded candidate query and the
/// explicit claim check.
fn claim_blocker_sql(instance: &str, uri: &str, id: &str, order: &str, created: &str) -> String {
    let earlier = admission_precedes_sql(
        "remote_admission_order",
        "created_at",
        "id",
        order,
        created,
        id,
    );
    format!(
        "job_type = 'remote_upload' AND remote_instance_id = {instance} \
         AND remote_source_uri = {uri} AND id != {id} \
         AND (status = 'running' OR (status = 'queued' AND remote_cancel_requested = false \
         AND {earlier}))"
    )
}

impl Repository {
    /// Keep resume status/decode/compatibility reads before one retirement
    /// boundary. Drop this guard before entering the locked resume transition.
    pub async fn remote_upload_resume_preflight_guard(&self) -> tokio::sync::OwnedMutexGuard<()> {
        self.remote_job_transition_lock.clone().lock_owned().await
    }
    async fn remote_job_row(&self, instance: &str, id: &RecordId) -> Result<Option<JobRow>> {
        Ok(self.db.query("SELECT * FROM processing_job WHERE id = $id AND job_type = 'remote_upload' AND remote_instance_id = $instance LIMIT 1")
            .bind(("id", id.clone())).bind(("instance", instance.to_string())).await?.take(0)?)
    }
    async fn remote_request_row(
        &self,
        instance: &str,
        request: &str,
    ) -> Result<Option<AdmissionRow>> {
        Ok(self.db.query("SELECT remote_payload_fingerprint, remote_input, remote_admission FROM processing_job WHERE job_type = 'remote_upload' AND remote_instance_id = $instance AND remote_request_id = $request LIMIT 1")
            .bind(("instance", instance.to_string())).bind(("request", request.to_string())).await?.take(0)?)
    }
    pub async fn find_remote_upload_admission(
        &self,
        instance: &str,
        request: &str,
        fingerprint: &str,
    ) -> Result<Option<RemoteJobAdmission>> {
        identity(instance)?;
        identity(request)?;
        match self.remote_request_row(instance, request).await? {
            Some(row) if row.remote_payload_fingerprint == fingerprint => {
                Ok(Some(RemoteJobAdmission {
                    result: row.remote_admission,
                    replayed: true,
                }))
            }
            Some(_) => Err(DbError::RemoteRequestConflict {
                instance_id: instance.into(),
                request_id: request.into(),
            }),
            None => Ok(None),
        }
    }
    pub async fn admit_remote_upload(
        &self,
        input: RemoteUploadInput,
    ) -> Result<RemoteJobAdmission> {
        validate_input(&input)?;
        let _gate = self.remote_job_transition_lock.lock().await;
        let payload = serde_json::to_value(&input)
            .map_err(|error| DbError::InvalidRemoteRequest(error.to_string()))?;
        if let Some(row) = self
            .remote_request_row(&input.authenticated_instance_id, &input.request_id)
            .await?
        {
            if row.remote_payload_fingerprint != input.payload_fingerprint
                || row.remote_input != payload
            {
                return Err(DbError::RemoteRequestConflict {
                    instance_id: input.authenticated_instance_id,
                    request_id: input.request_id,
                });
            }
            return Ok(RemoteJobAdmission {
                result: row.remote_admission,
                replayed: true,
            });
        }
        if input.enrichment.is_some() {
            let receipts: Vec<RecordId> = self.db.query("SELECT VALUE id FROM remote_mutation_receipt WHERE instance_id=$instance AND request_id=$request LIMIT 1").bind(("instance",input.authenticated_instance_id.clone())).bind(("request",input.request_id.clone())).await?.take(0)?;
            if !receipts.is_empty() {
                return Err(DbError::RemoteRequestConflict {
                    instance_id: input.authenticated_instance_id.clone(),
                    request_id: input.request_id.clone(),
                });
            }
        }
        let key = digest(
            "graphrag-remote-upload-job-v1",
            &[&input.authenticated_instance_id, &input.request_id],
        );
        let id = RecordId::new("processing_job", key);
        let source_key = digest(
            "graphrag-remote-upload-source-v1",
            &[&input.authenticated_instance_id, &input.document_key],
        );
        let source_id = RecordId::new("source", source_key.clone());
        let source_uri = format!("mcp://upload/{source_key}");
        let now = Utc::now();
        let result = serde_json::json!({"job_id": record_id_to_string(&id), "source_id":record_id_to_string(&source_id), "source_uri": source_uri, "status": "queued", "created_at": now.to_rfc3339()});
        let journal = if input.enrichment.is_some() {
            "IF array::len((SELECT VALUE id FROM remote_mutation_receipt WHERE instance_id=$instance AND request_id=$request LIMIT 1))!=0 { THROW 'enrichment-request-conflict'; }; CREATE $receipt SET instance_id=$instance, request_id=$request, operation=$operation, target=$source, payload_fingerprint=$fingerprint, payload=$input, result=$admission, created_at=time::now(),updated_at=time::now(); "
        } else {
            ""
        };
        let receipt = RecordId::new(
            "remote_mutation_receipt",
            digest(
                "graphrag-enrichment-admission-v1",
                &[&input.authenticated_instance_id, &input.request_id],
            ),
        );
        let operation = if input.enrichment.as_ref().is_some_and(|i| i.rollback) {
            "rollback_enrichment"
        } else {
            "enrich_source"
        };
        self.db.query(format!("BEGIN TRANSACTION; {journal} LET $prior_orders = (SELECT VALUE remote_admission_order FROM processing_job WHERE job_type = 'remote_upload' AND remote_instance_id = $instance AND remote_source_uri = $uri AND remote_admission_order IS NOT NONE ORDER BY remote_admission_order DESC LIMIT 1); CREATE $id SET job_type = 'remote_upload', status = 'queued', total_count = 0, completed_count = 0, failed_count = 0, item_ids = [], created_at = <datetime>$now, updated_at = <datetime>$now, remote_instance_id = $instance, remote_request_id = $request, remote_payload_fingerprint = $fingerprint, remote_input = $input, remote_admission = $admission, remote_source_uri = $uri, remote_admission_order = IF array::len($prior_orders) = 0 THEN 1 ELSE $prior_orders[0] + 1 END, remote_phase = $phase, remote_migration_contract_version = $migration_contract, remote_cancel_requested = false; COMMIT TRANSACTION;"))
            .bind(("source",source_id)).bind(("receipt",receipt)).bind(("operation",operation)).bind(("id", id)).bind(("now", now.to_rfc3339())).bind(("instance", input.authenticated_instance_id))
            .bind(("migration_contract", input.policy_migration.as_ref().map(|_| 1_i64)))
            .bind(("phase", if input.enrichment.is_some() { "enrichment_admitted" } else if input.policy_migration.is_some() { "migration_admitted" } else { "admitted" })).bind(("request", input.request_id)).bind(("fingerprint", input.payload_fingerprint))
            .bind(("input", payload)).bind(("admission", result.clone())).bind(("uri", source_uri)).await?.check()?;
        Ok(RemoteJobAdmission {
            result,
            replayed: false,
        })
    }
    pub async fn get_remote_upload_job(
        &self,
        instance: &str,
        id: &str,
    ) -> Result<Option<RemoteUploadJob>> {
        identity(instance)?;
        self.remote_job_row(instance, &job_id(id)?)
            .await?
            .map(JobRow::public)
            .transpose()
    }

    pub async fn get_remote_upload_job_status(
        &self,
        instance: &str,
        id: &str,
    ) -> Result<Option<RemoteUploadJobStatus>> {
        identity(instance)?;
        Ok(self
            .remote_job_status_row(instance, &job_id(id)?)
            .await?
            .map(JobRow::status))
    }

    async fn remote_job_status_row(&self, instance: &str, id: &RecordId) -> Result<Option<JobRow>> {
        let row: Option<JobRow> = self.db.query(format!("SELECT {STATUS_FIELDS} FROM processing_job WHERE id = $id AND job_type = 'remote_upload' AND remote_instance_id = $instance LIMIT 1"))
            .bind(("id", id.clone())).bind(("instance", instance.to_string())).await?.take(0)?;
        Ok(row)
    }

    async fn remote_job_rows(&self, instance: &str, limit: usize) -> Result<Vec<JobRow>> {
        identity(instance)?;
        if !(1..=200).contains(&limit) {
            return Err(DbError::InvalidRemoteRequest(
                "job limit must be 1–200".into(),
            ));
        }
        Ok(self.db.query("SELECT * FROM processing_job WHERE job_type = 'remote_upload' AND remote_instance_id = $instance ORDER BY updated_at DESC, id ASC LIMIT $limit")
            .bind(("instance", instance.to_string())).bind(("limit", limit)).await?.take(0)?)
    }

    pub async fn list_remote_upload_jobs(
        &self,
        instance: &str,
        limit: usize,
    ) -> Result<Vec<RemoteUploadJob>> {
        self.remote_job_rows(instance, limit)
            .await?
            .into_iter()
            .map(JobRow::public)
            .collect()
    }

    pub async fn list_remote_upload_job_statuses(
        &self,
        instance: &str,
        limit: usize,
    ) -> Result<Vec<RemoteUploadJobStatus>> {
        identity(instance)?;
        if !(1..=200).contains(&limit) {
            return Err(DbError::InvalidRemoteRequest(
                "job limit must be 1–200".into(),
            ));
        }
        let rows: Vec<JobRow> = self.db.query(format!("SELECT {STATUS_FIELDS} FROM processing_job WHERE job_type = 'remote_upload' AND remote_instance_id = $instance ORDER BY updated_at DESC, id ASC LIMIT $limit"))
            .bind(("instance", instance.to_string())).bind(("limit", limit)).await?.take(0)?;
        Ok(rows.into_iter().map(JobRow::status).collect())
    }

    async fn request_remote_upload_cancellation(
        &self,
        instance: &str,
        id: &str,
    ) -> Result<RecordId> {
        identity(instance)?;
        let id = job_id(id)?;
        // Deliberately no transition/lifecycle mutex. A provider-blocked worker
        // can observe this immediately, and an entered DB phase can finish.
        self.db.query("UPDATE $id SET remote_cancel_requested = true, finished_at = IF status = 'queued' THEN time::now() ELSE finished_at END, status = IF status = 'queued' THEN 'cancelled' ELSE status END, updated_at = time::now() WHERE job_type = 'remote_upload' AND remote_instance_id = $instance AND (status = 'queued' OR status = 'running') RETURN NONE")
            .bind(("id", id.clone())).bind(("instance", instance.to_string())).await?.check()?;
        Ok(id)
    }

    pub async fn cancel_remote_upload_job(
        &self,
        instance: &str,
        id: &str,
    ) -> Result<RemoteUploadJob> {
        let id = self
            .request_remote_upload_cancellation(instance, id)
            .await?;
        self.remote_job_row(instance, &id)
            .await?
            .ok_or_else(|| DbError::NotFound("remote upload job".into(), record_id_to_string(&id)))?
            .public()
    }

    pub async fn cancel_remote_upload_job_status(
        &self,
        instance: &str,
        id: &str,
    ) -> Result<RemoteUploadJobStatus> {
        let id = self
            .request_remote_upload_cancellation(instance, id)
            .await?;
        Ok(self
            .remote_job_status_row(instance, &id)
            .await?
            .ok_or_else(|| DbError::NotFound("remote upload job".into(), record_id_to_string(&id)))?
            .status())
    }
    pub async fn resume_remote_upload_job(
        &self,
        instance: &str,
        id: &str,
    ) -> Result<RemoteUploadJob> {
        identity(instance)?;
        let id = job_id(id)?;
        let _gate = self.remote_job_transition_lock.lock().await;
        let row = self.remote_job_row(instance, &id).await?.ok_or_else(|| {
            DbError::NotFound("remote upload job".into(), record_id_to_string(&id))
        })?;
        if row.remote_phase == "retired" {
            return Err(DbError::RemoteJobSourceConflict(record_id_to_string(&id)));
        }
        if !matches!(row.status.as_str(), "failed" | "cancelled") {
            return Err(DbError::RemoteJobOwnershipLost(record_id_to_string(&id)));
        }
        let job = row.public()?;
        // A newer execution can already own this source before it prepares a
        // generation. Do not queue its older prepared origin underneath it:
        // generation preparation would then see a conflicting queued owner.
        let active: Vec<RecordId> = self.db.query("SELECT VALUE id FROM processing_job WHERE job_type = 'remote_upload' AND remote_instance_id = $instance AND remote_source_uri = $uri AND id != $job AND status = 'running' LIMIT 1")
            .bind(("instance", instance.to_string())).bind(("uri", job.source_uri.clone())).bind(("job", id.clone())).await?.take(0)?;
        if !active.is_empty() {
            return Err(DbError::RemoteJobSourceConflict(record_id_to_string(&id)));
        }
        if job.input.enrichment.is_none() {
            self.ensure_remote_source_current(&job).await?;
        }
        if job.input.policy_migration.is_some() && job.source_generation.is_none() {
            let _lifecycle = self.proposal_acceptance_lock.lock().await;
            let source = self
                .get_source(&job.source_uri)
                .await?
                .ok_or_else(|| DbError::RemoteJobSourceConflict(record_id_to_string(&id)))?;
            let origin = source.metadata["remote_upload"]["job_id"]
                .as_str()
                .ok_or_else(|| DbError::RemoteJobSourceConflict(record_id_to_string(&id)))?;
            let original = self
                .get_remote_upload_job(instance, origin)
                .await?
                .ok_or_else(|| DbError::RemoteJobSourceConflict(record_id_to_string(&id)))?;
            if !policy_migration::matches_origin(&source, &original, &job.input)?
                || job.input.expected_source_revision.as_deref()
                    != Some(uploaded_source_revision(&source, &original.input.markdown)?.as_str())
            {
                return Err(DbError::RemoteJobSourceConflict(record_id_to_string(&id)));
            }
        }
        let row: Option<JobRow> = self.db.query("UPDATE $id SET status = 'queued', remote_cancel_requested = false, remote_service_epoch = NONE, remote_worker_token = NONE, last_error = NONE, finished_at = NONE, updated_at = time::now() WHERE job_type = 'remote_upload' AND remote_instance_id = $instance AND (status = 'failed' OR status = 'cancelled') RETURN AFTER")
            .bind(("id", id.clone())).bind(("instance", instance.to_string())).await?.take(0)?;
        row.ok_or_else(|| DbError::RemoteJobOwnershipLost(record_id_to_string(&id)))?
            .public()
    }
    async fn claim_remote_upload_locked(
        &self,
        instance: &str,
        id: &RecordId,
        epoch: &str,
        worker: &str,
    ) -> Result<Option<RemoteUploadJob>> {
        identity(instance)?;
        identity(epoch)?;
        identity(worker)?;
        let Some(row) = self.remote_job_row(instance, id).await? else {
            return Ok(None);
        };
        if row.status != "queued" || row.remote_cancel_requested || row.remote_phase == "retired" {
            return Ok(None);
        }
        // Decode and validate durable payload before changing ownership. Keep
        // the validated snapshot; the committing response carries only the ID,
        // so no payload decode can lose the newly acquired execution fence.
        let mut job = row.public()?;
        validate_input(&job.input)?;
        if job.input.authenticated_instance_id != instance || job.input.request_id != job.request_id
        {
            return Err(DbError::InvalidRemoteRequest(
                "stored upload identity does not match its owner".into(),
            ));
        }
        // Claim is source ownership, including the window before generation
        // preparation and the period after promotion through terminalization.
        let blockers: Vec<RecordId> = self
            .db
            .query(format!(
                "SELECT VALUE id FROM processing_job WHERE {} LIMIT 1",
                claim_blocker_sql("$instance", "$uri", "$job", "$order", "<datetime>$created")
            ))
            .bind(("instance", instance.to_string()))
            .bind(("uri", job.source_uri.clone()))
            .bind(("job", id.clone()))
            .bind(("order", job.admission_order))
            .bind(("created", job.job.created_at.to_rfc3339()))
            .await?
            .take(0)?;
        if !blockers.is_empty() {
            return Ok(None);
        }
        let now = Utc::now();
        let mut response = migration_probe("claim.owner_update", "sdk_await", self.db.query("UPDATE $id SET status = 'running', remote_service_epoch = $epoch, remote_worker_token = $worker, updated_at = <datetime>$now WHERE job_type = 'remote_upload' AND remote_instance_id = $instance AND status = 'queued' AND remote_cancel_requested = false RETURN VALUE id")
            .bind(("id", id.clone())).bind(("instance", instance.to_string())).bind(("epoch", epoch.to_string())).bind(("worker", worker.to_string())).bind(("now", now.to_rfc3339()))).await?;
        let claimed: Vec<RecordId> =
            migration_probe_result("claim.owner_update", "response_take", response.take(0))?;
        if claimed.is_empty() {
            return Ok(None);
        }
        job.job.status = "running".into();
        job.job.updated_at = now;
        job.service_epoch = Some(epoch.into());
        job.worker_token = Some(worker.into());
        Ok(Some(job))
    }
    pub async fn claim_remote_upload_job(
        &self,
        instance: &str,
        id: &str,
        epoch: &str,
        worker: &str,
    ) -> Result<RemoteUploadJob> {
        let id = job_id(id)?;
        let _gate = self.remote_job_transition_lock.lock().await;
        self.claim_remote_upload_locked(instance, &id, epoch, worker)
            .await?
            .ok_or_else(|| DbError::RemoteJobOwnershipLost(record_id_to_string(&id)))
    }
    pub async fn claim_next_remote_upload(
        &self,
        epoch: &str,
        worker: &str,
    ) -> Result<Option<RemoteUploadJob>> {
        identity(epoch)?;
        identity(worker)?;
        let _gate = self.remote_job_transition_lock.lock().await;
        // Select only eligible document heads before applying the bound. A
        // backlog behind one active source cannot hide unrelated documents.
        // Read identities only so malformed execution input remains eligible
        // for validation quarantine rather than breaking the candidate query.
        let blockers = claim_blocker_sql(
            "$parent.remote_instance_id",
            "$parent.remote_source_uri",
            "$parent.id",
            "$parent.remote_admission_order",
            "$parent.created_at",
        );
        let query = format!(
            "SELECT id, remote_instance_id, created_at FROM processing_job WHERE job_type = 'remote_upload' AND remote_instance_id IS NOT NONE AND status = 'queued' AND remote_cancel_requested = false AND array::len((SELECT VALUE id FROM processing_job WHERE {blockers} LIMIT 1)) = 0 ORDER BY created_at ASC, id ASC LIMIT $limit"
        );
        let mut remaining = MAX_REMOTE_CLAIM_SCAN;
        while remaining > 0 {
            let mut response = migration_probe(
                "claim.queue_read",
                "sdk_await",
                self.db.query(&query).bind(("limit", remaining)),
            )
            .await?;
            let rows: Vec<QueuedJobIdentity> =
                migration_probe_result("claim.queue_read", "response_take", response.take(0))?;
            if rows.is_empty() {
                break;
            }
            for row in rows {
                remaining -= 1;
                match migration_probe(
                    "claim",
                    "owned_claim",
                    self.claim_remote_upload_locked(
                        &row.remote_instance_id,
                        &row.id,
                        epoch,
                        worker,
                    ),
                )
                .await
                {
                    Ok(Some(job)) => return Ok(Some(job)),
                    Ok(None) => continue,
                    Err(DbError::InvalidRemoteRequest(_)) => {
                        // Deterministic saved-input errors are terminal before
                        // a worker fence. Retain input/checkpoint for repair and
                        // explicit resume; never hide storage faults. Reselect
                        // within the bound to expose this document's next head.
                        self.db.query("UPDATE $id SET status = 'failed', last_error = 'validation', finished_at = time::now(), updated_at = time::now(), remote_service_epoch = NONE, remote_worker_token = NONE WHERE job_type = 'remote_upload' AND remote_instance_id = $instance AND status = 'queued' AND remote_cancel_requested = false RETURN NONE")
                            .bind(("id", row.id)).bind(("instance", row.remote_instance_id)).await?.check()?;
                    }
                    Err(error) => return Err(error),
                }
            }
        }
        Ok(None)
    }
    async fn owned_remote_upload(
        &self,
        lease: &RemoteJobLease,
        allow_cancel: bool,
    ) -> Result<RemoteUploadJob> {
        identity(&lease.instance_id)?;
        identity(&lease.service_epoch)?;
        identity(&lease.worker_token)?;
        let row = self
            .remote_job_row(&lease.instance_id, &lease.job_id)
            .await?
            .ok_or_else(|| DbError::RemoteJobOwnershipLost(record_id_to_string(&lease.job_id)))?;
        if row.status != "running"
            || row.remote_service_epoch.as_deref() != Some(&lease.service_epoch)
            || row.remote_worker_token.as_deref() != Some(&lease.worker_token)
        {
            return Err(DbError::RemoteJobOwnershipLost(record_id_to_string(
                &lease.job_id,
            )));
        }
        if row.remote_cancel_requested && !allow_cancel {
            return Err(DbError::RemoteJobCancelled(record_id_to_string(
                &lease.job_id,
            )));
        }
        row.public()
    }
    pub async fn owned_remote_upload_job(&self, lease: &RemoteJobLease) -> Result<RemoteUploadJob> {
        self.owned_remote_upload(lease, false).await
    }
    async fn ensure_remote_source_current(&self, job: &RemoteUploadJob) -> Result<()> {
        let expected_job = record_id_to_string(job.job.id.as_ref().expect("persisted ID"));
        if job.phase == "retired" {
            return Err(DbError::RemoteJobSourceConflict(expected_job));
        }
        if job.source_generation.is_none() {
            // An older unprepared request cannot supersede a newer request
            // that already acquired this logical document. Keep the sequence
            // in the durable journal so deletion/recreation cannot reset it.
            // Use the same total order as scheduling, including legacy ties
            // and mixed histories, so an old resume cannot bypass the fence.
            let later = admission_precedes_sql(
                "$order",
                "<datetime>$created",
                "$job",
                "remote_admission_order",
                "created_at",
                "id",
            );
            let newer: Vec<RecordId> = self.db.query(format!("SELECT VALUE id FROM processing_job WHERE job_type = 'remote_upload' AND remote_instance_id = $instance AND remote_source_uri = $uri AND id != $job AND remote_source_generation IS NOT NONE AND {later} LIMIT 1"))
                .bind(("instance", job.instance_id.clone())).bind(("uri", job.source_uri.clone()))
                .bind(("job", job.job.id.clone().expect("persisted ID"))).bind(("order", job.admission_order))
                .bind(("created", job.job.created_at.to_rfc3339())).await?.take(0)?;
            if !newer.is_empty() {
                return Err(DbError::RemoteJobSourceConflict(expected_job));
            }
        }
        if let Some(generation) = job.source_generation {
            let source = self.get_source(&job.source_uri).await?.ok_or_else(|| {
                DbError::RemoteJobSourceConflict(record_id_to_string(
                    job.job.id.as_ref().expect("persisted ID"),
                ))
            })?;
            let preserves_enrichment_origin = job.job.scope.as_deref() == Some("unchanged")
                && super::source_enrichment::source_entity_enrichment_origin_current(&source);
            let origin = source
                .metadata
                .get("remote_upload_pending")
                .or_else(|| source.metadata.get("remote_upload"));
            if source.generation != generation
                || source.id != job.source_id
                || source.source_type != SourceType::Markdown
                || origin.is_none_or(|origin| {
                    (!preserves_enrichment_origin && origin["job_id"] != expected_job)
                        || origin["instance_id"] != job.instance_id
                        || origin["document_key"] != job.input.document_key
                })
            {
                return Err(DbError::RemoteJobSourceConflict(expected_job));
            }
        }
        Ok(())
    }
    pub async fn begin_remote_upload_generation(&self, lease: &RemoteJobLease) -> Result<Source> {
        let _gate = self.remote_job_transition_lock.lock().await;
        let _lifecycle = self.proposal_acceptance_lock.lock().await;
        let job = self.owned_remote_upload(lease, false).await?;
        self.ensure_remote_source_current(&job).await?;
        if job.source_generation.is_some() {
            return self.get_source(&job.source_uri).await?.ok_or_else(|| {
                DbError::RemoteJobSourceConflict(record_id_to_string(&lease.job_id))
            });
        }
        let input = &job.input;
        let source_key = digest(
            "graphrag-remote-upload-source-v1",
            &[&input.authenticated_instance_id, &input.document_key],
        );
        let source_id = RecordId::new("source", source_key);
        let prior = self.get_source(&job.source_uri).await?;
        // Revalidate a client's negative lookup under the generation lock. An
        // earlier queued upload may have published since that lookup.
        if input.create_only && prior.is_some() {
            return Err(DbError::RemoteJobSourceConflict(record_id_to_string(
                &lease.job_id,
            )));
        }
        let hash = graphrag_core::normalized_content_hash(&input.markdown);
        let mut prior_completed = false;
        let mut prior_exact_input_matches = false;
        let mut prior_revision_matches = input.expected_source_revision.is_none();
        if let Some(source) = &prior {
            if source.metadata["graph_enrichment_managed"] == true && !input.extract_entities {
                return Err(DbError::RemoteJobSourceConflict(record_id_to_string(
                    &lease.job_id,
                )));
            }
            let origin = source
                .metadata
                .get("remote_upload_pending")
                .unwrap_or(&source.metadata["remote_upload"]);
            if source.source_type != SourceType::Markdown
                || source.id.as_ref() != Some(&source_id)
                || origin["instance_id"] != input.authenticated_instance_id
                || origin["document_key"] != input.document_key
            {
                return Err(DbError::RemoteJobSourceConflict(record_id_to_string(
                    &lease.job_id,
                )));
            }
            if let Some(owner) = origin["job_id"].as_str() {
                if let Some(other) = self.get_remote_upload_job(&job.instance_id, owner).await? {
                    if matches!(other.job.status.as_str(), "running" | "queued") {
                        return Err(DbError::RemoteJobSourceConflict(record_id_to_string(
                            &lease.job_id,
                        )));
                    }
                    prior_completed = other.job.status == "completed";
                    if let Some(expected) = &input.expected_source_revision {
                        let migration_valid =
                            policy_migration::matches_origin(source, &other, input)?;
                        prior_revision_matches =
                            uploaded_source_revision(source, &other.input.markdown)? == *expected
                                && (migration_valid
                                    || (input.policy_migration.is_none()
                                        && uploaded_processing_compatible(
                                            if source.metadata["graph_enrichment_managed"] == true {
                                                &source.metadata["remote_upload"]
                                                    ["processing_options"]
                                            } else {
                                                &other.input.processing_options
                                            },
                                            &input.processing_options,
                                            input.extract_entities,
                                            input.preserve_unchanged,
                                        )));
                    }
                    let mut prior_provenance = other.input.source_provenance.clone();
                    let mut desired_provenance = input.source_provenance.clone();
                    for provenance in [&mut prior_provenance, &mut desired_provenance] {
                        if let Some(metadata) = provenance["metadata"].as_object_mut() {
                            metadata.remove("collection_id");
                        }
                    }
                    prior_exact_input_matches = other.input.markdown == input.markdown
                        && prior_provenance == desired_provenance;
                }
            }
        }
        if !prior_revision_matches {
            return Err(DbError::RemoteJobSourceConflict(record_id_to_string(
                &lease.job_id,
            )));
        }
        let unchanged = prior_completed
            && prior
                .as_ref()
                .is_some_and(|source| source.metadata["remote_upload_retired"] != true)
            && prior.as_ref().is_some_and(|source| {
                source.status == SourceIngestionStatus::Ready
                    && source.content_hash.as_deref() == Some(hash.as_str())
                    && source.title == input.title
                    && source.metadata["remote_upload"]["processing_options"]
                        == input.processing_options
                    && source.metadata["remote_upload"]["extract_entities"]
                        == input.extract_entities
            });
        if input.preserve_unchanged && (!unchanged || !prior_exact_input_matches) {
            return Err(DbError::RemoteJobSourceConflict(record_id_to_string(
                &lease.job_id,
            )));
        }
        let enriched_replacement = !unchanged
            && prior
                .as_ref()
                .is_some_and(|source| source.metadata["graph_enrichment_managed"] == true);
        if enriched_replacement && input.expected_source_revision.is_none() {
            return Err(DbError::InvalidRemoteRequest(
                "Replacing an enriched source requires its exact reviewed source revision".into(),
            ));
        }
        let generation = prior.as_ref().map_or(1, |source| {
            if unchanged {
                source.successful_generation
            } else {
                source.generation.saturating_add(1)
            }
        });
        let items = if unchanged {
            self.get_source_chunks(&source_id)
                .await?
                .into_iter()
                .map(|note| record_id_to_string(note.id.as_ref().expect("stored chunk ID")))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        if items.len() > MAX_REMOTE_UPLOAD_CHUNKS {
            return Err(DbError::InvalidRemoteRequest(
                "existing uploaded source exceeds the chunk bound".into(),
            ));
        }
        let action = if unchanged {
            "unchanged"
        } else if prior.is_some() {
            "updated"
        } else {
            "created"
        };
        let mut metadata = prior
            .as_ref()
            .map_or_else(|| serde_json::json!({}), |source| source.metadata.clone());
        if let Some(metadata) = metadata.as_object_mut() {
            metadata.remove("remote_upload_retired");
        }
        let preserves_enrichment_origin =
            unchanged && metadata.get("entity_enrichment_v1").is_some();
        if preserves_enrichment_origin
            && !prior
                .as_ref()
                .is_some_and(super::source_enrichment::source_entity_enrichment_origin_current)
        {
            return Err(DbError::RemoteJobSourceConflict(record_id_to_string(
                &lease.job_id,
            )));
        }
        if unchanged {
            if !preserves_enrichment_origin {
                metadata["remote_upload"] = serde_json::json!({"instance_id":job.instance_id,"document_key":input.document_key,"request_id":job.request_id,"source":input.source_provenance,"job_id":record_id_to_string(&lease.job_id),"processing_options":input.processing_options,"extract_entities":input.extract_entities});
            }
            if let Some(metadata) = metadata.as_object_mut() {
                metadata.remove("remote_upload_pending");
            }
        } else {
            let origin = serde_json::json!({"instance_id":job.instance_id,"document_key":input.document_key,"request_id":job.request_id,"source":input.source_provenance,"job_id":record_id_to_string(&lease.job_id),"processing_options":input.processing_options,"extract_entities":input.extract_entities});
            metadata["remote_upload_pending"] = origin;
            if let Some(fields) = metadata.as_object_mut() {
                fields.remove("entity_enrichment_v1");
            }
        }
        let source_sql: String = if unchanged {
            "UPDATE $source SET metadata = $metadata, updated_at = time::now(); ".into()
        } else {
            "UPSERT $source SET source_type = 'markdown', title = $title, uri = $uri, normalized_uri = $uri, content = $markdown, content_hash = $hash, generation = $generation, successful_generation = $successful, status = 'pending', last_error = NONE, metadata = $metadata, created_at = IF created_at = NONE THEN time::now() ELSE created_at END, updated_at = time::now(); ".into()
        };
        let checkpoint = items.last().cloned();
        let mut response = migration_probe("upload.begin_generation", "sdk_await", self.db.query(format!("BEGIN TRANSACTION; {}{} UPDATE $job SET remote_source_id = $source, remote_source_generation = $generation, source_generation = $generation_label, scope = $action, item_ids = $items, total_count = $total, completed_count = IF $unchanged THEN $total ELSE 0 END, checkpoint = $checkpoint, remote_phase = $phase, remote_migration_contract_version = IF $rebuild THEN 1 ELSE remote_migration_contract_version END; COMMIT TRANSACTION;", guard_sql(), source_sql))
            .bind(("job", lease.job_id.clone())).bind(("instance", lease.instance_id.clone())).bind(("epoch", lease.service_epoch.clone())).bind(("worker", lease.worker_token.clone()))
            .bind(("source", source_id)).bind(("title", input.title.clone())).bind(("uri", job.source_uri.clone())).bind(("markdown", input.markdown.clone())).bind(("hash", hash))
            .bind(("generation", generation as i64)).bind(("successful", prior.map_or(0, |source| source.successful_generation) as i64)).bind(("generation_label", generation.to_string()))
            .bind(("rebuild", enriched_replacement)).bind(("metadata", metadata)).bind(("action", action)).bind(("total", items.len() as i64)).bind(("items", items)).bind(("unchanged",unchanged)).bind(("checkpoint",checkpoint)).bind(("phase", if unchanged {"promoted"} else if input.policy_migration.is_some() || enriched_replacement {"migration_preparing"} else {"preparing"}))).await?;
        check_write(response.take_errors(), lease, "upload.begin_generation")?;
        self.get_source(&job.source_uri)
            .await?
            .ok_or_else(|| DbError::RemoteJobSourceConflict(record_id_to_string(&lease.job_id)))
    }
    pub async fn stage_remote_upload_notes(
        &self,
        lease: &RemoteJobLease,
        mut notes: Vec<Note>,
    ) -> Result<RemoteUploadJob> {
        if notes.len() > MAX_REMOTE_UPLOAD_CHUNKS
            || notes.iter().any(|note| {
                note.content.trim().is_empty()
                    || note.content.len() > MAX_REMOTE_UPLOAD_BYTES
                    || note.embedding.iter().any(|value| !value.is_finite())
            })
        {
            return Err(DbError::InvalidRemoteRequest(
                "prepared upload chunks exceed bounds or contain invalid vectors".into(),
            ));
        }
        let _gate = self.remote_job_transition_lock.lock().await;
        let _lifecycle = self.proposal_acceptance_lock.lock().await;
        let job = self.owned_remote_upload(lease, false).await?;
        self.ensure_remote_source_current(&job).await?;
        if matches!(
            job.phase.as_str(),
            "staged"
                | "promoted"
                | "extracting"
                | "migration_staged"
                | "migration_extracting"
                | "migration_promoted"
        ) {
            return Ok(job);
        }
        if !matches!(job.phase.as_str(), "preparing" | "migration_preparing") {
            return Err(DbError::RemoteJobOwnershipLost(record_id_to_string(
                &lease.job_id,
            )));
        }
        let source_id = job
            .source_id
            .clone()
            .ok_or_else(|| DbError::RemoteJobSourceConflict(record_id_to_string(&lease.job_id)))?;
        for (index, note) in notes.iter_mut().enumerate() {
            let index = index.to_string();
            note.id = Some(RecordId::new(
                "note",
                digest(
                    "graphrag-remote-upload-chunk-v1",
                    &[&job.instance_id, &job.request_id, &index],
                ),
            ));
            note.source_id = Some(source_id.clone());
            note.source_generation = job.source_generation;
            if note.search_content.is_none() {
                note.search_content = Some(derived_search_content(note));
            }
        }
        let ids = notes
            .iter()
            .map(|note| record_id_to_string(note.id.as_ref().expect("assigned ID")))
            .collect::<Vec<_>>();
        let total = notes.len() as i64;
        let mut response = migration_probe("upload.stage_notes", "sdk_await", self.db.query(format!("BEGIN TRANSACTION; {} FOR $item IN $notes {{ LET $note_id = $item.id; LET $content = IF array::len($item.embedding) = 0 THEN object::remove($item, 'embedding') ELSE $item END; CREATE $note_id CONTENT $content; }}; UPDATE $job SET item_ids = $ids, total_count = $total, completed_count = 0, failed_count = 0, checkpoint = NONE, remote_phase = $phase; COMMIT TRANSACTION;", guard_sql()))
            .bind(("job", lease.job_id.clone())).bind(("instance", lease.instance_id.clone())).bind(("epoch", lease.service_epoch.clone())).bind(("worker", lease.worker_token.clone()))
            .bind(("phase", if job.phase == "migration_preparing" { "migration_staged" } else { "staged" }))
            .bind(("notes", notes)).bind(("ids", ids)).bind(("total", total))).await?;
        check_write(response.take_errors(), lease, "upload.stage_notes")?;
        self.owned_remote_upload(lease, true).await
    }
    pub async fn reconcile_remote_upload(
        &self,
        lease: &RemoteJobLease,
        successors: &[(RecordId, RecordId, bool)],
    ) -> Result<SourceDeleteSummary> {
        let _gate = self.remote_job_transition_lock.lock().await;
        let _lifecycle = self.proposal_acceptance_lock.lock().await;
        let job = migration_probe(
            "reconcile",
            "owner_read",
            self.owned_remote_upload(lease, false),
        )
        .await?;
        migration_probe(
            "reconcile",
            "source_fence",
            self.ensure_remote_source_current(&job),
        )
        .await?;
        if !matches!(
            job.phase.as_str(),
            "staged" | "promoted" | "migration_promoted"
        ) {
            return Err(DbError::RemoteJobOwnershipLost(record_id_to_string(
                &lease.job_id,
            )));
        }
        let mut source = self
            .get_source(&job.source_uri)
            .await?
            .ok_or_else(|| DbError::RemoteJobSourceConflict(record_id_to_string(&lease.job_id)))?;
        let source_id = source
            .id
            .as_ref()
            .ok_or_else(|| DbError::RemoteJobSourceConflict(record_id_to_string(&lease.job_id)))?;
        let outcome = if source.successful_generation == source.generation {
            // A restart after visibility promotion must never copy dependents
            // from the now-hidden generation again. Cleanup is retry-safe.
            self.complete_file_import_locked(&mut source).await
        } else {
            for (old, new, _) in successors {
                let old_note: Option<Note> = self.db.select(old.clone()).await?;
                let new_note: Option<Note> = self.db.select(new.clone()).await?;
                if old_note.as_ref().is_none_or(|note| {
                    note.source_id.as_ref() != Some(source_id)
                        || note.source_generation != Some(source.successful_generation)
                }) || new_note.as_ref().is_none_or(|note| {
                    note.source_id.as_ref() != Some(source_id)
                        || note.source_generation != job.source_generation
                        || !job.job.item_ids.contains(&record_id_to_string(new))
                }) {
                    return Err(DbError::RemoteJobSourceConflict(record_id_to_string(
                        &lease.job_id,
                    )));
                }
            }
            self.copy_note_dependents_to_successors_locked(successors)
                .await?;
            self.complete_file_import_locked(&mut source).await
        };
        if source.successful_generation == source.generation {
            // Even deferred cleanup failure cannot erase a successful visibility
            // transition. Resume observes promoted and retries cleanup/extraction.
            self.update_remote_phase_locked(
                lease,
                if job.input.policy_migration.is_some() || job.phase == "migration_promoted" {
                    "migration_promoted"
                } else {
                    "promoted"
                },
                ProcessingJobUpdate::default(),
                true,
            )
            .await?;
        }
        outcome
    }
    /// Exact prepared chunks, including the hidden staged generation. Worker
    /// ownership and the pinned source generation are checked before reading.
    pub async fn remote_upload_notes(&self, lease: &RemoteJobLease) -> Result<Vec<Note>> {
        let _gate = self.remote_job_transition_lock.lock().await;
        let _lifecycle = self.proposal_acceptance_lock.lock().await;
        let job = self.owned_remote_upload(lease, false).await?;
        self.ensure_remote_source_current(&job).await?;
        let mut notes = Vec::with_capacity(job.job.item_ids.len());
        for id in &job.job.item_ids {
            let id = parse_record_id(id, Some("note"))?;
            let note: Note = self.db.select(id).await?.ok_or_else(|| {
                DbError::RemoteJobSourceConflict(record_id_to_string(&lease.job_id))
            })?;
            if note.source_id != job.source_id || note.source_generation != job.source_generation {
                return Err(DbError::RemoteJobSourceConflict(record_id_to_string(
                    &lease.job_id,
                )));
            }
            notes.push(note);
        }
        Ok(notes)
    }
    async fn update_remote_phase_locked(
        &self,
        lease: &RemoteJobLease,
        phase: &str,
        update: ProcessingJobUpdate,
        allow_cancel: bool,
    ) -> Result<RemoteUploadJob> {
        let job = self.owned_remote_upload(lease, allow_cancel).await?;
        self.ensure_remote_source_current(&job).await?;
        let completed = update
            .completed_count
            .unwrap_or(job.job.completed_count as u64);
        let failed = update.failed_count.unwrap_or(job.job.failed_count as u64);
        if update.status.is_some()
            || update.finish
            || completed < job.job.completed_count as u64
            || failed < job.job.failed_count as u64
            || completed.saturating_add(failed) > job.job.total_count as u64
        {
            return Err(DbError::InvalidRemoteRequest(
                "remote job checkpoints must be monotonic and bounded by their admitted items"
                    .into(),
            ));
        }
        let mut response = migration_probe("upload.phase_update", "sdk_await", self.db.query("UPDATE $job SET remote_phase = $phase, completed_count = $completed, failed_count = $failed, checkpoint = IF $checkpoint_set THEN $checkpoint ELSE checkpoint END, last_error = IF $error_set THEN $error ELSE last_error END, updated_at = time::now() WHERE job_type = 'remote_upload' AND remote_instance_id = $instance AND status = 'running' AND remote_service_epoch = $epoch AND remote_worker_token = $worker RETURN AFTER")
            .bind(("job", lease.job_id.clone())).bind(("instance", lease.instance_id.clone())).bind(("epoch", lease.service_epoch.clone())).bind(("worker", lease.worker_token.clone()))
            .bind(("phase", phase.to_string())).bind(("completed", count_to_i64(completed)?)).bind(("failed", count_to_i64(failed)?))
            .bind(("checkpoint_set", update.checkpoint.is_some())).bind(("checkpoint", update.checkpoint.flatten())).bind(("error_set", update.last_error.is_some())).bind(("error", update.last_error.flatten()))).await?;
        let row: Option<JobRow> =
            migration_probe_result("upload.phase_update", "response_take", response.take(0))?;
        let row = migration_probe_result(
            "upload.phase_update",
            "row_presence",
            row.ok_or_else(|| DbError::RemoteJobOwnershipLost(record_id_to_string(&lease.job_id))),
        )?;
        migration_probe_result("upload.phase_update", "public_decode", row.public())
    }
    pub async fn checkpoint_remote_upload_job(
        &self,
        lease: &RemoteJobLease,
        phase: &str,
        update: ProcessingJobUpdate,
    ) -> Result<RemoteUploadJob> {
        let _gate = self.remote_job_transition_lock.lock().await;
        let job = self.owned_remote_upload(lease, false).await?;
        if phase != "extracting" || !matches!(job.phase.as_str(), "promoted" | "extracting") {
            return Err(DbError::RemoteJobOwnershipLost(record_id_to_string(
                &lease.job_id,
            )));
        }
        self.update_remote_phase_locked(lease, phase, update, false)
            .await
    }
    pub async fn persist_remote_upload_entities(
        &self,
        lease: &RemoteJobLease,
        item_index: usize,
        entities: Vec<Entity>,
    ) -> Result<RemoteUploadJob> {
        let _gate = self.remote_job_transition_lock.lock().await;
        let job = self.owned_remote_upload(lease, false).await?;
        self.ensure_remote_source_current(&job).await?;
        if !matches!(job.phase.as_str(), "promoted" | "extracting") {
            return Err(DbError::RemoteJobOwnershipLost(record_id_to_string(
                &lease.job_id,
            )));
        }
        let done = (job.job.completed_count + job.job.failed_count) as usize;
        if item_index < done {
            return Ok(job);
        }
        if item_index != done || item_index >= job.job.item_ids.len() {
            return Err(DbError::InvalidRemoteRequest(
                "extraction checkpoint must advance its exact next item".into(),
            ));
        }
        let note_id = parse_record_id(&job.job.item_ids[item_index], Some("note"))?;
        let _lifecycle = self.proposal_acceptance_lock.lock().await;
        self.ensure_remote_source_current(&job).await?;
        let note: Option<Note> = self.db.select(note_id.clone()).await?;
        let extraction_scope = match note {
            Some(note) => self.note_extraction_scope(&note).await?,
            None => None,
        };
        let entity_names = entities
            .iter()
            .map(Entity::effective_identity_key)
            .collect::<Vec<_>>();
        let mut response = migration_probe("upload.persist_entities", "sdk_await", self.db.query(format!("BEGIN TRANSACTION; {} LET $selected = (SELECT VALUE id FROM note WHERE id = $note AND source_id = $source AND source_generation = $generation AND source_generation = source_id.successful_generation); IF array::len($selected) != 1 {{ THROW '{FENCE}'; }}; {} IF $extraction_scope != NONE {{ UPDATE $note SET extraction_scope = $extraction_scope; }}; DELETE mentions WHERE in = $note; {} UPDATE $job SET remote_phase = 'extracting', completed_count += 1, checkpoint = $checkpoint; COMMIT TRANSACTION;", guard_sql(), super::notes::replacement_entities_transaction(), super::notes::replacement_mentions_transaction("$note")))
            .bind(("job", lease.job_id.clone())).bind(("instance", lease.instance_id.clone())).bind(("epoch", lease.service_epoch.clone())).bind(("worker", lease.worker_token.clone()))
            .bind(("extraction_scope", extraction_scope)).bind(("note", note_id)).bind(("source", job.source_id.clone())).bind(("generation", job.source_generation)).bind(("checkpoint", job.job.item_ids[item_index].clone())).bind(("replacement_entities", entities)).bind(("replacement_entity_names", entity_names))).await?;
        check_write(response.take_errors(), lease, "upload.persist_entities")?;
        self.owned_remote_upload(lease, true).await
    }
    pub async fn finish_remote_upload_job(
        &self,
        lease: &RemoteJobLease,
        status: ProcessingJobStatus,
        error: Option<String>,
        result: Option<serde_json::Value>,
    ) -> Result<RemoteUploadJob> {
        if !matches!(
            status,
            ProcessingJobStatus::Completed
                | ProcessingJobStatus::Failed
                | ProcessingJobStatus::Cancelled
        ) || result.as_ref().is_some_and(|value| !value.is_object())
            || error.as_ref().is_some_and(|value| value.len() > 512)
            || result.as_ref().is_some_and(|value| {
                serde_json::to_vec(value)
                    .map_or(true, |bytes| bytes.len() > MAX_REMOTE_UPLOAD_BYTES)
            })
        {
            return Err(DbError::InvalidRemoteRequest(
                "remote job finish requires a terminal status and object result".into(),
            ));
        }
        let _gate = self.remote_job_transition_lock.lock().await;
        let job = self.owned_remote_upload(lease, true).await?;
        let _lifecycle = self.proposal_acceptance_lock.lock().await;
        if status == ProcessingJobStatus::Completed && !job.cancel_requested {
            self.ensure_remote_source_current(&job).await?;
            if !matches!(
                job.phase.as_str(),
                "promoted" | "extracting" | "migration_promoted"
            ) {
                return Err(DbError::RemoteJobOwnershipLost(record_id_to_string(
                    &lease.job_id,
                )));
            }
            if job.input.extract_entities
                && job.job.completed_count + job.job.failed_count != job.job.total_count
            {
                return Err(DbError::InvalidRemoteRequest(
                    "entity extraction has unprocessed items".into(),
                ));
            }
        }
        // Cancellation is independent of both mutexes. Resolve its persisted
        // flag in the same UPDATE as terminalization, never from the earlier
        // ownership snapshot taken before waiting for the lifecycle gate.
        let mut response = migration_probe("upload.finish", "sdk_await", self.db.query("UPDATE $job SET status = IF remote_cancel_requested THEN 'cancelled' ELSE $status END, remote_phase = IF remote_cancel_requested = false AND $status = 'completed' THEN 'completed' ELSE remote_phase END, completed_count = IF remote_cancel_requested = false AND $status = 'completed' AND $skip_extract THEN total_count ELSE completed_count END, last_error = IF remote_cancel_requested THEN 'cancelled' ELSE $error END, remote_result = IF remote_result.policy_migration_stage != NONE AND (remote_cancel_requested OR $status != 'completed') THEN remote_result ELSE (IF remote_cancel_requested THEN NONE ELSE $result END) END, finished_at = time::now(), updated_at = time::now(), remote_service_epoch = NONE, remote_worker_token = NONE WHERE job_type = 'remote_upload' AND remote_instance_id = $instance AND status = 'running' AND remote_service_epoch = $epoch AND remote_worker_token = $worker RETURN AFTER")
            .bind(("job", lease.job_id.clone())).bind(("instance", lease.instance_id.clone())).bind(("epoch", lease.service_epoch.clone())).bind(("worker", lease.worker_token.clone()))
            .bind(("status", status.as_str())).bind(("skip_extract", !job.input.extract_entities)).bind(("error", error)).bind(("result", result))).await?;
        let row: Option<JobRow> =
            migration_probe_result("upload.finish", "response_take", response.take(0))?;
        let row = migration_probe_result(
            "upload.finish",
            "row_presence",
            row.ok_or_else(|| DbError::RemoteJobOwnershipLost(record_id_to_string(&lease.job_id))),
        )?;
        migration_probe_result("upload.finish", "public_decode", row.public())
    }
    async fn remote_upload_recovery_fence(
        &self,
        lease: &RemoteJobLease,
    ) -> Result<Option<RecoveryFence>> {
        Ok(self.db.query("SELECT status, remote_service_epoch, remote_worker_token, remote_cancel_requested FROM processing_job WHERE id = $job AND job_type = 'remote_upload' AND remote_instance_id = $instance LIMIT 1")
            .bind(("job", lease.job_id.clone())).bind(("instance", lease.instance_id.clone())).await?.take(0)?)
    }

    /// Settle a failed execution using only its durable ownership fence. Saved
    /// input/result decoding is deliberately excluded, so a damaged payload
    /// cannot strand a committed running claim. Storage errors remain retryable.
    pub async fn recover_remote_upload_job(
        &self,
        lease: &RemoteJobLease,
        error_code: &str,
    ) -> Result<()> {
        identity(&lease.instance_id)?;
        identity(&lease.service_epoch)?;
        identity(&lease.worker_token)?;
        if error_code.is_empty()
            || error_code.len() > 512
            || error_code.chars().any(char::is_control)
        {
            return Err(DbError::InvalidRemoteRequest(
                "invalid recovery error category".into(),
            ));
        }
        let _gate = self.remote_job_transition_lock.lock().await;
        let Some(fence) = self.remote_upload_recovery_fence(lease).await? else {
            return Ok(());
        };
        if fence.status != "running"
            || fence.remote_service_epoch.as_deref() != Some(&lease.service_epoch)
            || fence.remote_worker_token.as_deref() != Some(&lease.worker_token)
        {
            return Ok(());
        }
        // The read validates the minimal persisted boolean, while the UPDATE
        // below resolves it again atomically after waiting for lifecycle work.
        let _cancel_was_requested = fence.remote_cancel_requested;
        let _lifecycle = self.proposal_acceptance_lock.lock().await;
        let mut response = migration_probe("upload.recover_settle", "sdk_await", self.db.query("UPDATE $job SET status = IF remote_cancel_requested THEN 'cancelled' ELSE 'failed' END, last_error = IF remote_cancel_requested THEN 'cancelled' ELSE (IF $error = 'cancelled' THEN 'interrupted' ELSE $error END) END, remote_result = IF remote_result.policy_migration_stage != NONE THEN remote_result ELSE NONE END, finished_at = time::now(), updated_at = time::now(), remote_service_epoch = NONE, remote_worker_token = NONE WHERE job_type = 'remote_upload' AND remote_instance_id = $instance AND status = 'running' AND remote_service_epoch = $epoch AND remote_worker_token = $worker RETURN VALUE id")
            .bind(("job", lease.job_id.clone())).bind(("instance", lease.instance_id.clone())).bind(("epoch", lease.service_epoch.clone())).bind(("worker", lease.worker_token.clone())).bind(("error", error_code.to_string()))).await?;
        let settled: Vec<RecordId> =
            migration_probe_result("upload.recover_settle", "response_take", response.take(0))?;
        if settled.is_empty() {
            // A missing/changed owner is safe; a still-owned fence must never
            // be mistaken for successful settlement (for example denied writes).
            if self
                .remote_upload_recovery_fence(lease)
                .await?
                .is_some_and(|current| {
                    current.status == "running"
                        && current.remote_service_epoch.as_deref() == Some(&lease.service_epoch)
                        && current.remote_worker_token.as_deref() == Some(&lease.worker_token)
                })
            {
                return Err(DbError::QueryFailed(
                    "remote upload terminalization did not settle its owned fence".into(),
                ));
            }
        }
        Ok(())
    }

    pub async fn reconcile_interrupted_remote_uploads(&self, current_epoch: &str) -> Result<usize> {
        identity(current_epoch)?;
        let _gate = self.remote_job_transition_lock.lock().await;
        let rows: Vec<RecordId> = self.db.query("UPDATE processing_job SET status = IF remote_cancel_requested THEN 'cancelled' ELSE (IF remote_input.enrichment != NONE THEN 'queued' ELSE 'failed' END) END, last_error = IF remote_cancel_requested THEN 'cancelled' ELSE 'interrupted' END, finished_at = time::now(), updated_at = time::now(), remote_service_epoch = NONE, remote_worker_token = NONE WHERE job_type = 'remote_upload' AND remote_instance_id IS NOT NONE AND status = 'running' AND (remote_service_epoch = NONE OR remote_service_epoch != $epoch) RETURN VALUE id")
            .bind(("epoch", current_epoch.to_string())).await?.take(0)?;
        Ok(rows.len())
    }
}
