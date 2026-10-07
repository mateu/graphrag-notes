//! Explicit reviewed enrichment transport. DEVELOPMENT ONLY; no scans or automatic grants.
use crate::remote_jobs::{lease, options, source};
use crate::*;
use graphrag_db::{RemoteEnrichmentInput, RemoteUploadInput, SourceEnrichmentPlan};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PrepareSourceEnrichment {
    pub request_id: String,
    pub source_id: String,
    pub revision: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReviewedEnrichmentReceipt {
    pub version: u32,
    pub plan: Value,
    pub plan_sha256: String,
    pub target_policy_sha256: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExecuteSourceEnrichment {
    /// Separate operation identity (rollback must use a new identity).
    pub request_id: String,
    pub reviewed: ReviewedEnrichmentReceipt,
    pub confirmed: bool,
    #[serde(default)]
    pub rollback_source_revision: Option<String>,
}
pub fn enrichment_receipt(
    plan: &SourceEnrichmentPlan,
) -> ApplicationResult<ReviewedEnrichmentReceipt> {
    let plan =
        serde_json::to_value(plan).map_err(|e| ApplicationError::Validation(e.to_string()))?;
    Ok(ReviewedEnrichmentReceipt {
        version: 1,
        plan_sha256: format!("{:x}", Sha256::digest(serde_json::to_vec(&plan).unwrap())),
        target_policy_sha256: graphrag_db::uploaded_processing_policy_sha256(
            &plan["target_processing_options"],
        )?,
        plan,
    })
}
fn validated(
    caller: &CallerIdentity,
    request: &ExecuteSourceEnrichment,
) -> ApplicationResult<SourceEnrichmentPlan> {
    let plan: SourceEnrichmentPlan = serde_json::from_value(request.reviewed.plan.clone())
        .map_err(|_| {
            ApplicationError::Validation("Exact reviewed enrichment plan required".into())
        })?;
    if !request.confirmed
        || plan.instance_id != caller.instance_id
        || enrichment_receipt(&plan)?.plan_sha256 != request.reviewed.plan_sha256
        || enrichment_receipt(&plan)?.target_policy_sha256 != request.reviewed.target_policy_sha256
        || request.reviewed.version != 1
    {
        return Err(ApplicationError::Validation(
            "Confirmed owner-bound exact reviewed plan and policy digests required".into(),
        ));
    }
    Ok(plan)
}
pub(crate) async fn prepare(
    app: &EmbeddedApplication,
    caller: CallerIdentity,
    request: PrepareSourceEnrichment,
) -> ApplicationResult<ReviewedEnrichmentReceipt> {
    let current = source(app, &request.source_id).await?;
    if current.instance_id != caller.instance_id || current.revision != request.revision {
        return Err(ApplicationError::RevisionConflict(
            "Owned current source revision required".into(),
        ));
    }
    let plan = SourceEnrichmentPlan {
        instance_id: caller.instance_id,
        request_id: request.request_id,
        source_id: current.id,
        document_key: current.document_key,
        expected_source_revision: current.revision,
        expected_content_sha256: current
            .content_hash
            .ok_or_else(|| ApplicationError::Validation("Ready content required".into()))?,
        expected_generation: current.generation,
        original_policy_sha256: current.processing_policy_sha256,
        target_processing_options: options(app),
        confirmed: true,
    };
    app.repo
        .preflight_source_entity_enrichment(plan.clone())
        .await?;
    enrichment_receipt(&plan)
}
pub(crate) async fn admit(
    app: &EmbeddedApplication,
    caller: CallerIdentity,
    request: ExecuteSourceEnrichment,
    rollback: bool,
) -> ApplicationResult<UploadAdmission> {
    let plan = validated(&caller, &request)?;
    if (!rollback && request.request_id != plan.request_id)
        || (rollback && request.request_id == plan.request_id)
    {
        return Err(ApplicationError::Validation(
            "Conversion must use the reviewed identity and rollback a distinct identity".into(),
        ));
    }
    let fingerprint = format!(
        "{:x}",
        Sha256::digest(
            serde_json::to_vec(&json!(["reviewed-enrichment-v1", rollback, &request])).unwrap()
        )
    );
    if let Some(saved) = app
        .repo
        .find_remote_upload_admission(&caller.instance_id, &request.request_id, &fingerprint)
        .await?
    {
        return admission(&request.request_id, saved);
    }
    if plan.target_processing_options != options(app) {
        return Err(ApplicationError::Compatibility(
            "Restore exact reviewed target configuration".into(),
        ));
    }
    let current = source(app, &plan.source_id).await?;
    if current.instance_id != caller.instance_id {
        return Err(ApplicationError::Validation("Owned source required".into()));
    }
    if rollback && request.rollback_source_revision.as_deref() != Some(current.revision.as_str()) {
        return Err(ApplicationError::RevisionConflict(
            "Review exact current source revision before rollback".into(),
        ));
    }
    if !rollback {
        app.repo
            .preflight_source_entity_enrichment(plan.clone())
            .await?;
    } else {
        app.repo.inspect_source_entity_enrichment(&plan).await?;
    }
    let saved = app
        .repo
        .admit_remote_upload(RemoteUploadInput {
            authenticated_instance_id: caller.instance_id,
            request_id: request.request_id.clone(),
            payload_fingerprint: fingerprint,
            document_key: plan.document_key.clone(),
            markdown: current.content,
            title: current.title,
            source_provenance: current.provenance,
            extract_entities: true,
            preserve_unchanged: false,
            create_only: false,
            expected_source_revision: Some(plan.expected_source_revision.clone()),
            policy_migration: None,
            processing_options: plan.target_processing_options.clone(),
            enrichment: Some(RemoteEnrichmentInput { plan, rollback }),
        })
        .await?;
    admission(&request.request_id, saved)
}
fn admission(
    request: &str,
    saved: graphrag_db::RemoteJobAdmission,
) -> ApplicationResult<UploadAdmission> {
    Ok(UploadAdmission {
        request_id: request.into(),
        job_id: saved.result["job_id"].as_str().unwrap_or_default().into(),
        source_id: saved.result["source_id"]
            .as_str()
            .unwrap_or_default()
            .into(),
        source_uri: saved.result["source_uri"]
            .as_str()
            .unwrap_or_default()
            .into(),
        replayed: saved.replayed,
    })
}
pub(crate) async fn read_plan(
    app: &EmbeddedApplication,
    caller: CallerIdentity,
    id: &str,
) -> ApplicationResult<ReviewedEnrichmentReceipt> {
    let job = app
        .repo
        .get_remote_upload_job(&caller.instance_id, id)
        .await?
        .ok_or_else(|| ApplicationError::NotFound("Owned enrichment job required".into()))?;
    let input = job
        .input
        .enrichment
        .ok_or_else(|| ApplicationError::Validation("Enrichment job required".into()))?;
    enrichment_receipt(&input.plan)
}
pub(crate) async fn execute(
    app: &EmbeddedApplication,
    execution: &RemoteJobExecution,
    cancel: ActionCancellation,
) -> ApplicationResult<()> {
    let lease = lease(execution)?;
    let job = app.repo.owned_remote_upload_job(&lease).await?;
    let input = job
        .input
        .enrichment
        .ok_or_else(|| ApplicationError::Validation("Enrichment input required".into()))?;
    if input.plan.target_processing_options != options(app) {
        return Err(ApplicationError::Compatibility(
            "Restore exact reviewed policy before recovery".into(),
        ));
    }
    if input.rollback {
        let _worker = app.repo.source_entity_enrichment_worker_guard().await;
        app.repo
            .rollback_source_entity_enrichment_leased(&input.plan, true, Some(&lease))
            .await?;
    } else {
        app.enrich_uploaded_source_leased(
            CallerIdentity {
                instance_id: execution.instance_id.clone(),
            },
            input.plan,
            cancel,
            Some(&lease),
        )
        .await?;
    }
    Ok(())
}
