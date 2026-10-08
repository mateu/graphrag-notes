//! Durable upload admission and owned worker execution, independent of clients.
use crate::{
    inference::{CancellableEmbedder, CancellableExtractor},
    *,
};
use graphrag_agents::{markdown_chunk_successors, LibrarianAgent};
use graphrag_core::record_id_to_string;
use graphrag_db::{
    parse_record_id, ProcessingJobStatus, ProcessingJobUpdate, RemoteJobLease, RemoteUploadInput,
    RemoteUploadJob, RemoteUploadJobStatus,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{future::Future, sync::Arc};

/// Frozen upload journal identity; unrelated local API changes never invalidate retries.
const REMOTE_UPLOAD_PAYLOAD_VERSION: u32 = 1;

/// Runs the final preflight and then enters the non-cancellable publication region.
///
/// Source-enrichment cancellation is cooperative. Once the preflight succeeds, the
/// repository promotion is allowed to finish so a committed graph-policy epoch is
/// reported as success rather than being misreported as a cancelled conversion.
/// A caller whose future is interrupted while publication is in flight must inspect
/// the persisted enrichment status and retry or explicitly roll back; it cannot
/// infer whether the transaction committed from the dropped future.
async fn promote_after_final_preflight<T, P, F, Fut>(
    preflight: P,
    promote: F,
) -> ApplicationResult<T>
where
    P: FnOnce() -> ApplicationResult<()>,
    F: FnOnce() -> Fut,
    Fut: Future<Output = ApplicationResult<T>>,
{
    preflight()?;
    // This is entry into non-cancellable publication, not the datastore commit. Do not add a cancellation check below
    // it: cancellation arriving now must not turn a committed promotion into a
    // misleading cancelled outcome.
    promote().await
}

fn job_id(id: &str) -> ApplicationResult<()> {
    if id.len() > 512 || id.chars().any(char::is_control) || !id.starts_with("processing_job:") {
        return Err(ApplicationError::Validation(
            "Use the canonical processing_job:ID returned by upload_source".into(),
        ));
    }
    parse_record_id(id, Some("processing_job"))?;
    Ok(())
}

pub(crate) fn options(app: &EmbeddedApplication) -> Value {
    let embedding = app.embedder.capabilities();
    let extraction = app.extractor.capabilities();
    // Runtime contains usize::MAX for an unlimited chunk bound. Persist its
    // canonical JSON as text rather than coercing it into Surreal's signed
    // integer range; the exact snapshot must round-trip for safe resume.
    let mut snapshot = json!({"runtime": serde_json::to_string(&app.runtime).expect("runtime configuration serializes"), "embedding": {"provider":embedding.provider,"model":embedding.model,"cache_identity":embedding.cache_identity}, "extraction":{"provider":extraction.provider,"model":extraction.model,"cache_identity":extraction.cache_identity}});
    // Endpoints distinguish semantic backends, even under identical model
    // names. Persist only their identity: URLs can contain private hosts or
    // credentials, and these snapshots travel in portable archives.
    snapshot["embedding"]["endpoint_identity"] = json!(endpoint_identity(&embedding.endpoint));
    snapshot["extraction"]["endpoint_identity"] = json!(endpoint_identity(&extraction.endpoint));
    snapshot["extraction"]["cache_version"] = json!(graphrag_agents::extraction_cache_version());
    if let Some(dimension) = embedding.known_dimension {
        snapshot["embedding"]["dimension"] = json!(dimension.to_string());
    }
    snapshot
}

fn endpoint_identity(endpoint: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(b"graphrag-remote-upload-endpoint-v1\0");
    digest.update(endpoint.as_bytes());
    format!("{:x}", digest.finalize())
}

fn compatible(app: &EmbeddedApplication, job: &RemoteUploadJob) -> ApplicationResult<()> {
    if job.input.policy_migration.as_ref().is_some_and(|intent| {
        graphrag_db::uploaded_processing_policy_sha256(&options(app))
            .ok()
            .as_deref()
            != intent["target_policy_sha256"].as_str()
    }) {
        return Err(ApplicationError::Compatibility("Reviewed migration target policy changed; restore the exact configured policy before resuming".into()));
    }
    if !graphrag_db::uploaded_processing_compatible(
        &job.input.processing_options,
        &options(app),
        job.input.extract_entities,
        job.input.preserve_unchanged,
    ) {
        return Err(ApplicationError::Compatibility("Uploaded job configuration changed or lacks provider identity; restore its server endpoint/model/chunk settings before resuming, or submit a new upload request".into()));
    }
    Ok(())
}

pub(crate) fn view(job: RemoteUploadJobStatus) -> ApplicationResult<RemoteJobStatus> {
    Ok(RemoteJobStatus {
        id: record_id_to_string(
            job.job
                .id
                .as_ref()
                .ok_or_else(|| ApplicationError::Internal("Job identity missing".into()))?,
        ),
        job_type: match job.enrichment {
            Some(true) => "remote_enrichment".into(),
            Some(false) => job.job.job_type,
            None => "remote_unknown".into(),
        },
        instance_id: job.instance_id,
        status: job.job.status,
        phase: job.phase,
        cancellation_requested: job.cancel_requested,
        source_id: job
            .source_id
            .as_ref()
            .map(record_id_to_string)
            .unwrap_or_else(|| {
                job.admission["source_id"]
                    .as_str()
                    .unwrap_or_default()
                    .into()
            }),
        generation: job.source_generation,
        total: job.job.total_count.max(0) as u64,
        completed: job.job.completed_count.max(0) as u64,
        failed: job.job.failed_count.max(0) as u64,
        checkpoint: job.job.checkpoint,
        result: job.result.filter(|result| {
            result.get("policy_migration_stage").is_none()
                && result.get("unchanged_source_revision").is_none()
        }),
        error_code: job.job.last_error.map(|code| match code.as_str() {
            "interrupted"
            | "worker_interrupted"
            | "cancelled"
            | "validation"
            | "not_found"
            | "conflict"
            | "provider_unavailable"
            | "compatibility"
            | "service_unreachable"
            | "source_retired"
            | "internal" => code,
            _ => "internal".into(),
        }),
        created_at: job.job.created_at.to_rfc3339(),
        updated_at: job.job.updated_at.to_rfc3339(),
    })
}

pub(crate) fn lease(execution: &RemoteJobExecution) -> ApplicationResult<RemoteJobLease> {
    job_id(&execution.job_id)?;
    Ok(RemoteJobLease {
        job_id: parse_record_id(&execution.job_id, Some("processing_job"))?,
        instance_id: execution.instance_id.clone(),
        service_epoch: execution.service_epoch.clone(),
        worker_token: execution.worker_token.clone(),
    })
}

pub(crate) async fn upload(
    app: &EmbeddedApplication,
    caller: CallerIdentity,
    request: UploadSourceRequest,
) -> ApplicationResult<UploadAdmission> {
    let capture = RemoteCaptureRequest {
        request_id: request.request_id.clone(),
        content: request.content.clone(),
        title: request.title.clone(),
        tags: vec![],
        provenance: request.provenance.clone(),
    };
    super::remote_operations::validate_capture(&caller, &capture)?;
    if request.document_key.trim().is_empty()
        || request.document_key.trim() != request.document_key
        || request.document_key.chars().count() > 256
        || request.document_key.len() > 512
        || request.document_key.chars().any(char::is_control)
        || request.content.contains('\0')
    {
        return Err(ApplicationError::Validation("document_key must be nonempty, at most 256 characters/512 UTF-8 bytes, without controls or surrounding whitespace; content cannot contain NUL".into()));
    }
    if request.create_only
        && (request.preserve_unchanged || request.expected_source_revision.is_some())
    {
        return Err(ApplicationError::Validation(
            "create_only cannot accompany preserve_unchanged or expected_source_revision".into(),
        ));
    }
    if request
        .expected_source_revision
        .as_ref()
        .is_some_and(|revision| {
            revision.len() != 64
                || !revision
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        })
    {
        return Err(ApplicationError::Validation(
            "expected_source_revision must be lowercase SHA-256".into(),
        ));
    }
    if let Some(intent) = &request.policy_migration {
        let valid = [
            &intent.original_policy_sha256,
            &intent.target_policy_sha256,
            &intent.plan_sha256,
        ]
        .iter()
        .all(|digest| {
            digest.len() == 64
                && digest
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        });
        if !valid
            || !request.extract_entities
            || request.preserve_unchanged
            || request.create_only
            || request.expected_source_revision.is_none()
            || intent.original_policy_sha256 == intent.target_policy_sha256
        {
            return Err(ApplicationError::Validation("Policy migration requires distinct exact old/target/plan digests, extraction and an existing source revision".into()));
        }
    }
    // Keep legacy unguarded fingerprint bytes immutable for durable replay.
    let mut fingerprint_input = serde_json::to_vec(&(
        REMOTE_UPLOAD_PAYLOAD_VERSION,
        "uploaded_markdown",
        &request.document_key,
        &request.content,
        &request.title,
        &request.provenance,
        request.extract_entities,
    ))
    .map_err(|e| ApplicationError::Internal(e.to_string()))?;
    if request.preserve_unchanged {
        fingerprint_input.extend_from_slice(b"\0preserve_unchanged");
    }
    if request.create_only {
        fingerprint_input.extend_from_slice(b"\0create_only");
    }
    if let Some(revision) = &request.expected_source_revision {
        fingerprint_input.extend_from_slice(b"\0expected_source_revision\0");
        fingerprint_input.extend_from_slice(revision.as_bytes());
    }
    if let Some(intent) = &request.policy_migration {
        fingerprint_input.extend_from_slice(b"\0policy_migration\0");
        fingerprint_input.extend_from_slice(
            &serde_json::to_vec(intent)
                .map_err(|error| ApplicationError::Internal(error.to_string()))?,
        );
    }
    let fingerprint = format!("{:x}", Sha256::digest(&fingerprint_input));
    let admission = if let Some(admission) = app
        .repo
        .find_remote_upload_admission(&caller.instance_id, &request.request_id, &fingerprint)
        .await?
    {
        admission
    } else {
        if let Some(intent) = &request.policy_migration {
            if intent.target_policy_sha256
                != graphrag_db::uploaded_processing_policy_sha256(&options(app))?
            {
                return Err(ApplicationError::Compatibility(
                    "Reviewed migration target policy changed before admission; preview again"
                        .into(),
                ));
            }
        }
        let preview = LibrarianAgent::new(
            app.repo.clone(),
            app.embedder.clone(),
            app.extractor.clone(),
        )
        .with_runtime_config(app.runtime.clone())
        .preview_markdown_chunks(&request.content, &request.document_key)
        .map_err(|_| {
            ApplicationError::Validation(
                "Upload cannot be prepared under the server's Markdown chunk configuration".into(),
            )
        })?;
        if preview.len() > MAX_REMOTE_UPLOAD_CHUNKS {
            return Err(ApplicationError::Validation(
                "Uploaded Markdown exceeds 200 chunks under server settings".into(),
            ));
        }
        let mut processing_options = options(app);
        // Retain the original snapshot for supplied-source metadata registration
        // and unused extraction settings. Durable generation guards revalidate
        // source ownership/revision/input before publishing any change.
        if request.preserve_unchanged
            || (!request.extract_entities && request.expected_source_revision.is_some())
        {
            let id = graphrag_db::repository::uploaded_source_id(
                &caller.instance_id,
                &request.document_key,
            );
            if let Some(source) = app.repo.get_source(&id).await? {
                let origin = &source.metadata["remote_upload"];
                let prior = &origin["processing_options"];
                if origin["instance_id"] == caller.instance_id
                    && origin["document_key"] == request.document_key
                    && origin["extract_entities"] == request.extract_entities
                    && graphrag_db::uploaded_processing_compatible(
                        prior,
                        &processing_options,
                        request.extract_entities,
                        request.preserve_unchanged,
                    )
                {
                    processing_options = prior.clone();
                }
            }
        }
        app.repo
            .admit_remote_upload(RemoteUploadInput {
                authenticated_instance_id: caller.instance_id,
                request_id: request.request_id.clone(),
                payload_fingerprint: fingerprint,
                document_key: request.document_key,
                markdown: request.content,
                title: request.title,
                source_provenance: serde_json::to_value(request.provenance)
                    .map_err(|e| ApplicationError::Internal(e.to_string()))?,
                extract_entities: request.extract_entities,
                preserve_unchanged: request.preserve_unchanged,
                create_only: request.create_only,
                expected_source_revision: request.expected_source_revision,
                policy_migration: request
                    .policy_migration
                    .map(serde_json::to_value)
                    .transpose()
                    .map_err(|error| ApplicationError::Internal(error.to_string()))?,
                enrichment: None,
                processing_options,
            })
            .await?
    };
    Ok(UploadAdmission {
        request_id: request.request_id,
        job_id: admission.result["job_id"]
            .as_str()
            .ok_or_else(|| ApplicationError::Internal("Admission job ID missing".into()))?
            .into(),
        source_id: admission.result["source_id"]
            .as_str()
            .ok_or_else(|| ApplicationError::Internal("Admission source ID missing".into()))?
            .into(),
        source_uri: admission.result["source_uri"]
            .as_str()
            .ok_or_else(|| ApplicationError::Internal("Admission URI missing".into()))?
            .into(),
        replayed: admission.replayed,
    })
}

pub(crate) async fn source(
    app: &EmbeddedApplication,
    id: &str,
) -> ApplicationResult<UploadedSource> {
    if id.len() > 512 || id.chars().any(char::is_control) || !id.starts_with("source:") {
        return Err(ApplicationError::Validation(
            "Use the canonical source:ID returned by upload_source".into(),
        ));
    }
    let source = app.repo.get_source(id).await?.ok_or_else(|| {
        ApplicationError::NotFound(
            "Uploaded source has not been prepared yet; inspect its job".into(),
        )
    })?;
    let origin = source
        .metadata
        .get("remote_upload_pending")
        .or_else(|| source.metadata.get("remote_upload"))
        .ok_or_else(|| ApplicationError::NotFound("This is not an uploaded source".into()))?;
    let uri = source
        .normalized_uri
        .as_ref()
        .or(source.uri.as_ref())
        .filter(|uri| uri.starts_with("mcp://upload/"))
        .ok_or_else(|| ApplicationError::NotFound("This is not an uploaded source".into()))?
        .clone();
    let instance = origin["instance_id"]
        .as_str()
        .ok_or_else(|| ApplicationError::Internal("Uploaded source owner missing".into()))?;
    let origin_job_id = origin["job_id"].as_str().ok_or_else(|| {
        ApplicationError::Internal("Uploaded source input history missing".into())
    })?;
    let origin_job = app
        .repo
        .get_remote_upload_job(instance, origin_job_id)
        .await?
        .ok_or_else(|| {
            ApplicationError::NotFound(
                "Uploaded source input history missing; inspect its job".into(),
            )
        })?;
    if origin_job.source_uri != uri
        || origin_job.input.document_key != origin["document_key"].as_str().unwrap_or_default()
        || origin_job.source_id != source.id
    {
        return Err(ApplicationError::RevisionConflict(
            "Uploaded source input history has a different owner".into(),
        ));
    }
    // Normalized unchanged refreshes retain backing text for existing byte
    // spans. The public source contract returns the exact latest supplied input.
    let content = origin_job.input.markdown;
    let revision = graphrag_db::uploaded_source_revision(&source, &content)?;
    let enriched = app.repo.source_entity_enrichment_current(&source).await?;
    let extract_entities = origin["extract_entities"]
        .as_bool()
        .unwrap_or(origin_job.input.extract_entities);
    let applied_options = &origin["processing_options"];
    let extraction_complete = enriched
        || (source.metadata.get("entity_enrichment_v1").is_none()
            && origin_job.input.extract_entities
            && (origin_job.job.status == "completed" || origin_job.phase == "migration_promoted")
            && source.successful_generation == source.generation
            && source.status == graphrag_core::SourceIngestionStatus::Ready
            && origin_job.source_generation == Some(source.successful_generation)
            && origin_job.job.failed_count == 0
            && origin_job.job.completed_count == origin_job.job.total_count);
    let reviewed_enrichment_v1 = app
        .repo
        .reviewed_source_enrichment_proof(&source, &content)
        .await?;
    Ok(UploadedSource {
        id: record_id_to_string(
            source
                .id
                .as_ref()
                .ok_or_else(|| ApplicationError::Internal("Source ID missing".into()))?,
        ),
        uri,
        title: source.title.clone(),
        content,
        content_hash: source.content_hash.clone(),
        generation: source.generation,
        successful_generation: source.successful_generation,
        graph_policy_epoch: source.metadata["graph_policy_revision"]
            .as_u64()
            .unwrap_or(0),
        reviewed_enrichment_v1,
        retired: source.metadata["remote_upload_retired"] == true
            && source.successful_generation == 0
            && source.content_hash.is_none()
            && source.status == graphrag_core::SourceIngestionStatus::Ready,
        status: serde_json::to_value(source.status)
            .map_err(|e| ApplicationError::Internal(e.to_string()))?
            .as_str()
            .unwrap_or_default()
            .into(),
        instance_id: origin["instance_id"].as_str().unwrap_or_default().into(),
        document_key: origin["document_key"].as_str().unwrap_or_default().into(),
        provenance: origin["source"].clone(),
        extract_entities,
        processing_policy_sha256: format!(
            "{:x}",
            Sha256::digest(
                serde_json::to_vec(applied_options)
                    .map_err(|e| ApplicationError::Internal(e.to_string()))?
            )
        ),
        latest_upload_request_id: origin_job.request_id.clone(),
        configured_processing_policy_sha256: graphrag_db::uploaded_processing_policy_sha256(
            &options(app),
        )?,
        processing_policy_current: (!extract_entities || extraction_complete)
            && graphrag_db::uploaded_processing_compatible(
                applied_options,
                &options(app),
                extract_entities,
                false,
            ),
        ingestion_policy_current: graphrag_db::uploaded_processing_compatible(
            &origin_job.input.processing_options,
            &options(app),
            false,
            false,
        ),
        extraction_policy_current: extract_entities.then(|| {
            extraction_complete && applied_options["extraction"] == options(app)["extraction"]
        }),
        revision,
    })
}

pub(crate) async fn get(
    app: &EmbeddedApplication,
    caller: CallerIdentity,
    id: &str,
) -> ApplicationResult<RemoteJobStatus> {
    job_id(id)?;
    view(
        app.repo
            .get_remote_upload_job_status(&caller.instance_id, id)
            .await?
            .ok_or_else(|| {
                ApplicationError::NotFound("This instance has no uploaded job with that ID".into())
            })?,
    )
}
pub(crate) async fn list(
    app: &EmbeddedApplication,
    caller: CallerIdentity,
    limit: usize,
    include_enrichment: bool,
) -> ApplicationResult<RemoteJobList> {
    if !(1..=MAX_REMOTE_JOB_LIST).contains(&limit) {
        return Err(ApplicationError::Validation(
            "Job list limit must be between 1 and 100".into(),
        ));
    }
    Ok(RemoteJobList {
        jobs: app
            .repo
            .list_remote_upload_job_statuses(&caller.instance_id, limit, include_enrichment)
            .await?
            .into_iter()
            .map(view)
            .collect::<ApplicationResult<_>>()?,
    })
}
pub(crate) async fn cancel(
    app: &EmbeddedApplication,
    caller: CallerIdentity,
    id: &str,
) -> ApplicationResult<RemoteJobStatus> {
    job_id(id)?;
    view(
        app.repo
            .cancel_remote_upload_job_status(&caller.instance_id, id)
            .await?,
    )
}
pub(crate) async fn resume(
    app: &EmbeddedApplication,
    caller: CallerIdentity,
    id: &str,
) -> ApplicationResult<RemoteJobStatus> {
    job_id(id)?;
    // Retirement is definitive even when its saved input is damaged or current
    // provider configuration differs. Read the bounded status before decoding
    // execution input or producing compatibility/repair guidance.
    let preflight = app.repo.remote_upload_resume_preflight_guard().await;
    let status = app
        .repo
        .get_remote_upload_job_status(&caller.instance_id, id)
        .await?
        .ok_or_else(|| {
            ApplicationError::NotFound("This instance has no uploaded job with that ID".into())
        })?;
    if status.phase == "retired" {
        return Err(ApplicationError::RevisionConflict(
            "This upload was retired with its source and cannot resume. Use a new upload request only to deliberately recreate the source.".into(),
        ));
    }
    let job = app
        .repo
        .get_remote_upload_job(&caller.instance_id, id)
        .await?
        .ok_or_else(|| {
            ApplicationError::NotFound("This instance has no uploaded job with that ID".into())
        })?;
    compatible(app, &job)?;
    drop(preflight);
    view(
        app.repo
            .resume_remote_upload_job(&caller.instance_id, id)
            .await?
            .into(),
    )
}

pub(crate) async fn claim(
    app: &EmbeddedApplication,
    epoch: &str,
    worker: &str,
) -> ApplicationResult<Option<RemoteJobExecution>> {
    Ok(app
        .repo
        .claim_next_remote_upload(epoch, worker)
        .await?
        .map(|job| RemoteJobExecution {
            job_id: record_id_to_string(job.job.id.as_ref().expect("claimed job has ID")),
            instance_id: job.instance_id,
            service_epoch: epoch.into(),
            worker_token: worker.into(),
        }))
}

pub(crate) async fn execute(
    app: &EmbeddedApplication,
    execution: RemoteJobExecution,
    cancellation: ActionCancellation,
) -> ApplicationResult<()> {
    let lease = lease(&execution)?;
    let job = app.repo.owned_remote_upload_job(&lease).await?;
    let result = if job.input.enrichment.is_some() {
        crate::remote_enrichment::execute(app, &execution, cancellation.clone()).await
    } else {
        run(app, &lease, &cancellation).await
    };
    if let Err(error) = &result {
        // Attempt immediate settlement for direct callers. The server retains
        // this fenced execution on every error and retries recovery until a
        // terminal state or changed owner is observed, including storage faults.
        let _ = recover(app, execution, error.code()).await;
    }
    result
}

async fn run(
    app: &EmbeddedApplication,
    lease: &RemoteJobLease,
    cancellation: &ActionCancellation,
) -> ApplicationResult<()> {
    let librarian = LibrarianAgent::new(
        app.repo.clone(),
        Arc::new(CancellableEmbedder(
            app.embedder.clone(),
            cancellation.clone(),
        )),
        Arc::new(CancellableExtractor(
            app.extractor.clone(),
            cancellation.clone(),
        )),
    )
    .with_runtime_config(app.runtime.clone())
    .with_cancellation_flag(cancellation.flag());
    loop {
        let job = app.repo.owned_remote_upload_job(lease).await?;
        compatible(app, &job)?;
        if job.cancel_requested || cancellation.is_cancelled() {
            return Err(ApplicationError::Cancelled);
        }
        match job.phase.as_str() {
            "queued" | "admitted" | "migration_admitted" => {
                app.repo.begin_remote_upload_generation(lease).await?;
            }
            "preparing" | "migration_preparing" => {
                let source = app.repo.begin_remote_upload_generation(lease).await?;
                let job = app.repo.owned_remote_upload_job(lease).await?;
                if !matches!(job.phase.as_str(), "preparing" | "migration_preparing") {
                    continue;
                }
                app.require_providers(false, cancellation).await?;
                let source_id = source.id.clone().ok_or_else(|| {
                    ApplicationError::Internal("Prepared source ID missing".into())
                })?;
                // A semantic provider/chunk configuration change forces a fresh
                // preparation; exact old chunks must not reuse stale vectors.
                let existing = if source.metadata["remote_upload"]["processing_options"]
                    == job.input.processing_options
                    || ((job.input.policy_migration.is_some()
                        || job.phase == "migration_preparing")
                        && graphrag_db::uploaded_processing_compatible(
                            &source.metadata["remote_upload"]["processing_options"],
                            &job.input.processing_options,
                            false,
                            false,
                        )) {
                    app.repo.get_source_chunks(&source_id).await?
                } else {
                    Vec::new()
                };
                let notes = librarian
                    .prepare_markdown_generation(
                        &job.input.markdown,
                        source_id,
                        Some(source.generation),
                        &existing,
                    )
                    .await?;
                if cancellation.is_cancelled() {
                    return Err(ApplicationError::Cancelled);
                }
                if notes.len() > MAX_REMOTE_UPLOAD_CHUNKS {
                    return Err(ApplicationError::Validation(
                        "Uploaded Markdown exceeds 200 chunks under server settings".into(),
                    ));
                }
                app.repo.stage_remote_upload_notes(lease, notes).await?;
            }
            "staged" | "migration_staged" => {
                let source_id = job
                    .source_id
                    .ok_or_else(|| ApplicationError::Internal("Staged source ID missing".into()))?;
                let old = app.repo.get_source_chunks(&source_id).await?;
                let staged = app.repo.remote_upload_notes(lease).await?;
                let successors = markdown_chunk_successors(&old, &staged);
                if job.phase == "migration_staged" {
                    app.repo
                        .prepare_policy_migration_extraction(lease, &successors)
                        .await?;
                } else {
                    app.repo.reconcile_remote_upload(lease, &successors).await?;
                }
            }
            "migration_extracting" => {
                if !job.input.extract_entities {
                    return Err(ApplicationError::Validation(
                        "Graph rebuild phase requires enabled extraction".into(),
                    ));
                }
                let notes = app.repo.remote_upload_notes(lease).await?;
                if (job.job.completed_count as usize) < notes.len() {
                    let available = tokio::select! {biased; _=cancellation.cancelled()=>return Err(ApplicationError::Cancelled),result=app.extractor.health()=>result.unwrap_or(false)};
                    if !available {
                        return Err(ApplicationError::ProviderUnavailable("Migration extraction provider unavailable; previous searchable generation retained".into()));
                    }
                }
                for (index, note) in notes
                    .iter()
                    .enumerate()
                    .skip(job.job.completed_count as usize)
                {
                    compatible(app, &job)?;
                    if cancellation.is_cancelled() {
                        return Err(ApplicationError::Cancelled);
                    }
                    let entities = librarian.prepare_note_entities(note).await?;
                    compatible(app, &job)?;
                    if cancellation.is_cancelled() {
                        return Err(ApplicationError::Cancelled);
                    }
                    app.repo
                        .checkpoint_policy_migration_entities(lease, index, entities)
                        .await?;
                }
                compatible(app, &job)?;
                let source = job
                    .source_id
                    .as_ref()
                    .ok_or_else(|| ApplicationError::Internal("Migration source missing".into()))?;
                let old = app.repo.get_source_chunks(source).await?;
                app.repo
                    .promote_policy_migration(lease, &markdown_chunk_successors(&old, &notes))
                    .await?;
            }
            "promoted" | "migration_promoted" => {
                if job.job.scope.as_deref() != Some("unchanged") {
                    // Promotion may have succeeded before cleanup/retargeting failed.
                    // Repair that safe boundary before extraction or final success.
                    app.repo.reconcile_remote_upload(lease, &[]).await?;
                }
                if job.input.policy_migration.is_some()
                    || job.phase == "migration_promoted"
                    || job.job.scope.as_deref() == Some("unchanged")
                    || !job.input.extract_entities
                    || job.job.item_ids.is_empty()
                {
                    return complete(app, lease, &job).await;
                }
                app.repo
                    .checkpoint_remote_upload_job(
                        lease,
                        "extracting",
                        ProcessingJobUpdate {
                            completed_count: Some(0),
                            failed_count: Some(0),
                            checkpoint: Some(None),
                            ..Default::default()
                        },
                    )
                    .await?;
            }
            "extracting" => {
                let available = tokio::select! {biased; _=cancellation.cancelled()=>return Err(ApplicationError::Cancelled),result=app.extractor.health()=>result.unwrap_or(false)};
                if !available {
                    return Err(ApplicationError::ProviderUnavailable("Extraction provider unavailable; uploaded generation and checkpoint retained".into()));
                }
                let notes = app.repo.remote_upload_notes(lease).await?;
                let start = job.job.checkpoint.as_ref().map_or(Ok(0), |id| {
                    job.job
                        .item_ids
                        .iter()
                        .position(|item| item == id)
                        .map(|index| index + 1)
                        .ok_or_else(|| {
                            ApplicationError::Internal(
                                "Unknown durable extraction checkpoint".into(),
                            )
                        })
                })?;
                for (index, note) in notes.into_iter().enumerate().skip(start) {
                    if cancellation.is_cancelled() {
                        return Err(ApplicationError::Cancelled);
                    }
                    let entities = librarian.prepare_note_entities(&note).await?;
                    if cancellation.is_cancelled() {
                        return Err(ApplicationError::Cancelled);
                    }
                    app.repo
                        .persist_remote_upload_entities(lease, index, entities)
                        .await?;
                }
                let saved = app.repo.owned_remote_upload_job(lease).await?;
                return complete(app, lease, &saved).await;
            }
            phase => {
                return Err(ApplicationError::Validation(format!(
                    "Unsupported uploaded job phase {phase:?}; no work was started"
                )));
            }
        }
    }
}

async fn complete(
    app: &EmbeddedApplication,
    lease: &RemoteJobLease,
    job: &RemoteUploadJob,
) -> ApplicationResult<()> {
    let result = json!({"source_id":job.source_id.as_ref().map(record_id_to_string),"source_uri":job.source_uri,"generation":job.source_generation,"note_ids":job.job.item_ids,"action":job.job.scope,"extracted":job.input.extract_entities});
    let saved = app
        .repo
        .finish_remote_upload_job(lease, ProcessingJobStatus::Completed, None, Some(result))
        .await?;
    if saved.job.status == "cancelled" {
        return Err(ApplicationError::Cancelled);
    }
    Ok(())
}

pub(crate) async fn interrupt(
    app: &EmbeddedApplication,
    execution: RemoteJobExecution,
) -> ApplicationResult<()> {
    recover(app, execution, "worker_interrupted").await
}

pub(crate) async fn recover(
    app: &EmbeddedApplication,
    execution: RemoteJobExecution,
    error_code: &str,
) -> ApplicationResult<()> {
    let lease = lease(&execution)?;
    let code = match error_code {
        "validation"
        | "not_found"
        | "conflict"
        | "provider_unavailable"
        | "compatibility"
        | "service_unreachable"
        | "cancelled"
        | "internal"
        | "worker_interrupted" => error_code,
        _ => "internal",
    };
    app.repo.recover_remote_upload_job(&lease, code).await?;
    Ok(())
}

impl EmbeddedApplication {
    /// Service-owner development API, intentionally not an MCP tool or background scan.
    /// Requires reviewed exact plan, same authenticated owner and current server policy.
    pub async fn enrich_uploaded_source(
        &self,
        caller: CallerIdentity,
        plan: graphrag_db::SourceEnrichmentPlan,
        cancellation: ActionCancellation,
    ) -> ApplicationResult<graphrag_db::SourceEnrichmentStatus> {
        self.enrich_uploaded_source_leased(caller, plan, cancellation, None)
            .await
    }
    pub async fn enrich_uploaded_source_leased(
        &self,
        caller: CallerIdentity,
        plan: graphrag_db::SourceEnrichmentPlan,
        cancellation: ActionCancellation,
        lease: Option<&RemoteJobLease>,
    ) -> ApplicationResult<graphrag_db::SourceEnrichmentStatus> {
        let check = || -> ApplicationResult<()> {
            if caller.instance_id != plan.instance_id {
                return Err(ApplicationError::Validation(
                    "Enrichment owner does not match authenticated caller".into(),
                ));
            }
            if plan.target_processing_options != options(self) {
                return Err(ApplicationError::Compatibility("Reviewed enrichment policy differs from server policy; re-review before proceeding".into()));
            }
            if cancellation.is_cancelled() {
                return Err(ApplicationError::Cancelled);
            }
            Ok(())
        };
        check()?;
        let _worker = tokio::select! { biased; _=cancellation.cancelled()=>return Err(ApplicationError::Cancelled), guard=self.repo.source_entity_enrichment_worker_guard()=>guard };
        check()?;
        if let Some(owner) = lease {
            self.repo.owned_remote_upload_job(owner).await?;
        }
        let staged = self
            .repo
            .begin_source_entity_enrichment_leased(plan.clone(), lease)
            .await?;
        if staged.status == "promoted" {
            return Ok(staged);
        }
        if staged.status != "staged" {
            return Err(ApplicationError::Validation(
                "Rolled-back enrichment cannot be implicitly restarted".into(),
            ));
        }
        let notes = self.repo.source_entity_enrichment_notes(&plan).await?;
        if staged.completed < notes.len() {
            let available = tokio::select! { biased; _=cancellation.cancelled()=>return Err(ApplicationError::Cancelled), result=self.extractor.health()=>result.unwrap_or(false) };
            if !available {
                return Err(ApplicationError::ProviderUnavailable("Enrichment extraction unavailable; policy remains vector-only and checkpoints retained".into()));
            }
        }
        let librarian = LibrarianAgent::new(
            self.repo.clone(),
            self.embedder.clone(),
            Arc::new(CancellableExtractor(
                self.extractor.clone(),
                cancellation.clone(),
            )),
        )
        .with_runtime_config(self.runtime.clone())
        .with_cancellation_flag(cancellation.flag());
        for (index, note) in notes.iter().enumerate().skip(staged.completed) {
            check()?;
            if let Some(owner) = lease {
                self.repo.owned_remote_upload_job(owner).await?;
            }
            let entities = librarian.prepare_note_entities(note).await?;
            check()?;
            self.repo
                .checkpoint_source_entity_enrichment_leased(&plan, index, entities, lease)
                .await?;
        }
        promote_after_final_preflight(check, || async {
            Ok(self
                .repo
                .promote_source_entity_enrichment_leased(&plan, lease)
                .await?)
        })
        .await
    }
}

#[cfg(test)]
#[path = "enrichment_publication_tests.rs"]
mod enrichment_publication_tests;
