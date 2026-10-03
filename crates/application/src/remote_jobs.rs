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
use std::sync::Arc;

/// Frozen upload journal identity; unrelated local API changes never invalidate retries.
const REMOTE_UPLOAD_PAYLOAD_VERSION: u32 = 1;

fn job_id(id: &str) -> ApplicationResult<()> {
    if id.len() > 512 || id.chars().any(char::is_control) || !id.starts_with("processing_job:") {
        return Err(ApplicationError::Validation(
            "Use the canonical processing_job:ID returned by upload_source".into(),
        ));
    }
    parse_record_id(id, Some("processing_job"))?;
    Ok(())
}

fn options(app: &EmbeddedApplication) -> Value {
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
    if job.input.processing_options != options(app) {
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
        job_type: job.job.job_type,
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
        result: job.result,
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
    let fingerprint = format!(
        "{:x}",
        Sha256::digest(
            serde_json::to_vec(&(
                REMOTE_UPLOAD_PAYLOAD_VERSION,
                "uploaded_markdown",
                &request.document_key,
                &request.content,
                &request.title,
                &request.provenance,
                request.extract_entities
            ))
            .map_err(|e| ApplicationError::Internal(e.to_string()))?
        )
    );
    let admission = if let Some(admission) = app
        .repo
        .find_remote_upload_admission(&caller.instance_id, &request.request_id, &fingerprint)
        .await?
    {
        admission
    } else {
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
                processing_options: options(app),
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
    let revision = format!(
        "{:x}",
        Sha256::digest(
            serde_json::to_vec(&(
                id,
                &source.title,
                &source.content,
                &content,
                &source.content_hash,
                source.generation,
                source.successful_generation,
                &source.status,
                origin
            ))
            .map_err(|e| ApplicationError::Internal(e.to_string()))?
        )
    );
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
        status: serde_json::to_value(source.status)
            .map_err(|e| ApplicationError::Internal(e.to_string()))?
            .as_str()
            .unwrap_or_default()
            .into(),
        instance_id: origin["instance_id"].as_str().unwrap_or_default().into(),
        document_key: origin["document_key"].as_str().unwrap_or_default().into(),
        provenance: origin["source"].clone(),
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
) -> ApplicationResult<RemoteJobList> {
    if !(1..=MAX_REMOTE_JOB_LIST).contains(&limit) {
        return Err(ApplicationError::Validation(
            "Job list limit must be between 1 and 100".into(),
        ));
    }
    Ok(RemoteJobList {
        jobs: app
            .repo
            .list_remote_upload_job_statuses(&caller.instance_id, limit)
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
    let job = app
        .repo
        .get_remote_upload_job(&caller.instance_id, id)
        .await?
        .ok_or_else(|| {
            ApplicationError::NotFound("This instance has no uploaded job with that ID".into())
        })?;
    compatible(app, &job)?;
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
    let result = run(app, &lease, &cancellation).await;
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
            "queued" | "admitted" => {
                app.repo.begin_remote_upload_generation(lease).await?;
            }
            "preparing" => {
                let source = app.repo.begin_remote_upload_generation(lease).await?;
                let job = app.repo.owned_remote_upload_job(lease).await?;
                if job.phase != "preparing" {
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
                {
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
            "staged" => {
                let source_id = job
                    .source_id
                    .ok_or_else(|| ApplicationError::Internal("Staged source ID missing".into()))?;
                let old = app.repo.get_source_chunks(&source_id).await?;
                let staged = app.repo.remote_upload_notes(lease).await?;
                app.repo
                    .reconcile_remote_upload(lease, &markdown_chunk_successors(&old, &staged))
                    .await?;
            }
            "promoted" => {
                if job.job.scope.as_deref() != Some("unchanged") {
                    // Promotion may have succeeded before cleanup/retargeting failed.
                    // Repair that safe boundary before extraction or final success.
                    app.repo.reconcile_remote_upload(lease, &[]).await?;
                }
                if job.job.scope.as_deref() == Some("unchanged")
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
                )))
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
