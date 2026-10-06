//! Remote reads and durable capture reuse the embedded agents and repository.
use crate::inference::{CancellableEmbedder, CancellableExtractor};
use crate::*;
use async_trait::async_trait;
use graphrag_agents::{GraphMode, LibrarianAgent, SearchHitType, SearchScope};
use graphrag_db::RemoteCaptureInput;
use sha2::{Digest, Sha256};
use std::sync::Arc;

/// Stable payload identity excludes transport request ID and generated state.
pub fn remote_capture_fingerprint(request: &RemoteCaptureRequest) -> ApplicationResult<String> {
    let payload = serde_json::to_vec(&(
        REMOTE_CAPTURE_PAYLOAD_VERSION,
        &request.content,
        &request.title,
        &request.tags,
        &request.provenance,
    ))
    .map_err(|error| ApplicationError::Internal(error.to_string()))?;
    Ok(format!("{:x}", Sha256::digest(payload)))
}

fn validate_text(name: &str, value: &str, maximum: usize, required: bool) -> ApplicationResult<()> {
    if value.chars().count() > maximum || (required && value.trim().is_empty()) {
        return Err(ApplicationError::Validation(format!(
            "{name} must {}contain at most {maximum} characters",
            if required { "be nonempty and " } else { "" }
        )));
    }
    Ok(())
}

pub(crate) fn validate_since_days(days: Option<u32>) -> ApplicationResult<()> {
    if days.is_some_and(|days| days > MAX_REMOTE_SINCE_DAYS) {
        return Err(ApplicationError::Validation(format!(
            "since_days must be at most {MAX_REMOTE_SINCE_DAYS}"
        )));
    }
    Ok(())
}

fn credential_key(key: &str) -> bool {
    let normalized: String = key
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .map(|character| character.to_ascii_lowercase())
        .collect();
    matches!(normalized.as_str(), "key" | "auth" | "sig")
        || [
            "token",
            "secret",
            "password",
            "credential",
            "privatekey",
            "accesskey",
            "apikey",
            "authorization",
            "bearer",
            "signature",
            "session",
            "cookie",
            "jwt",
        ]
        .iter()
        .any(|marker| normalized.contains(marker))
}

fn validate_source_uri(uri: &str) -> ApplicationResult<()> {
    validate_text("source URI", uri, 2048, true)?;
    let uri = url::Url::parse(uri).map_err(|_| {
        ApplicationError::Validation(
            "source URI must be an absolute URI without credentials".into(),
        )
    })?;
    let fragment_has_credentials = uri.fragment().is_some_and(|fragment| {
        let query = fragment
            .split_once('?')
            .map_or(fragment, |(_, query)| query);
        url::form_urlencoded::parse(query.as_bytes()).any(|(key, _)| credential_key(&key))
    });
    if !uri.username().is_empty()
        || uri.password().is_some()
        || uri.query_pairs().any(|(key, _)| credential_key(&key))
        || fragment_has_credentials
    {
        return Err(ApplicationError::Validation(
            "source URI cannot contain user information or credential query/fragment fields".into(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_capture(
    caller: &CallerIdentity,
    request: &RemoteCaptureRequest,
) -> ApplicationResult<()> {
    for (name, value) in [
        ("instance ID", caller.instance_id.as_str()),
        ("request ID", request.request_id.as_str()),
    ] {
        validate_text(name, value, 128, true)?;
        if value.len() > 256 || value.trim() != value || value.chars().any(char::is_control) {
            return Err(ApplicationError::Validation(format!("{name} cannot exceed 256 UTF-8 bytes or contain control characters or surrounding whitespace")));
        }
    }
    if request.content.trim().is_empty() || request.content.len() > MAX_REMOTE_CAPTURE_BYTES {
        return Err(ApplicationError::Validation(
            "capture content must be nonempty and at most 65536 UTF-8 bytes".into(),
        ));
    }
    if let Some(title) = &request.title {
        validate_text("title", title, 512, false)?;
    }
    if request.tags.len() > 32 {
        return Err(ApplicationError::Validation(
            "capture supports at most 32 tags".into(),
        ));
    }
    for tag in &request.tags {
        validate_text("tag", tag, 64, true)?;
    }
    if let Some(provenance) = &request.provenance {
        if let Some(uri) = &provenance.uri {
            validate_source_uri(uri)?;
        }
        if let Some(label) = &provenance.label {
            validate_text("source label", label, 512, false)?;
        }
        if provenance.metadata.len() > 32 {
            return Err(ApplicationError::Validation(
                "source provenance supports at most 32 metadata entries".into(),
            ));
        }
        for (key, value) in &provenance.metadata {
            validate_text("source metadata key", key, 64, true)?;
            validate_text("source metadata value", value, 1024, false)?;
        }
    }
    Ok(())
}

fn capture_response(
    request_id: String,
    result: serde_json::Value,
    replayed: bool,
) -> ApplicationResult<RemoteCaptureResponse> {
    let record = serde_json::from_value(result).map_err(|error| {
        ApplicationError::Internal(format!(
            "Durable capture receipt has an unsupported result shape: {error}"
        ))
    })?;
    Ok(RemoteCaptureResponse {
        request_id,
        replayed,
        record,
    })
}

#[async_trait]
impl RemoteApplicationOperations for EmbeddedApplication {
    async fn service_readiness(&self) -> ApplicationResult<ApplicationReadiness> {
        self.readonly_readiness().await
    }
    async fn note_snapshot(&self, reference: RecordRef) -> ApplicationResult<RemoteNoteSnapshot> {
        self.remote_note_snapshot_impl(reference).await
    }
    async fn edit_remote(
        &self,
        caller: CallerIdentity,
        request: RemoteEditRequest,
        cancellation: ActionCancellation,
    ) -> ApplicationResult<RemoteMutationResponse> {
        self.remote_edit_impl(caller, request, cancellation).await
    }
    async fn delete_remote(
        &self,
        caller: CallerIdentity,
        request: RemoteDeleteRequest,
    ) -> ApplicationResult<RemoteMutationResponse> {
        self.remote_delete_impl(caller, request).await
    }
    async fn decide_remote(
        &self,
        caller: CallerIdentity,
        request: RemoteDecisionRequest,
    ) -> ApplicationResult<RemoteMutationResponse> {
        self.remote_decision_impl(caller, request).await
    }

    async fn build_context(
        &self,
        request: BuildContextRequest,
        cancellation: ActionCancellation,
    ) -> ApplicationResult<ContextResponse> {
        validate_text("query", &request.query, MAX_REMOTE_QUERY_CHARS, true)?;
        validate_since_days(request.since_days)?;
        if let Some(uri) = &request.source_uri {
            validate_text("source URI", uri, 2048, false)?;
        }
        if let Some(entity) = &request.entity_filter {
            validate_text("entity filter", entity, 512, false)?;
        }
        let mut options = self.augment_options.clone();
        options.max_chunks = request.max_chunks.unwrap_or(options.max_chunks);
        options.max_total_tokens = request.max_total_tokens.unwrap_or(options.max_total_tokens);
        options.max_chunk_tokens = request.max_chunk_tokens.unwrap_or(options.max_chunk_tokens);
        if options.max_chunks > MAX_RESULT_LIMIT
            || options.max_total_tokens > MAX_CONTEXT_TOKENS
            || options.max_chunk_tokens > MAX_CONTEXT_CHUNK_TOKENS
        {
            return Err(ApplicationError::Validation("context bounds are at most 200 chunks, 32768 total tokens, and 8192 tokens per chunk".into()));
        }
        if cancellation.is_cancelled() {
            return Err(ApplicationError::Cancelled);
        }
        let zero_budget = options.max_chunks == 0
            || options.max_total_tokens == 0
            || options.max_chunk_tokens == 0;
        if !zero_budget {
            self.require_providers(false, &cancellation).await?;
        }
        let scope = match request.scope {
            Scope::Notes => SearchScope::Notes,
            Scope::Messages => SearchScope::Messages,
            Scope::All => SearchScope::All,
        };
        let graph = match request.graph {
            GraphPolicy::Off => GraphMode::Off,
            GraphPolicy::Auto => GraphMode::Auto,
            GraphPolicy::On => GraphMode::On,
        };
        let context = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return Err(ApplicationError::Cancelled),
            result = self.search.build_augmented_context_with_graph(&request.query, scope,
                request.since_days, request.source_uri, request.entity_filter, options, graph) => result?,
        };
        let rendered_context = context.render_prompt_block();
        let diagnostics = serde_json::to_value(context.diagnostics)
            .map_err(|error| ApplicationError::Internal(error.to_string()))?;
        let mut chunks = Vec::with_capacity(context.chunks.len());
        for chunk in context.chunks {
            chunks.push(ContextChunk {
                citation: chunk.citation,
                id: chunk.id,
                hit_type: match chunk.hit_type {
                    SearchHitType::Note => "note",
                    SearchHitType::Message => "message",
                    SearchHitType::ConversationSummary => "conversation_summary",
                }
                .into(),
                title: chunk.title,
                snippet: chunk.snippet,
                created_at: chunk.created_at.map(|value| value.to_rfc3339()),
                source_uri: chunk.source_uri,
                score: chunk.score,
                conversation_uuid: chunk.conversation_uuid,
                message_index: chunk.message_index,
                role: chunk.role,
                rendered_tokens: chunk.rendered_tokens,
                approx_tokens: chunk.approx_tokens,
                truncated: chunk.truncated,
                selected_span_start: chunk.selected_span_start,
                selected_span_end: chunk.selected_span_end,
                graph: chunk
                    .graph
                    .map(serde_json::to_value)
                    .transpose()
                    .map_err(|error| ApplicationError::Internal(error.to_string()))?,
            });
        }
        Ok(ContextResponse {
            query: context.query,
            scope: request.scope,
            chunks,
            total_tokens: context.total_tokens,
            diagnostics,
            rendered_context,
        })
    }

    async fn capture_remote(
        &self,
        caller: CallerIdentity,
        request: RemoteCaptureRequest,
        cancellation: ActionCancellation,
    ) -> ApplicationResult<RemoteCaptureResponse> {
        validate_capture(&caller, &request)?;
        let fingerprint = remote_capture_fingerprint(&request)?;
        // Reconnect/retry succeeds even if inference has since become unavailable.
        if let Some(receipt) = self
            .repo
            .find_remote_capture_receipt(&caller.instance_id, &request.request_id, &fingerprint)
            .await?
        {
            return capture_response(request.request_id, receipt.result, true);
        }
        self.require_providers(!self.runtime.skip_entity_extraction, &cancellation)
            .await?;
        let librarian = LibrarianAgent::new(
            self.repo.clone(),
            Arc::new(CancellableEmbedder(
                self.embedder.clone(),
                cancellation.clone(),
            )),
            Arc::new(CancellableExtractor(
                self.extractor.clone(),
                cancellation.clone(),
            )),
        )
        .with_runtime_config(self.runtime.clone())
        .with_cancellation_flag(cancellation.flag());
        let (note, entities) = librarian
            .prepare_manual_capture(
                request.content.clone(),
                request.title.clone(),
                request.tags.clone(),
            )
            .await?;
        if cancellation.is_cancelled() {
            return Err(ApplicationError::Cancelled);
        }
        let source_provenance = serde_json::to_value(&request.provenance)
            .map_err(|error| ApplicationError::Internal(error.to_string()))?;
        let input = RemoteCaptureInput {
            authenticated_instance_id: caller.instance_id,
            request_id: request.request_id.clone(),
            payload_fingerprint: fingerprint,
            content: note.content,
            title: note.title,
            tags: note.tags,
            source_provenance,
        };
        // Never cancellation-select this atomic write. Its original response
        // snapshot and note/entities/source commit together in the repository.
        let receipt = self
            .repo
            .capture_remote_note(input, note.embedding, entities)
            .await?;
        capture_response(request.request_id, receipt.result, receipt.replayed)
    }
    async fn upload_source(
        &self,
        caller: CallerIdentity,
        request: UploadSourceRequest,
    ) -> ApplicationResult<UploadAdmission> {
        crate::remote_jobs::upload(self, caller, request).await
    }
    async fn get_uploaded_source(&self, id: &str) -> ApplicationResult<UploadedSource> {
        crate::remote_jobs::source(self, id).await
    }
    async fn get_remote_job(
        &self,
        caller: CallerIdentity,
        id: &str,
    ) -> ApplicationResult<RemoteJobStatus> {
        crate::remote_jobs::get(self, caller, id).await
    }
    async fn list_remote_jobs(
        &self,
        caller: CallerIdentity,
        limit: usize,
    ) -> ApplicationResult<RemoteJobList> {
        crate::remote_jobs::list(self, caller, limit).await
    }
    async fn cancel_remote_job(
        &self,
        caller: CallerIdentity,
        id: &str,
    ) -> ApplicationResult<RemoteJobStatus> {
        crate::remote_jobs::cancel(self, caller, id).await
    }
    async fn resume_remote_job(
        &self,
        caller: CallerIdentity,
        id: &str,
    ) -> ApplicationResult<RemoteJobStatus> {
        crate::remote_jobs::resume(self, caller, id).await
    }
    async fn reconcile_remote_jobs(&self, epoch: &str) -> ApplicationResult<()> {
        self.repo
            .reconcile_interrupted_remote_uploads(epoch)
            .await?;
        Ok(())
    }
    async fn claim_remote_job(
        &self,
        epoch: &str,
        worker: &str,
    ) -> ApplicationResult<Option<RemoteJobExecution>> {
        crate::remote_jobs::claim(self, epoch, worker).await
    }
    async fn remote_job_cancel_requested(
        &self,
        execution: &RemoteJobExecution,
    ) -> ApplicationResult<bool> {
        let job = self
            .repo
            .owned_remote_upload_job(&crate::remote_jobs::lease(execution)?)
            .await?;
        Ok(job.cancel_requested)
    }
    async fn execute_remote_job(
        &self,
        execution: RemoteJobExecution,
        cancel: ActionCancellation,
    ) -> ApplicationResult<()> {
        crate::remote_jobs::execute(self, execution, cancel).await
    }
    async fn interrupt_remote_job(&self, execution: RemoteJobExecution) -> ApplicationResult<()> {
        crate::remote_jobs::interrupt(self, execution).await
    }
    async fn recover_remote_job(
        &self,
        execution: RemoteJobExecution,
        error_code: String,
    ) -> ApplicationResult<()> {
        crate::remote_jobs::recover(self, execution, &error_code).await
    }
}
