//! Durable, all-or-nothing embedding reindexing.
//!
//! A reindex writes vectors into inactive v009 fields.  The repository swaps
//! those fields and model metadata in one transaction only after every item
//! in the persisted job scope has been successfully embedded.

use crate::{inference::validate_embedding_dim, AgentError, Result, SharedEmbedder};
use chrono::{Duration, Utc};
use graphrag_db::{
    compatibility::EmbeddingIdentity, ProcessingJob, ProcessingJobStatus, ProcessingJobType,
    ProcessingJobUpdate, Repository,
};
use std::collections::BTreeMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration as StdDuration;
use uuid::Uuid;

const REINDEX_BATCH_SIZE: usize = 32;
const REINDEX_LEASE: Duration = Duration::minutes(2);
const REINDEX_HEARTBEAT: StdDuration = StdDuration::from_secs(20);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReindexScope {
    pub notes: bool,
    pub messages: bool,
    pub summaries: bool,
}

impl ReindexScope {
    pub const fn all() -> Self {
        Self {
            notes: true,
            messages: true,
            summaries: true,
        }
    }

    pub const fn is_empty(self) -> bool {
        !self.notes && !self.messages && !self.summaries
    }

    pub const fn is_all(self) -> bool {
        self.notes && self.messages && self.summaries
    }

    pub fn label(self) -> String {
        let mut parts = Vec::new();
        if self.notes {
            parts.push("notes");
        }
        if self.messages {
            parts.push("messages");
        }
        if self.summaries {
            parts.push("summaries");
        }
        parts.join(",")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReindexPreview {
    pub scope: ReindexScope,
    pub item_ids: Vec<String>,
    pub item_fingerprints: BTreeMap<String, String>,
    /// Provider-neutral estimate for operators. Local providers do not expose
    /// a stable currency price, so character volume is the honest preflight
    /// cost unit rather than a fabricated dollar amount.
    pub estimated_input_characters: u64,
}

fn reindex_fingerprint(text: &str) -> String {
    graphrag_core::normalized_content_hash(text)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReindexResult {
    pub job_id: String,
    pub completed: u64,
    pub cancelled: bool,
}

pub struct ReindexAgent {
    repo: Repository,
    embedder: SharedEmbedder,
    cancellation_requested: Arc<AtomicBool>,
    lease_duration: Duration,
    lease_heartbeat: StdDuration,
}

impl ReindexAgent {
    pub fn new(repo: Repository, embedder: SharedEmbedder) -> Self {
        Self {
            repo,
            embedder,
            cancellation_requested: Arc::new(AtomicBool::new(false)),
            lease_duration: REINDEX_LEASE,
            lease_heartbeat: REINDEX_HEARTBEAT,
        }
    }

    pub fn with_cancellation_flag(mut self, requested: Arc<AtomicBool>) -> Self {
        self.cancellation_requested = requested;
        self
    }

    #[cfg(test)]
    fn with_lease_timing_for_test(
        mut self,
        lease_duration: Duration,
        lease_heartbeat: StdDuration,
    ) -> Self {
        self.lease_duration = lease_duration;
        self.lease_heartbeat = lease_heartbeat;
        self
    }

    pub async fn preview(&self, scope: ReindexScope) -> Result<ReindexPreview> {
        if scope.is_empty() {
            return Err(AgentError::Processing(
                "select at least one reindex scope".into(),
            ));
        }
        let item_ids = self
            .repo
            .snapshot_reindex_item_ids(scope.notes, scope.messages, scope.summaries)
            .await?;
        let mut estimated_input_characters = 0_u64;
        let mut item_fingerprints = BTreeMap::new();
        for id in &item_ids {
            if let Some(item) = self.repo.get_reindex_item(id).await? {
                estimated_input_characters =
                    estimated_input_characters.saturating_add(item.text.chars().count() as u64);
                item_fingerprints.insert(item.id, reindex_fingerprint(&item.text));
            }
        }
        Ok(ReindexPreview {
            scope,
            item_ids,
            item_fingerprints,
            estimated_input_characters,
        })
    }

    pub async fn start(
        &self,
        preview: ReindexPreview,
        identity: EmbeddingIdentity,
    ) -> Result<ReindexResult> {
        if let Some(metadata) = self.repo.portable_embedding_metadata().await? {
            if metadata.embedding != identity && !preview.scope.is_all() {
                return Err(AgentError::Processing(
                    "changing the active embedding model requires `reindex --all`; partial scopes would leave a mixed-model corpus".into(),
                ));
            }
        } else if !preview.scope.is_all() && self.repo.vector_bearing_record_count().await? != 0 {
            return Err(AgentError::Processing(
                "legacy vector-bearing data has no embedding metadata; use `reindex --all` before establishing a corpus-wide identity".into(),
            ));
        }
        let job = self
            .repo
            .create_reindex_processing_job(
                preview.item_ids.len() as u64,
                format!("reindex:{}", preview.scope.label()),
                preview.item_ids,
                &identity,
                preview.item_fingerprints,
            )
            .await?;
        let job_id = job
            .id
            .as_ref()
            .ok_or_else(|| AgentError::Processing("reindex job has no id".into()))?;
        let owner = Uuid::new_v4().to_string();
        let claimed = self
            .repo
            .claim_reindex_processing_job(job_id, &owner, Utc::now() + self.lease_duration)
            .await?;
        self.run_job(claimed, identity, owner).await
    }

    pub async fn resume(&self, job_id: &str, identity: EmbeddingIdentity) -> Result<ReindexResult> {
        let job = self
            .repo
            .get_processing_job(job_id)
            .await?
            .ok_or_else(|| AgentError::NotFound(format!("processing job {job_id}")))?;
        if job.job_type_enum() != Some(ProcessingJobType::Reindex) {
            return Err(AgentError::Processing("job is not a reindex job".into()));
        }
        if job.item_ids.is_empty() && job.total_count != 0 {
            return Err(AgentError::Processing(
                "reindex job has no durable item set".into(),
            ));
        }
        if job.target_embedding_provider.as_deref() != Some(identity.provider.as_str())
            || job.target_embedding_model.as_deref() != Some(identity.model.as_str())
            || job.target_embedding_dimension != Some(identity.dimension as i64)
        {
            return Err(AgentError::Processing(
                "reindex job target identity does not match the active provider/model/dimension; start a new reindex job instead of resuming mixed staging".into(),
            ));
        }
        let id = job
            .id
            .clone()
            .ok_or_else(|| AgentError::Processing("reindex job has no id".into()))?;
        let owner = Uuid::new_v4().to_string();
        let job = self
            .repo
            .claim_reindex_processing_job(&id, &owner, Utc::now() + self.lease_duration)
            .await
            .map_err(|_| {
                AgentError::Processing(
                    "reindex job is already owned by a live worker; retry after its lease expires"
                        .into(),
                )
            })?;
        let fingerprints = job.reindex_item_fingerprints.as_ref().ok_or_else(|| {
            AgentError::Processing(
                "reindex job lacks content fingerprints; start a new reindex job instead of resuming unsafe staging".into(),
            )
        })?;
        self.repo
            .adopt_reindex_staging(&job.item_ids, &owner)
            .await?;
        let checkpoint_index = job.checkpoint.as_ref().and_then(|checkpoint| {
            job.item_ids
                .iter()
                .position(|item_id| item_id == checkpoint)
        });
        let job = if let Some(checkpoint_index) = checkpoint_index {
            let mut rewind_to = None;
            for (index, item_id) in job.item_ids[..=checkpoint_index].iter().enumerate() {
                if let Some(item) = self.repo.get_reindex_item(item_id).await? {
                    let persisted = fingerprints.get(item_id);
                    if persisted != Some(&reindex_fingerprint(&item.text)) {
                        rewind_to = Some(index);
                        break;
                    }
                }
            }
            if let Some(rewind_to) = rewind_to {
                self.repo
                    .update_owned_reindex_processing_job(
                        &id,
                        &owner,
                        ProcessingJobUpdate {
                            completed_count: Some(rewind_to as u64),
                            checkpoint: Some(
                                rewind_to
                                    .checked_sub(1)
                                    .and_then(|index| job.item_ids.get(index).cloned()),
                            ),
                            ..Default::default()
                        },
                    )
                    .await?
            } else {
                job
            }
        } else {
            job
        };
        // A batch can stage some rows and then fail before its fingerprint and
        // checkpoint transaction. Those rows are intentionally retryable, so
        // their transient completed count must not survive a resume. The
        // durable checkpoint is the sole progress authority.
        let checkpoint_completed = job
            .checkpoint
            .as_ref()
            .and_then(|checkpoint| {
                job.item_ids
                    .iter()
                    .position(|item_id| item_id == checkpoint)
            })
            .map_or(0, |index| (index + 1) as u64);
        let job = if u64::try_from(job.completed_count.max(0)).unwrap_or(0) != checkpoint_completed
        {
            self.repo
                .update_owned_reindex_processing_job(
                    &id,
                    &owner,
                    ProcessingJobUpdate {
                        completed_count: Some(checkpoint_completed),
                        ..Default::default()
                    },
                )
                .await?
        } else {
            job
        };
        self.run_job(job, identity, owner).await
    }

    async fn run_job(
        &self,
        job: ProcessingJob,
        identity: EmbeddingIdentity,
        owner: String,
    ) -> Result<ReindexResult> {
        let job_id = job
            .id
            .clone()
            .ok_or_else(|| AgentError::Processing("reindex job has no id".into()))?;
        let job_text = graphrag_core::record_id_to_string(&job_id);
        // `checkpoint` is persisted only after the corresponding staged
        // batch commits.  On resume we can therefore skip that durable prefix
        // exactly; a crash during a later batch merely repeats the unclaimed
        // suffix, never re-embeds acknowledged work.
        let start_index = job
            .checkpoint
            .as_ref()
            .and_then(|checkpoint| job.item_ids.iter().position(|id| id == checkpoint))
            .map_or(0, |index| index.saturating_add(1));
        let mut completed = u64::try_from(job.completed_count.max(0)).unwrap_or(0);
        let mut fingerprints = job.reindex_item_fingerprints.clone().ok_or_else(|| {
            AgentError::Processing("reindex job lacks content fingerprints".into())
        })?;

        for ids in job.item_ids[start_index..].chunks(REINDEX_BATCH_SIZE) {
            self.renew_lease(&job_id, &owner).await?;
            if self.cancellation_requested.load(Ordering::Acquire) {
                self.repo
                    .update_owned_reindex_processing_job(
                        &job_id,
                        &owner,
                        ProcessingJobUpdate {
                            status: Some(ProcessingJobStatus::Cancelled),
                            completed_count: Some(completed),
                            finish: true,
                            ..Default::default()
                        },
                    )
                    .await?;
                return Ok(ReindexResult {
                    job_id: job_text,
                    completed,
                    cancelled: true,
                });
            }
            let mut present = Vec::new();
            for id in ids {
                if let Some(item) = self.repo.get_reindex_item(id).await? {
                    present.push(item);
                } else {
                    completed += 1;
                }
            }
            if !present.is_empty() {
                let texts = present
                    .iter()
                    .map(|item| item.text.clone())
                    .collect::<Vec<_>>();
                let embeddings = match self.embed_batch_with_lease(&job_id, &owner, &texts).await {
                    Ok(values) if values.len() == present.len() => values,
                    Ok(values) => {
                        return self
                            .fail(
                                &job_id,
                                &owner,
                                completed,
                                format!(
                                    "embedding provider returned {} embeddings for {} inputs",
                                    values.len(),
                                    present.len()
                                ),
                            )
                            .await
                    }
                    Err(error) => {
                        return self
                            .fail(&job_id, &owner, completed, error.to_string())
                            .await
                    }
                };
                for (item, vector) in present.iter().zip(embeddings) {
                    if let Err(error) = validate_embedding_dim(vector.len()) {
                        return self
                            .fail(&job_id, &owner, completed, error.to_string())
                            .await;
                    }
                    if let Err(error) = self
                        .repo
                        .stage_reindex_embedding(item, vector, &owner)
                        .await
                    {
                        return self
                            .fail(&job_id, &owner, completed, error.to_string())
                            .await;
                    }
                    fingerprints.insert(item.id.clone(), reindex_fingerprint(&item.text));
                    completed += 1;
                }
            }
            self.repo
                .update_reindex_item_fingerprints_owned(&job_id, &owner, fingerprints.clone())
                .await?;
            self.repo
                .update_owned_reindex_processing_job(
                    &job_id,
                    &owner,
                    ProcessingJobUpdate {
                        completed_count: Some(completed),
                        checkpoint: Some(ids.last().cloned()),
                        ..Default::default()
                    },
                )
                .await?;
        }
        if self.cancellation_requested.load(Ordering::Acquire) {
            self.repo
                .update_owned_reindex_processing_job(
                    &job_id,
                    &owner,
                    ProcessingJobUpdate {
                        status: Some(ProcessingJobStatus::Cancelled),
                        completed_count: Some(completed),
                        finish: true,
                        ..Default::default()
                    },
                )
                .await?;
            return Ok(ReindexResult {
                job_id: job_text,
                completed,
                cancelled: true,
            });
        }
        self.renew_lease(&job_id, &owner).await?;
        if let Err(error) = self
            .repo
            .commit_reindex(
                &job_id,
                &owner,
                &job.item_ids,
                &identity,
                job.scope.as_deref() == Some("reindex:notes,messages,summaries"),
                completed,
            )
            .await
        {
            return self
                .fail(&job_id, &owner, completed, error.to_string())
                .await;
        }
        Ok(ReindexResult {
            job_id: job_text,
            completed,
            cancelled: false,
        })
    }

    async fn fail<T>(
        &self,
        id: &surrealdb::types::RecordId,
        owner: &str,
        completed: u64,
        error: String,
    ) -> Result<T> {
        self.repo
            .update_owned_reindex_processing_job(
                id,
                owner,
                ProcessingJobUpdate {
                    status: Some(ProcessingJobStatus::Failed),
                    completed_count: Some(completed),
                    failed_count: Some(1),
                    last_error: Some(Some(error.clone())),
                    finish: true,
                    ..Default::default()
                },
            )
            .await?;
        Err(AgentError::Processing(error))
    }

    async fn renew_lease(&self, id: &surrealdb::types::RecordId, owner: &str) -> Result<()> {
        if self
            .repo
            .renew_reindex_processing_job_lease(id, owner, Utc::now() + self.lease_duration)
            .await?
        {
            Ok(())
        } else {
            Err(AgentError::Processing(
                "reindex job lease was lost to another worker; refusing to publish stale staging"
                    .into(),
            ))
        }
    }

    async fn embed_batch_with_lease(
        &self,
        id: &surrealdb::types::RecordId,
        owner: &str,
        texts: &[String],
    ) -> Result<Vec<Vec<f32>>> {
        let repo = self.repo.clone();
        let id = id.clone();
        let owner = owner.to_string();
        let heartbeat_id = id.clone();
        let heartbeat_owner = owner.clone();
        let lease_duration = self.lease_duration;
        let heartbeat = self.lease_heartbeat;
        let heartbeat_task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(heartbeat);
            interval.tick().await;
            loop {
                interval.tick().await;
                if !repo
                    .renew_reindex_processing_job_lease(
                        &heartbeat_id,
                        &heartbeat_owner,
                        Utc::now() + lease_duration,
                    )
                    .await?
                {
                    return Err(AgentError::Processing(
                        "reindex job lease was lost during embedding; refusing stale batch".into(),
                    ));
                }
            }
            #[allow(unreachable_code)]
            Ok::<(), AgentError>(())
        });
        let result = self.embedder.embed_batch(texts, false).await;
        heartbeat_task.abort();
        let _ = heartbeat_task.await;
        // A final owner-checked renewal turns a failed heartbeat or a delayed
        // provider response into a safe retry rather than stale staging.
        self.renew_lease(&id, &owner).await?;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DeterministicEmbedder, Embedder, InferenceCapabilities};
    use async_trait::async_trait;
    use graphrag_core::{
        ChatConversation, ChatMessage, Entity, EntityType, MessageRole, Note, SourceType,
    };
    use graphrag_db::{init_memory, ProcessingJobStatus};

    fn vector(value: f32) -> Vec<f32> {
        vec![value; graphrag_db::schema::EMBEDDING_DIMENSION]
    }

    struct CancellingEmbedder {
        cancelled: Arc<AtomicBool>,
        calls: Arc<std::sync::atomic::AtomicUsize>,
    }

    struct SlowEmbedder {
        delay: StdDuration,
    }

    #[async_trait]
    impl Embedder for SlowEmbedder {
        async fn embed(&self, _text: &str, _is_query: bool) -> Result<Vec<f32>> {
            tokio::time::sleep(self.delay).await;
            Ok(vector(0.7))
        }

        async fn embed_batch(&self, texts: &[String], _is_query: bool) -> Result<Vec<Vec<f32>>> {
            tokio::time::sleep(self.delay).await;
            Ok(texts.iter().map(|_| vector(0.7)).collect())
        }

        async fn health(&self) -> Result<bool> {
            Ok(true)
        }

        fn capabilities(&self) -> InferenceCapabilities {
            InferenceCapabilities {
                provider: "test".into(),
                model: "slow".into(),
                endpoint: "offline://test".into(),
                known_dimension: Some(1024),
                cache_identity: "slow".into(),
            }
        }
    }

    #[async_trait]
    impl Embedder for CancellingEmbedder {
        async fn embed(&self, _text: &str, _is_query: bool) -> Result<Vec<f32>> {
            Ok(vector(0.4))
        }

        async fn embed_batch(&self, texts: &[String], _is_query: bool) -> Result<Vec<Vec<f32>>> {
            self.calls.fetch_add(1, Ordering::AcqRel);
            self.cancelled.store(true, Ordering::Release);
            Ok(texts.iter().map(|_| vector(0.4)).collect())
        }

        async fn health(&self) -> Result<bool> {
            Ok(true)
        }

        fn capabilities(&self) -> InferenceCapabilities {
            InferenceCapabilities {
                provider: "test".into(),
                model: "cancel-after-batch".into(),
                endpoint: "offline://test".into(),
                known_dimension: Some(1024),
                cache_identity: "test".into(),
            }
        }
    }

    #[tokio::test]
    async fn failed_reindex_keeps_active_vectors_and_metadata() {
        let repo = Repository::new(init_memory().await.unwrap());
        repo.record_embedding_metadata(&EmbeddingIdentity::new("old", "old-model", 1024), None)
            .await
            .unwrap();
        let note = repo
            .create_note(Note::new("durable text").with_embedding(vector(0.1)))
            .await
            .unwrap();
        let failing: SharedEmbedder = Arc::new(
            DeterministicEmbedder::default()
                .with_identity("new", "new-model")
                .fail_next_requests(1, "offline"),
        );
        let agent = ReindexAgent::new(repo.clone(), failing);
        let preview = agent.preview(ReindexScope::all()).await.unwrap();
        let result = agent
            .start(preview, EmbeddingIdentity::new("new", "new-model", 1024))
            .await;
        assert!(result.is_err());
        assert_eq!(
            repo.get_note(&graphrag_core::record_id_to_string(
                note.id.as_ref().unwrap()
            ))
            .await
            .unwrap()
            .unwrap()
            .embedding,
            vector(0.1)
        );
        let metadata = repo.portable_embedding_metadata().await.unwrap().unwrap();
        assert_eq!(metadata.embedding.model, "old-model");
        let job = repo.list_processing_jobs(1).await.unwrap().pop().unwrap();
        assert_eq!(job.status, ProcessingJobStatus::Failed.as_str());
    }

    #[tokio::test]
    async fn heartbeat_keeps_reindex_lease_alive_during_a_slow_embedding_batch() {
        let repo = Repository::new(init_memory().await.unwrap());
        repo.create_note(Note::new("slow batch").with_embedding(vector(0.1)))
            .await
            .unwrap();
        let target = EmbeddingIdentity::new("test", "slow", 1024);
        let result = ReindexAgent::new(
            repo.clone(),
            Arc::new(SlowEmbedder {
                delay: StdDuration::from_millis(180),
            }),
        )
        .with_lease_timing_for_test(Duration::milliseconds(60), StdDuration::from_millis(10))
        .start(
            ReindexAgent::new(repo.clone(), Arc::new(DeterministicEmbedder::default()))
                .preview(ReindexScope::all())
                .await
                .unwrap(),
            target.clone(),
        )
        .await
        .unwrap();
        assert!(!result.cancelled);
        assert_eq!(
            repo.portable_embedding_metadata()
                .await
                .unwrap()
                .unwrap()
                .embedding,
            target
        );
    }

    #[tokio::test]
    async fn partial_reindex_cannot_change_corpus_model_identity() {
        let repo = Repository::new(init_memory().await.unwrap());
        repo.record_embedding_metadata(&EmbeddingIdentity::new("old", "old-model", 1024), None)
            .await
            .unwrap();
        repo.create_note(Note::new("partial").with_embedding(vector(0.1)))
            .await
            .unwrap();
        let embedder: SharedEmbedder =
            Arc::new(DeterministicEmbedder::default().with_identity("new", "new-model"));
        let agent = ReindexAgent::new(repo.clone(), embedder);
        let preview = agent
            .preview(ReindexScope {
                notes: true,
                messages: false,
                summaries: false,
            })
            .await
            .unwrap();
        let error = agent
            .start(preview, EmbeddingIdentity::new("new", "new-model", 1024))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("reindex --all"));
        assert!(repo.list_processing_jobs(10).await.unwrap().is_empty());
        assert_eq!(
            repo.portable_embedding_metadata()
                .await
                .unwrap()
                .unwrap()
                .embedding
                .model,
            "old-model"
        );
    }

    #[tokio::test]
    async fn cancelled_reindex_can_resume_and_cut_over_once_complete() {
        let repo = Repository::new(init_memory().await.unwrap());
        repo.record_embedding_metadata(&EmbeddingIdentity::new("old", "old-model", 1024), None)
            .await
            .unwrap();
        repo.create_note(Note::new("first").with_embedding(vector(0.1)))
            .await
            .unwrap();
        let flag = Arc::new(AtomicBool::new(true));
        let embedder: SharedEmbedder =
            Arc::new(DeterministicEmbedder::default().with_identity("new", "new-model"));
        let cancelled =
            ReindexAgent::new(repo.clone(), embedder.clone()).with_cancellation_flag(flag);
        let preview = cancelled.preview(ReindexScope::all()).await.unwrap();
        let result = cancelled
            .start(preview, EmbeddingIdentity::new("new", "new-model", 1024))
            .await
            .unwrap();
        assert!(result.cancelled);
        let persisted = repo
            .get_processing_job(&result.job_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(persisted.target_embedding_provider.as_deref(), Some("new"));
        assert_eq!(
            persisted.target_embedding_model.as_deref(),
            Some("new-model")
        );
        assert_eq!(persisted.target_embedding_dimension, Some(1024));
        assert_eq!(
            repo.portable_embedding_metadata()
                .await
                .unwrap()
                .unwrap()
                .embedding
                .model,
            "old-model"
        );
        let resumed = ReindexAgent::new(repo.clone(), embedder)
            .resume(
                &result.job_id,
                EmbeddingIdentity::new("new", "new-model", 1024),
            )
            .await
            .unwrap();
        assert!(!resumed.cancelled);
        assert_eq!(
            repo.portable_embedding_metadata()
                .await
                .unwrap()
                .unwrap()
                .embedding
                .model,
            "new-model"
        );
    }

    #[tokio::test]
    async fn resume_rejects_a_different_persisted_embedding_identity() {
        let repo = Repository::new(init_memory().await.unwrap());
        repo.record_embedding_metadata(&EmbeddingIdentity::new("old", "old-model", 1024), None)
            .await
            .unwrap();
        repo.create_note(Note::new("pinned").with_embedding(vector(0.1)))
            .await
            .unwrap();
        let cancelled = Arc::new(AtomicBool::new(true));
        let target: SharedEmbedder =
            Arc::new(DeterministicEmbedder::default().with_identity("target", "target-model"));
        let agent = ReindexAgent::new(repo.clone(), target).with_cancellation_flag(cancelled);
        let job = agent
            .start(
                agent.preview(ReindexScope::all()).await.unwrap(),
                EmbeddingIdentity::new("target", "target-model", 1024),
            )
            .await
            .unwrap();
        let different: SharedEmbedder =
            Arc::new(DeterministicEmbedder::default().with_identity("other", "other-model"));
        let error = ReindexAgent::new(repo.clone(), different)
            .resume(
                &job.job_id,
                EmbeddingIdentity::new("other", "other-model", 1024),
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("target identity"));
        assert_eq!(
            repo.get_processing_job(&job.job_id)
                .await
                .unwrap()
                .unwrap()
                .status,
            ProcessingJobStatus::Cancelled.as_str()
        );
    }

    #[tokio::test]
    async fn resume_resets_partial_batch_progress_to_its_durable_checkpoint() {
        let repo = Repository::new(init_memory().await.unwrap());
        let first = repo
            .create_note(Note::new("first partial stage").with_embedding(vector(0.1)))
            .await
            .unwrap();
        let _second = repo
            .create_note(Note::new("second partial stage").with_embedding(vector(0.1)))
            .await
            .unwrap();
        let target = EmbeddingIdentity::new("target", "target-model", 1024);
        let seed = ReindexAgent::new(
            repo.clone(),
            Arc::new(DeterministicEmbedder::default().with_identity("target", "target-model")),
        );
        let preview = seed.preview(ReindexScope::all()).await.unwrap();
        let job = repo
            .create_reindex_processing_job(
                preview.item_ids.len() as u64,
                format!("reindex:{}", preview.scope.label()),
                preview.item_ids,
                &target,
                preview.item_fingerprints,
            )
            .await
            .unwrap();
        let job_id = job.id.as_ref().unwrap().clone();
        let owner = "failed-partial-batch";
        repo.claim_reindex_processing_job(&job_id, owner, Utc::now() + Duration::minutes(1))
            .await
            .unwrap();
        let first_item = repo
            .get_reindex_item(&graphrag_core::record_id_to_string(
                first.id.as_ref().unwrap(),
            ))
            .await
            .unwrap()
            .unwrap();
        repo.stage_reindex_embedding(&first_item, vector(0.8), owner)
            .await
            .unwrap();
        // Model a failure after the first write in a batch but before the
        // batch fingerprint/checkpoint commit.
        repo.update_owned_reindex_processing_job(
            &job_id,
            owner,
            ProcessingJobUpdate {
                status: Some(ProcessingJobStatus::Failed),
                completed_count: Some(1),
                failed_count: Some(1),
                finish: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();

        let result = seed
            .resume(&graphrag_core::record_id_to_string(&job_id), target)
            .await
            .unwrap();
        assert_eq!(result.completed, 2);
        let persisted = repo
            .get_processing_job(&graphrag_core::record_id_to_string(&job_id))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(persisted.status, ProcessingJobStatus::Completed.as_str());
        assert_eq!(persisted.completed_count, 2);
    }

    #[tokio::test]
    async fn legacy_vectors_without_metadata_gain_identity_only_after_reindex() {
        let repo = Repository::new(init_memory().await.unwrap());
        repo.create_note(Note::new("legacy").with_embedding(vector(0.2)))
            .await
            .unwrap();
        assert!(repo.portable_embedding_metadata().await.unwrap().is_none());
        let embedder: SharedEmbedder = Arc::new(
            DeterministicEmbedder::default().with_identity("replacement", "replacement-model"),
        );
        let agent = ReindexAgent::new(repo.clone(), embedder);
        let preview = agent.preview(ReindexScope::all()).await.unwrap();
        agent
            .start(
                preview,
                EmbeddingIdentity::new("replacement", "replacement-model", 1024),
            )
            .await
            .unwrap();
        assert_eq!(
            repo.portable_embedding_metadata()
                .await
                .unwrap()
                .unwrap()
                .embedding
                .model,
            "replacement-model"
        );
    }

    #[tokio::test]
    async fn legacy_vectors_without_metadata_reject_partial_reindex() {
        let repo = Repository::new(init_memory().await.unwrap());
        repo.create_note(Note::new("legacy").with_embedding(vector(0.2)))
            .await
            .unwrap();
        assert!(repo.portable_embedding_metadata().await.unwrap().is_none());

        let embedder: SharedEmbedder = Arc::new(
            DeterministicEmbedder::default().with_identity("replacement", "replacement-model"),
        );
        let agent = ReindexAgent::new(repo.clone(), embedder);
        let preview = agent
            .preview(ReindexScope {
                notes: true,
                messages: false,
                summaries: false,
            })
            .await
            .unwrap();
        let error = agent
            .start(
                preview,
                EmbeddingIdentity::new("replacement", "replacement-model", 1024),
            )
            .await
            .unwrap_err();

        assert!(error.to_string().contains("reindex --all"));
        assert!(repo.list_processing_jobs(10).await.unwrap().is_empty());
        assert!(repo.portable_embedding_metadata().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn expired_running_reindex_is_recovered_but_live_owner_cannot_be_stolen() {
        let repo = Repository::new(init_memory().await.unwrap());
        repo.create_note(Note::new("orphaned").with_embedding(vector(0.1)))
            .await
            .unwrap();
        let target = EmbeddingIdentity::new("replacement", "replacement-model", 1024);
        let agent = ReindexAgent::new(
            repo.clone(),
            Arc::new(
                DeterministicEmbedder::default().with_identity("replacement", "replacement-model"),
            ),
        );
        let preview = agent.preview(ReindexScope::all()).await.unwrap();
        let job = repo
            .create_reindex_processing_job(
                preview.item_ids.len() as u64,
                format!("reindex:{}", preview.scope.label()),
                preview.item_ids,
                &target,
                preview.item_fingerprints,
            )
            .await
            .unwrap();
        let job_id = job.id.as_ref().unwrap().clone();

        // Simulate a process killed after claiming the job: a lease already
        // in the past is recoverable, and the resumed owner completes it.
        repo.claim_reindex_processing_job(&job_id, "crashed", Utc::now() - Duration::seconds(1))
            .await
            .unwrap();
        let result = agent
            .resume(&graphrag_core::record_id_to_string(&job_id), target.clone())
            .await
            .unwrap();
        assert!(!result.cancelled);

        // A fresh lease is not recoverable by a second process, so a live
        // worker cannot be joined or replaced concurrently.
        let second = repo
            .create_reindex_processing_job(
                0,
                "reindex:all".into(),
                Vec::new(),
                &target,
                BTreeMap::new(),
            )
            .await
            .unwrap();
        let second_id = second.id.as_ref().unwrap().clone();
        repo.claim_reindex_processing_job(&second_id, "live", Utc::now() + Duration::minutes(1))
            .await
            .unwrap();
        let error = agent
            .resume(&graphrag_core::record_id_to_string(&second_id), target)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("live worker"));
    }

    #[tokio::test]
    async fn full_reindex_clears_entity_vectors_before_model_identity_cutover() {
        let repo = Repository::new(init_memory().await.unwrap());
        repo.record_embedding_metadata(&EmbeddingIdentity::new("old", "old-model", 1024), None)
            .await
            .unwrap();
        repo.create_note(Note::new("reindexed").with_embedding(vector(0.1)))
            .await
            .unwrap();
        let mut entity = Entity::new("Old entity", EntityType::Concept).with_embedding(vector(0.2));
        entity.metadata = serde_json::json!({});
        repo.upsert_entity(entity).await.unwrap();
        assert_eq!(repo.vector_bearing_record_count().await.unwrap(), 2);

        let agent = ReindexAgent::new(
            repo.clone(),
            Arc::new(DeterministicEmbedder::default().with_identity("new", "new-model")),
        );
        let target = EmbeddingIdentity::new("new", "new-model", 1024);
        agent
            .start(
                agent.preview(ReindexScope::all()).await.unwrap(),
                target.clone(),
            )
            .await
            .unwrap();

        // The note has a fresh vector and the entity's old-model vector has
        // been cleared, so global metadata never labels it as new-model data.
        assert_eq!(repo.vector_bearing_record_count().await.unwrap(), 1);
        assert_eq!(
            repo.portable_embedding_metadata()
                .await
                .unwrap()
                .unwrap()
                .embedding,
            target
        );
    }

    #[tokio::test]
    async fn reindex_uses_the_same_canonical_inputs_as_ingestion() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut note = Note::new("displayed body").with_embedding(vector(0.1));
        note.search_content = Some("Top > Child\n\ndisplayed body".into());
        note.chunk_heading_path = vec!["Top".into(), "Child".into()];
        let note = repo.create_note(note).await.unwrap();
        let note_item = repo
            .get_reindex_item(&graphrag_core::record_id_to_string(
                note.id.as_ref().unwrap(),
            ))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(note_item.text, "Top > Child\n\ndisplayed body");

        let conversation = ChatConversation {
            uuid: "canonical-inputs".into(),
            name: "Canonical chat".into(),
            summary: "A durable summary".into(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            account: None,
            messages: Vec::new(),
        };
        let conversation_id = repo
            .upsert_conversation(
                &conversation,
                None,
                serde_json::json!({}),
                Some(vector(0.1)),
            )
            .await
            .unwrap();
        let message = ChatMessage {
            uuid: Some("blocks-only".into()),
            role: MessageRole::Assistant,
            content: "   ".into(),
            content_blocks: serde_json::json!([
                {"type": "text", "text": "first block"},
                {"type": "text", "text": "second block"}
            ]),
            created_at: None,
            updated_at: None,
            attachments: Vec::new(),
            files: Vec::new(),
        };
        let message_id = repo
            .upsert_message(
                &conversation_id,
                &conversation.uuid,
                0,
                &message,
                Some(vector(0.1)),
            )
            .await
            .unwrap();
        let message_item = repo
            .get_reindex_item(&graphrag_core::record_id_to_string(&message_id))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(message_item.text, "first block\n\nsecond block");
        let conversation_item = repo
            .get_reindex_item(&graphrag_core::record_id_to_string(&conversation_id))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            conversation_item.text,
            "Canonical chat\n\nA durable summary"
        );

        // Exercise the complete stage/cutover path too: the raw source
        // snapshots must validate all three canonical input forms.
        let target = EmbeddingIdentity::new("canonical", "canonical-model", 1024);
        let agent = ReindexAgent::new(
            repo.clone(),
            Arc::new(
                DeterministicEmbedder::default().with_identity("canonical", "canonical-model"),
            ),
        );
        agent
            .start(
                agent.preview(ReindexScope::all()).await.unwrap(),
                target.clone(),
            )
            .await
            .unwrap();
        assert_eq!(
            repo.portable_embedding_metadata()
                .await
                .unwrap()
                .unwrap()
                .embedding,
            target
        );
    }

    #[tokio::test]
    async fn full_reindex_snapshots_pending_file_generation_notes() {
        let repo = Repository::new(init_memory().await.unwrap());
        repo.record_embedding_metadata(&EmbeddingIdentity::new("old", "old-model", 1024), None)
            .await
            .unwrap();
        let pending = repo
            .begin_file_import(
                SourceType::Markdown,
                "pending.md".into(),
                "file:///pending.md".into(),
                "pending source".into(),
                "sha256:pending".into(),
                false,
            )
            .await
            .unwrap();
        let hidden = repo
            .create_note(
                Note::new("hidden pending generation")
                    .with_embedding(vector(0.1))
                    .with_source(pending.source.id.as_ref().unwrap().clone())
                    .with_source_generation(pending.source.generation),
            )
            .await
            .unwrap();
        let hidden_id = graphrag_core::record_id_to_string(hidden.id.as_ref().unwrap());
        let agent = ReindexAgent::new(
            repo.clone(),
            Arc::new(DeterministicEmbedder::default().with_identity("new", "new-model")),
        );
        let preview = agent.preview(ReindexScope::all()).await.unwrap();
        assert!(preview.item_ids.contains(&hidden_id));
        agent
            .start(preview, EmbeddingIdentity::new("new", "new-model", 1024))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn commit_marks_job_completed_with_its_vector_and_metadata_cutover() {
        let repo = Repository::new(init_memory().await.unwrap());
        let note = repo
            .create_note(Note::new("atomic completion").with_embedding(vector(0.1)))
            .await
            .unwrap();
        let note_id = graphrag_core::record_id_to_string(note.id.as_ref().unwrap());
        let target = EmbeddingIdentity::new("new", "new-model", 1024);
        let mut fingerprints = BTreeMap::new();
        fingerprints.insert(note_id.clone(), reindex_fingerprint("atomic completion"));
        let job = repo
            .create_reindex_processing_job(
                1,
                "reindex:notes".into(),
                vec![note_id.clone()],
                &target,
                fingerprints,
            )
            .await
            .unwrap();
        let job_id = job.id.as_ref().unwrap().clone();
        let owner = "atomic-owner";
        repo.claim_reindex_processing_job(&job_id, owner, Utc::now() + Duration::minutes(1))
            .await
            .unwrap();
        let item = repo.get_reindex_item(&note_id).await.unwrap().unwrap();
        repo.stage_reindex_embedding(&item, vector(0.8), owner)
            .await
            .unwrap();
        repo.commit_reindex(&job_id, owner, &[note_id], &target, false, 1)
            .await
            .unwrap();

        let job = repo
            .get_processing_job(&graphrag_core::record_id_to_string(&job_id))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(job.status, ProcessingJobStatus::Completed.as_str());
        assert_eq!(job.completed_count, 1);
        assert_eq!(
            repo.portable_embedding_metadata()
                .await
                .unwrap()
                .unwrap()
                .embedding,
            target
        );
    }

    #[tokio::test]
    async fn resume_allows_a_deleted_snapshot_item_to_remain_absent_at_cutover() {
        let repo = Repository::new(init_memory().await.unwrap());
        repo.record_embedding_metadata(&EmbeddingIdentity::new("old", "old-model", 1024), None)
            .await
            .unwrap();
        let mut deleted_id = None;
        for number in 0..(REINDEX_BATCH_SIZE + 1) {
            let note = repo
                .create_note(
                    Note::new(format!("deleted snapshot {number}")).with_embedding(vector(0.1)),
                )
                .await
                .unwrap();
            if number == REINDEX_BATCH_SIZE {
                deleted_id = Some(graphrag_core::record_id_to_string(
                    note.id.as_ref().unwrap(),
                ));
            }
        }
        let cancellation = Arc::new(AtomicBool::new(false));
        let cancelled = ReindexAgent::new(
            repo.clone(),
            Arc::new(CancellingEmbedder {
                cancelled: cancellation.clone(),
                calls: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            }),
        )
        .with_cancellation_flag(cancellation);
        let target = EmbeddingIdentity::new("new", "new-model", 1024);
        let job = cancelled
            .start(
                cancelled.preview(ReindexScope::all()).await.unwrap(),
                target.clone(),
            )
            .await
            .unwrap();
        assert!(job.cancelled);
        repo.delete_note(&deleted_id.unwrap()).await.unwrap();

        let result = ReindexAgent::new(
            repo.clone(),
            Arc::new(DeterministicEmbedder::default().with_identity("new", "new-model")),
        )
        .resume(&job.job_id, target.clone())
        .await
        .unwrap();
        assert!(!result.cancelled);
        assert_eq!(result.completed, (REINDEX_BATCH_SIZE + 1) as u64);
        assert_eq!(
            repo.portable_embedding_metadata()
                .await
                .unwrap()
                .unwrap()
                .embedding,
            target
        );
    }

    #[tokio::test]
    async fn commit_revalidates_staged_source_before_publishing_metadata() {
        let repo = Repository::new(init_memory().await.unwrap());
        let note = repo
            .create_note(Note::new("before staging").with_embedding(vector(0.1)))
            .await
            .unwrap();
        let note_id = graphrag_core::record_id_to_string(note.id.as_ref().unwrap());
        let target = EmbeddingIdentity::new("new", "new-model", 1024);
        let mut fingerprints = BTreeMap::new();
        fingerprints.insert(note_id.clone(), reindex_fingerprint("before staging"));
        let job = repo
            .create_reindex_processing_job(
                1,
                "reindex:notes".into(),
                vec![note_id.clone()],
                &target,
                fingerprints,
            )
            .await
            .unwrap();
        let job_id = job.id.as_ref().unwrap().clone();
        let owner = "commit-owner";
        repo.claim_reindex_processing_job(&job_id, owner, Utc::now() + Duration::minutes(1))
            .await
            .unwrap();
        let item = repo.get_reindex_item(&note_id).await.unwrap().unwrap();
        repo.stage_reindex_embedding(&item, vector(0.9), owner)
            .await
            .unwrap();

        let mut edited = repo.get_note(&note_id).await.unwrap().unwrap();
        edited.content = "edited after staging".into();
        repo.update_note(&note_id, edited).await.unwrap();

        assert!(repo
            .commit_reindex(&job_id, owner, &[note_id.clone()], &target, false, 1)
            .await
            .is_err());
        assert_eq!(
            repo.get_note(&note_id).await.unwrap().unwrap().embedding,
            vector(0.1)
        );
        assert!(repo.portable_embedding_metadata().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn incompatible_dimension_fails_before_cutover() {
        let repo = Repository::new(init_memory().await.unwrap());
        repo.record_embedding_metadata(&EmbeddingIdentity::new("old", "old-model", 1024), None)
            .await
            .unwrap();
        repo.create_note(Note::new("dimension").with_embedding(vector(0.3)))
            .await
            .unwrap();
        let embedder: SharedEmbedder = Arc::new(
            DeterministicEmbedder::default()
                .with_identity("small", "small-model")
                .with_default_embedding(vec![0.0; 8]),
        );
        let agent = ReindexAgent::new(repo.clone(), embedder);
        let preview = agent.preview(ReindexScope::all()).await.unwrap();
        assert!(agent
            .start(preview, EmbeddingIdentity::new("small", "small-model", 8))
            .await
            .is_err());
        assert_eq!(
            repo.portable_embedding_metadata()
                .await
                .unwrap()
                .unwrap()
                .embedding
                .model,
            "old-model"
        );
    }

    #[tokio::test]
    async fn resume_skips_checkpointed_batches_instead_of_reembedding_them() {
        let repo = Repository::new(init_memory().await.unwrap());
        repo.record_embedding_metadata(&EmbeddingIdentity::new("old", "old-model", 1024), None)
            .await
            .unwrap();
        for number in 0..(REINDEX_BATCH_SIZE + 1) {
            repo.create_note(Note::new(format!("resume {number}")).with_embedding(vector(0.1)))
                .await
                .unwrap();
        }
        let cancelled = Arc::new(AtomicBool::new(false));
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let stopping: SharedEmbedder = Arc::new(CancellingEmbedder {
            cancelled: cancelled.clone(),
            calls: calls.clone(),
        });
        let agent = ReindexAgent::new(repo.clone(), stopping).with_cancellation_flag(cancelled);
        let preview = agent.preview(ReindexScope::all()).await.unwrap();
        let cancelled_run = agent
            .start(preview, EmbeddingIdentity::new("new", "new", 1024))
            .await
            .unwrap();
        assert!(cancelled_run.cancelled);
        assert_eq!(calls.load(Ordering::Acquire), 1);

        let resumed_embedder: SharedEmbedder =
            Arc::new(DeterministicEmbedder::default().with_identity("new", "new"));
        let resumed = ReindexAgent::new(repo, resumed_embedder)
            .resume(
                &cancelled_run.job_id,
                EmbeddingIdentity::new("new", "new", 1024),
            )
            .await
            .unwrap();
        assert_eq!(resumed.completed, (REINDEX_BATCH_SIZE + 1) as u64);
    }

    #[tokio::test]
    async fn resume_reembeds_a_checkpointed_note_edited_after_staging() {
        let repo = Repository::new(init_memory().await.unwrap());
        repo.record_embedding_metadata(&EmbeddingIdentity::new("old", "old-model", 1024), None)
            .await
            .unwrap();
        let mut first_id = None;
        for number in 0..(REINDEX_BATCH_SIZE + 1) {
            let note = repo
                .create_note(Note::new(format!("snapshot {number}")).with_embedding(vector(0.1)))
                .await
                .unwrap();
            if number == 0 {
                first_id = Some(graphrag_core::record_id_to_string(
                    note.id.as_ref().unwrap(),
                ));
            }
        }
        let first_id = first_id.unwrap();
        let cancelled = Arc::new(AtomicBool::new(false));
        let stopping: SharedEmbedder = Arc::new(CancellingEmbedder {
            cancelled: cancelled.clone(),
            calls: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        });
        let agent = ReindexAgent::new(repo.clone(), stopping).with_cancellation_flag(cancelled);
        let job = agent
            .start(
                agent.preview(ReindexScope::all()).await.unwrap(),
                EmbeddingIdentity::new("new", "new-model", 1024),
            )
            .await
            .unwrap();
        assert!(job.cancelled);
        let mut edited = repo.get_note(&first_id).await.unwrap().unwrap();
        edited.content = "edited after checkpoint".into();
        repo.update_note(&first_id, edited).await.unwrap();

        let fixture = DeterministicEmbedder::default().with_identity("new", "new-model");
        let expected = fixture
            .embed("edited after checkpoint", false)
            .await
            .unwrap();
        let resumed = ReindexAgent::new(repo.clone(), Arc::new(fixture))
            .resume(
                &job.job_id,
                EmbeddingIdentity::new("new", "new-model", 1024),
            )
            .await
            .unwrap();
        assert!(!resumed.cancelled);
        assert_eq!(
            repo.get_note(&first_id).await.unwrap().unwrap().embedding,
            expected
        );
    }
}
