//! Durable, all-or-nothing embedding reindexing.
//!
//! A reindex writes vectors into inactive v009 fields.  The repository swaps
//! those fields and model metadata in one transaction only after every item
//! in the persisted job scope has been successfully embedded.

use crate::{inference::validate_embedding_dim, AgentError, Result, SharedEmbedder};
use graphrag_db::{
    compatibility::EmbeddingIdentity, ProcessingJob, ProcessingJobStatus, ProcessingJobType,
    ProcessingJobUpdate, Repository,
};
use std::collections::BTreeMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

const REINDEX_BATCH_SIZE: usize = 32;

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
}

impl ReindexAgent {
    pub fn new(repo: Repository, embedder: SharedEmbedder) -> Self {
        Self {
            repo,
            embedder,
            cancellation_requested: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn with_cancellation_flag(mut self, requested: Arc<AtomicBool>) -> Self {
        self.cancellation_requested = requested;
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
        self.run_job(job, identity).await
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
        let fingerprints = job.reindex_item_fingerprints.as_ref().ok_or_else(|| {
            AgentError::Processing(
                "reindex job lacks content fingerprints; start a new reindex job instead of resuming unsafe staging".into(),
            )
        })?;
        let id = job
            .id
            .clone()
            .ok_or_else(|| AgentError::Processing("reindex job has no id".into()))?;
        let checkpoint_index = job.checkpoint.as_ref().and_then(|checkpoint| {
            job.item_ids
                .iter()
                .position(|item_id| item_id == checkpoint)
        });
        if let Some(checkpoint_index) = checkpoint_index {
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
                    .update_processing_job(
                        &id,
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
                    .await?;
            }
        }
        let resumed = self.repo.resume_processing_job(&id).await?;
        self.run_job(resumed, identity).await
    }

    async fn run_job(
        &self,
        job: ProcessingJob,
        identity: EmbeddingIdentity,
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
            if self.cancellation_requested.load(Ordering::Acquire) {
                self.repo
                    .update_processing_job(
                        &job_id,
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
                let embeddings = match self.embedder.embed_batch(&texts, false).await {
                    Ok(values) if values.len() == present.len() => values,
                    Ok(values) => {
                        return self
                            .fail(
                                &job_id,
                                completed,
                                format!(
                                    "embedding provider returned {} embeddings for {} inputs",
                                    values.len(),
                                    present.len()
                                ),
                            )
                            .await
                    }
                    Err(error) => return self.fail(&job_id, completed, error.to_string()).await,
                };
                for (item, vector) in present.iter().zip(embeddings) {
                    if let Err(error) = validate_embedding_dim(vector.len()) {
                        return self.fail(&job_id, completed, error.to_string()).await;
                    }
                    if let Err(error) = self
                        .repo
                        .stage_reindex_embedding(&item.id, vector, &item.text)
                        .await
                    {
                        return self.fail(&job_id, completed, error.to_string()).await;
                    }
                    fingerprints.insert(item.id.clone(), reindex_fingerprint(&item.text));
                    completed += 1;
                }
            }
            self.repo
                .update_reindex_item_fingerprints(&job_id, fingerprints.clone())
                .await?;
            self.repo
                .update_processing_job(
                    &job_id,
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
                .update_processing_job(
                    &job_id,
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
        if let Err(error) = self.repo.commit_reindex(&job.item_ids, &identity).await {
            return self.fail(&job_id, completed, error.to_string()).await;
        }
        self.repo
            .update_processing_job(
                &job_id,
                ProcessingJobUpdate {
                    status: Some(ProcessingJobStatus::Completed),
                    completed_count: Some(completed),
                    checkpoint: Some(None),
                    last_error: Some(None),
                    finish: true,
                    ..Default::default()
                },
            )
            .await?;
        Ok(ReindexResult {
            job_id: job_text,
            completed,
            cancelled: false,
        })
    }

    async fn fail<T>(
        &self,
        id: &surrealdb::types::RecordId,
        completed: u64,
        error: String,
    ) -> Result<T> {
        self.repo
            .update_processing_job(
                id,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DeterministicEmbedder, Embedder, InferenceCapabilities};
    use async_trait::async_trait;
    use graphrag_core::Note;
    use graphrag_db::{init_memory, ProcessingJobStatus};

    fn vector(value: f32) -> Vec<f32> {
        vec![value; graphrag_db::schema::EMBEDDING_DIMENSION]
    }

    struct CancellingEmbedder {
        cancelled: Arc<AtomicBool>,
        calls: Arc<std::sync::atomic::AtomicUsize>,
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
