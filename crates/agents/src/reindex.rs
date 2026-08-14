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
        Ok(ReindexPreview {
            scope,
            item_ids: self
                .repo
                .snapshot_reindex_item_ids(scope.notes, scope.messages, scope.summaries)
                .await?,
        })
    }

    pub async fn start(
        &self,
        preview: ReindexPreview,
        identity: EmbeddingIdentity,
    ) -> Result<ReindexResult> {
        let job = self
            .repo
            .create_processing_job_with_scope(
                ProcessingJobType::Reindex,
                None,
                preview.item_ids.len() as u64,
                Some(format!("reindex:{}", preview.scope.label())),
                preview.item_ids,
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
        let id = job
            .id
            .clone()
            .ok_or_else(|| AgentError::Processing("reindex job has no id".into()))?;
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
        let mut completed = 0_u64;

        for ids in job.item_ids.chunks(REINDEX_BATCH_SIZE) {
            if self.cancellation_requested.load(Ordering::Acquire) {
                self.repo
                    .update_processing_job(
                        &job_id,
                        ProcessingJobUpdate {
                            status: Some(ProcessingJobStatus::Cancelled),
                            completed_count: Some(completed),
                            checkpoint: Some(None),
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
                    if let Err(error) = self.repo.stage_reindex_embedding(&item.id, vector).await {
                        return self.fail(&job_id, completed, error.to_string()).await;
                    }
                    completed += 1;
                }
            }
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
    use crate::DeterministicEmbedder;
    use graphrag_core::Note;
    use graphrag_db::{init_memory, ProcessingJobStatus};

    fn vector(value: f32) -> Vec<f32> {
        vec![value; graphrag_db::schema::EMBEDDING_DIMENSION]
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
        let preview = agent
            .preview(ReindexScope {
                notes: true,
                messages: false,
                summaries: false,
            })
            .await
            .unwrap();
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
        let preview = cancelled
            .preview(ReindexScope {
                notes: true,
                messages: false,
                summaries: false,
            })
            .await
            .unwrap();
        let result = cancelled
            .start(preview, EmbeddingIdentity::new("new", "new-model", 1024))
            .await
            .unwrap();
        assert!(result.cancelled);
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
        let preview = agent
            .preview(ReindexScope {
                notes: true,
                messages: false,
                summaries: false,
            })
            .await
            .unwrap();
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
        let preview = agent
            .preview(ReindexScope {
                notes: true,
                messages: false,
                summaries: false,
            })
            .await
            .unwrap();
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
}
