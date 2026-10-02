//! Lightweight host-file metadata for discovery plans; no source content or vectors.
use super::*;

#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct FileSourceSnapshot {
    pub id: RecordId,
    pub uri: Option<String>,
    pub normalized_uri: Option<String>,
    pub content_hash: Option<String>,
    pub generation: u64,
    pub successful_generation: u64,
    pub status: SourceIngestionStatus,
}

impl Repository {
    pub async fn file_source_snapshots(&self) -> Result<Vec<FileSourceSnapshot>> {
        Ok(self.db.query(
            "SELECT id, uri, normalized_uri, content_hash, generation, successful_generation, status \
             FROM source WHERE source_type = 'markdown' ORDER BY normalized_uri, id"
        ).await?.take(0)?)
    }

    /// Recover a folder worker abandoned by a killed process. The persistent
    /// RocksDB handle provides process ownership; other job kinds retain their
    /// existing resume transitions.
    pub async fn resume_folder_sync_job(&self, id: &RecordId) -> Result<ProcessingJob> {
        let job: Option<ProcessingJob> = self.db.query(
            "UPDATE $id SET status = 'running', last_error = NONE, finished_at = NONE, updated_at = time::now() \
             WHERE job_type = 'folder_sync' AND status IN ['running', 'failed', 'cancelled'] RETURN AFTER"
        ).bind(("id", id.clone())).await?.take(0)?;
        job.ok_or_else(|| {
            DbError::NotFound("resumable folder sync job".into(), record_id_to_string(id))
        })
    }
}
