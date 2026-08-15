//! Durable processing-job and inference-cache ownership.
//!
//! Cache writes rely on the schema's unique semantic key. They remain a
//! single UPSERT so concurrent misses converge without changing the value.

use super::*;

/// The coarse operation of a durable local processing job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessingJobType {
    Embedding,
    EntityExtraction,
    Reindex,
}

impl ProcessingJobType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::EntityExtraction => "entity_extraction",
            Self::Reindex => "reindex",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "embedding" => Some(Self::Embedding),
            "entity_extraction" => Some(Self::EntityExtraction),
            "reindex" => Some(Self::Reindex),
            _ => None,
        }
    }
}

/// State transitions are intentionally small: workers own `running`, while
/// command handlers can request cancellation between atomic item mutations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessingJobStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl ProcessingJobStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Persisted checkpoint and aggregate counts for a local processing run.
#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct ProcessingJob {
    pub id: Option<RecordId>,
    pub job_type: String,
    pub source_generation: Option<String>,
    pub scope: Option<String>,
    /// Reindex jobs pin this identity so a later resume cannot apply a
    /// different provider/model's staged vectors to the same generation.
    #[serde(default)]
    pub target_embedding_provider: Option<String>,
    #[serde(default)]
    pub target_embedding_model: Option<String>,
    #[serde(default)]
    pub target_embedding_dimension: Option<i64>,
    /// Snapshot text fingerprints for reindex items. They make a checkpoint
    /// safe to resume even when an item was edited after its vector staged.
    #[serde(default)]
    pub reindex_item_fingerprints: Option<std::collections::BTreeMap<String, String>>,
    /// Renewable owner lease for a reindex worker. An expired owner may be
    /// replaced after a crash, but a live owner cannot be resumed concurrently.
    #[serde(default)]
    pub reindex_lease_owner: Option<String>,
    #[serde(default)]
    pub reindex_lease_expires_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub item_ids: Vec<String>,
    pub status: String,
    pub total_count: i64,
    pub completed_count: i64,
    pub failed_count: i64,
    pub checkpoint: Option<String>,
    pub last_error: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
}

impl ProcessingJob {
    pub fn job_type_enum(&self) -> Option<ProcessingJobType> {
        ProcessingJobType::parse(&self.job_type)
    }
}

/// Partial update with explicit nested options for nullable fields: `None`
/// leaves a field unchanged; `Some(None)` clears it.
#[derive(Debug, Clone, Default)]
pub struct ProcessingJobUpdate {
    pub status: Option<ProcessingJobStatus>,
    pub completed_count: Option<u64>,
    pub failed_count: Option<u64>,
    pub checkpoint: Option<Option<String>>,
    pub last_error: Option<Option<String>>,
    pub finish: bool,
}

/// An audited cache entry. `cache_key` is computed by the processing layer;
/// remaining fields make collision/debug inspection deterministic.
#[derive(Debug, Clone)]
pub struct InferenceCacheEntry {
    pub cache_key: String,
    pub operation: String,
    pub provider: String,
    pub model: String,
    pub version: String,
    pub input_hash: String,
    pub value: serde_json::Value,
}

/// One durable reindex item. Its id determines whether `text` comes from a
/// note, chat message, or conversation summary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReindexItem {
    pub id: String,
    /// Exact canonical input submitted to the embedding provider.
    pub text: String,
    /// Source fields from which `text` was derived. Commit validates this
    /// snapshot transactionally before making the staged vector visible.
    pub source_snapshot: serde_json::Value,
}

fn canonical_note_reindex_text(
    content: &str,
    search_content: Option<&str>,
    chunk_heading_path: &[String],
) -> String {
    search_content.map_or_else(
        || {
            let headings = chunk_heading_path
                .iter()
                .filter(|heading| !heading.is_empty())
                .cloned()
                .collect::<Vec<_>>()
                .join(" > ");
            if headings.is_empty() {
                content.to_string()
            } else {
                format!("{headings}\n\n{content}")
            }
        },
        str::to_string,
    )
}

fn canonical_message_reindex_text(content: &str, content_blocks: &serde_json::Value) -> String {
    if !content.trim().is_empty() {
        return content.to_string();
    }
    let parts = content_blocks
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|block| block.get("text").and_then(serde_json::Value::as_str))
        .filter(|text| !text.trim().is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    if parts.is_empty() {
        "[empty message]".to_string()
    } else {
        parts.join("\n\n")
    }
}

fn canonical_conversation_reindex_text(title: Option<&str>, summary: &str) -> String {
    let title = title
        .filter(|title| !title.is_empty())
        .unwrap_or("Untitled Conversation");
    format!("{title}\n\n{summary}")
}

impl Repository {
    pub async fn create_processing_job(
        &self,
        job_type: ProcessingJobType,
        source_generation: Option<String>,
        total_count: u64,
    ) -> Result<ProcessingJob> {
        self.create_processing_job_with_scope(
            job_type,
            source_generation,
            total_count,
            None,
            Vec::new(),
        )
        .await
    }

    /// Persist the exact item set selected for a bounded job. Resume uses this
    /// durable scope instead of silently substituting a new page of notes.
    pub async fn create_processing_job_with_scope(
        &self,
        job_type: ProcessingJobType,
        source_generation: Option<String>,
        total_count: u64,
        scope: Option<String>,
        item_ids: Vec<String>,
    ) -> Result<ProcessingJob> {
        let job: Option<ProcessingJob> = self
            .db
            .query(
                "CREATE processing_job SET job_type = $job_type, source_generation = $source_generation, scope = $scope, item_ids = $item_ids, \
                 status = 'running', total_count = $total_count, completed_count = 0, failed_count = 0, \
                 checkpoint = NONE, last_error = NONE, created_at = time::now(), updated_at = time::now(), \
                 finished_at = NONE RETURN AFTER",
            )
            .bind(("job_type", job_type.as_str()))
            .bind(("source_generation", source_generation))
            .bind(("scope", scope))
            .bind(("item_ids", item_ids))
            .bind(("total_count", count_to_i64(total_count)?))
            .await?
            .take(0)?;
        job.ok_or_else(|| DbError::QueryFailed("create_processing_job".into()))
    }

    /// Persist a reindex job together with the exact embedding identity it is
    /// allowed to stage and promote. This is intentionally separate from the
    /// generic inference-job constructor: regular jobs have no corpus-wide
    /// model cutover contract.
    pub async fn create_reindex_processing_job(
        &self,
        total_count: u64,
        scope: String,
        item_ids: Vec<String>,
        embedding: &EmbeddingIdentity,
        item_fingerprints: std::collections::BTreeMap<String, String>,
    ) -> Result<ProcessingJob> {
        let dimension = i64::try_from(embedding.dimension).map_err(|_| {
            DbError::QueryFailed("embedding dimension exceeds database integer range".into())
        })?;
        let job: Option<ProcessingJob> = self
            .db
            .query(
                "CREATE processing_job SET job_type = 'reindex', scope = $scope, item_ids = $item_ids, \
                 target_embedding_provider = $provider, target_embedding_model = $model, target_embedding_dimension = $dimension, \
                 reindex_item_fingerprints = $fingerprints, \
                 status = 'queued', total_count = $total_count, completed_count = 0, failed_count = 0, \
                 checkpoint = NONE, last_error = NONE, created_at = time::now(), updated_at = time::now(), \
                 finished_at = NONE RETURN AFTER",
            )
            .bind(("scope", scope))
            .bind(("item_ids", item_ids))
            .bind(("provider", embedding.provider.clone()))
            .bind(("model", embedding.model.clone()))
            .bind(("dimension", dimension))
            .bind(("fingerprints", item_fingerprints))
            .bind(("total_count", count_to_i64(total_count)?))
            .await?
            .take(0)?;
        job.ok_or_else(|| DbError::QueryFailed("create reindex processing_job".into()))
    }

    /// Replace the job's fingerprint snapshot only after the corresponding
    /// staged vectors committed. A crash before this write repeats work; it
    /// never treats a stale staged vector as current.
    pub async fn update_reindex_item_fingerprints(
        &self,
        id: &RecordId,
        fingerprints: std::collections::BTreeMap<String, String>,
    ) -> Result<()> {
        self.db
            .query("UPDATE $id SET reindex_item_fingerprints = $fingerprints, updated_at = time::now()")
            .bind(("id", id.clone()))
            .bind(("fingerprints", fingerprints))
            .await?
            .check()?;
        Ok(())
    }

    /// Update a reindex checkpoint only while this worker still owns its
    /// durable lease. A recovered worker must never advance another owner's
    /// checkpoint.
    pub async fn update_reindex_item_fingerprints_owned(
        &self,
        id: &RecordId,
        owner: &str,
        fingerprints: std::collections::BTreeMap<String, String>,
    ) -> Result<()> {
        let job: Option<ProcessingJob> = self
            .db
            .query(
                "UPDATE $id SET reindex_item_fingerprints = $fingerprints, updated_at = time::now() \
                 WHERE job_type = 'reindex' AND status = 'running' AND reindex_lease_owner = $owner \
                 AND reindex_lease_expires_at >= time::now() RETURN AFTER",
            )
            .bind(("id", id.clone()))
            .bind(("owner", owner.to_string()))
            .bind(("fingerprints", fingerprints))
            .await?
            .take(0)?;
        if job.is_none() {
            return Err(DbError::NotFound(
                "owned running reindex processing_job".into(),
                record_id_to_string(id),
            ));
        }
        Ok(())
    }

    /// Atomically acquire a reindex job. A live lease cannot be stolen; an
    /// expired lease represents a worker killed between durable checkpoints.
    pub async fn claim_reindex_processing_job(
        &self,
        id: &RecordId,
        owner: &str,
        lease_expires_at: DateTime<Utc>,
    ) -> Result<ProcessingJob> {
        let job: Option<ProcessingJob> = self
            .db
            .query(
                "UPDATE $id SET status = 'running', reindex_lease_owner = $owner, \
                 reindex_lease_expires_at = $lease_expires_at, last_error = NONE, finished_at = NONE, \
                 updated_at = time::now() WHERE job_type = 'reindex' AND (status = 'queued' OR \
                 status = 'cancelled' OR status = 'failed' OR (status = 'running' AND \
                 (reindex_lease_owner = $owner OR reindex_lease_expires_at IS NONE OR \
                 reindex_lease_expires_at < time::now()))) RETURN AFTER",
            )
            .bind(("id", id.clone()))
            .bind(("owner", owner.to_string()))
            .bind(("lease_expires_at", lease_expires_at))
            .await?
            .take(0)?;
        job.ok_or_else(|| {
            DbError::NotFound(
                "claimable reindex processing_job".into(),
                record_id_to_string(id),
            )
        })
    }

    /// Renew a running reindex lease. A false result means another worker has
    /// recovered the job or the lease expired before this worker could renew.
    pub async fn renew_reindex_processing_job_lease(
        &self,
        id: &RecordId,
        owner: &str,
        lease_expires_at: DateTime<Utc>,
    ) -> Result<bool> {
        let job: Option<ProcessingJob> = self
            .db
            .query(
                "UPDATE $id SET reindex_lease_expires_at = $lease_expires_at, updated_at = time::now() \
                 WHERE job_type = 'reindex' AND status = 'running' AND reindex_lease_owner = $owner \
                 AND reindex_lease_expires_at >= time::now() RETURN AFTER",
            )
            .bind(("id", id.clone()))
            .bind(("owner", owner.to_string()))
            .bind(("lease_expires_at", lease_expires_at))
            .await?
            .take(0)?;
        Ok(job.is_some())
    }

    /// Transfer still-valid staged rows to a recovered worker. The exact
    /// source snapshot condition intentionally leaves edited rows for the new
    /// worker to re-embed rather than adopting a stale vector.
    pub async fn adopt_reindex_staging(&self, item_ids: &[String], owner: &str) -> Result<()> {
        let mut notes = Vec::new();
        let mut messages = Vec::new();
        let mut conversations = Vec::new();
        for id in item_ids {
            let record = parse_record_id(id, None)?;
            match record.table.as_str() {
                "note" => notes.push(record),
                "message" => messages.push(record),
                "conversation" => conversations.push(record),
                table => {
                    return Err(DbError::QueryFailed(format!(
                        "{table} is not a supported reindex record table"
                    )))
                }
            }
        }
        // Each row remains safely retryable until `commit_reindex` publishes
        // the complete generation atomically. Adoption therefore does not
        // need to make unrelated record tables one transaction: splitting
        // these independent, idempotent updates avoids SurrealDB aborting the
        // entire resume when an empty/unrelated table update fails inside a
        // multi-statement transaction. A crash partway through simply leaves
        // some valid staged rows for the next owner to adopt again.
        if !notes.is_empty() {
            self.db
                .query(
                    "UPDATE note SET reindex_staging_owner = $owner WHERE id IN $ids AND reindex_embedding IS NOT NONE AND reindex_source_snapshot.content = content AND reindex_source_snapshot.search_content = search_content AND reindex_source_snapshot.chunk_heading_path = chunk_heading_path",
                )
                .bind(("owner", owner.to_string()))
                .bind(("ids", notes))
                .await?
                .check()?;
        }
        if !messages.is_empty() {
            self.db
                .query(
                    "UPDATE message SET reindex_staging_owner = $owner WHERE id IN $ids AND reindex_embedding IS NOT NONE AND reindex_source_snapshot.content = content AND reindex_source_snapshot.content_blocks = content_blocks",
                )
                .bind(("owner", owner.to_string()))
                .bind(("ids", messages))
                .await?
                .check()?;
        }
        if !conversations.is_empty() {
            self.db
                .query(
                    "UPDATE conversation SET reindex_staging_owner = $owner WHERE id IN $ids AND reindex_summary_embedding IS NOT NONE AND reindex_source_snapshot.title = title AND reindex_source_snapshot.summary = summary",
                )
                .bind(("owner", owner.to_string()))
                .bind(("ids", conversations))
                .await?
                .check()?;
        }
        Ok(())
    }

    /// Update progress only after an item's atomic database mutation has
    /// completed.  A crash can therefore repeat at most the checkpoint item;
    /// the processing callers use idempotent upserts/reconciliation.
    pub async fn update_processing_job(
        &self,
        id: &RecordId,
        update: ProcessingJobUpdate,
    ) -> Result<ProcessingJob> {
        let status = update.status.map(ProcessingJobStatus::as_str);
        let job: Option<ProcessingJob> = self
            .db
            .query(
                "UPDATE $id SET status = IF $status = NONE THEN status ELSE $status END, \
                 completed_count = IF $completed_count = NONE THEN completed_count ELSE $completed_count END, \
                 failed_count = IF $failed_count = NONE THEN failed_count ELSE $failed_count END, \
                 checkpoint = IF $checkpoint_set THEN $checkpoint ELSE checkpoint END, \
                 last_error = IF $last_error_set THEN $last_error ELSE last_error END, \
                 finished_at = IF $finish THEN time::now() ELSE finished_at END, updated_at = time::now() \
                 RETURN AFTER",
            )
            .bind(("id", id.clone()))
            .bind(("status", status))
            .bind((
                "completed_count",
                update.completed_count.map(count_to_i64).transpose()?,
            ))
            .bind(("failed_count", update.failed_count.map(count_to_i64).transpose()?))
            .bind(("checkpoint_set", update.checkpoint.is_some()))
            .bind(("checkpoint", update.checkpoint.flatten()))
            .bind(("last_error_set", update.last_error.is_some()))
            .bind(("last_error", update.last_error.flatten()))
            .bind(("finish", update.finish))
            .await?
            .take(0)?;
        job.ok_or_else(|| DbError::NotFound("processing_job".into(), record_id_to_string(id)))
    }

    /// Apply a reindex worker update only if its lease is still live.
    pub async fn update_owned_reindex_processing_job(
        &self,
        id: &RecordId,
        owner: &str,
        update: ProcessingJobUpdate,
    ) -> Result<ProcessingJob> {
        let status = update.status.map(ProcessingJobStatus::as_str);
        let job: Option<ProcessingJob> = self
            .db
            .query(
                "UPDATE $id SET status = IF $status = NONE THEN status ELSE $status END, \
                 completed_count = IF $completed_count = NONE THEN completed_count ELSE $completed_count END, \
                 failed_count = IF $failed_count = NONE THEN failed_count ELSE $failed_count END, \
                 checkpoint = IF $checkpoint_set THEN $checkpoint ELSE checkpoint END, \
                 last_error = IF $last_error_set THEN $last_error ELSE last_error END, \
                 finished_at = IF $finish THEN time::now() ELSE finished_at END, updated_at = time::now() \
                 WHERE job_type = 'reindex' AND status = 'running' AND reindex_lease_owner = $owner \
                 AND reindex_lease_expires_at >= time::now() RETURN AFTER",
            )
            .bind(("id", id.clone()))
            .bind(("owner", owner.to_string()))
            .bind(("status", status))
            .bind((
                "completed_count",
                update.completed_count.map(count_to_i64).transpose()?,
            ))
            .bind(("failed_count", update.failed_count.map(count_to_i64).transpose()?))
            .bind(("checkpoint_set", update.checkpoint.is_some()))
            .bind(("checkpoint", update.checkpoint.flatten()))
            .bind(("last_error_set", update.last_error.is_some()))
            .bind(("last_error", update.last_error.flatten()))
            .bind(("finish", update.finish))
            .await?
            .take(0)?;
        job.ok_or_else(|| {
            DbError::NotFound(
                "owned running reindex processing_job".into(),
                record_id_to_string(id),
            )
        })
    }

    pub async fn get_processing_job(&self, id: &str) -> Result<Option<ProcessingJob>> {
        let raw = id.strip_prefix("processing_job:").unwrap_or(id);
        Ok(self.db.select(("processing_job", raw)).await?)
    }

    pub async fn list_processing_jobs(&self, limit: usize) -> Result<Vec<ProcessingJob>> {
        let limit = i64::try_from(limit)
            .map_err(|_| DbError::QueryFailed("job limit exceeds database integer range".into()))?;
        Ok(self
            .db
            .query("SELECT * FROM processing_job ORDER BY updated_at DESC LIMIT $limit")
            .bind(("limit", limit))
            .await?
            .take(0)?)
    }

    /// A cancelled job is never silently restarted. `resume` must explicitly
    /// move it back to running after the caller has reconstructed the work.
    pub async fn cancel_processing_job(&self, id: &RecordId) -> Result<ProcessingJob> {
        let job: Option<ProcessingJob> = self
            .db
            .query(
                "UPDATE $id SET status = 'cancelled', updated_at = time::now(), finished_at = time::now() \
                 WHERE status = 'running' OR status = 'queued' RETURN AFTER",
            )
            .bind(("id", id.clone()))
            .await?
            .take(0)?;
        job.ok_or_else(|| {
            DbError::NotFound("runnable processing_job".into(), record_id_to_string(id))
        })
    }

    pub async fn resume_processing_job(&self, id: &RecordId) -> Result<ProcessingJob> {
        let job: Option<ProcessingJob> = self
            .db
            .query(
                "UPDATE $id SET status = 'running', last_error = NONE, finished_at = NONE, updated_at = time::now() \
                 WHERE status = 'cancelled' OR status = 'failed' RETURN AFTER",
            )
            .bind(("id", id.clone()))
            .await?
            .take(0)?;
        job.ok_or_else(|| {
            DbError::NotFound("resumable processing_job".into(), record_id_to_string(id))
        })
    }

    /// Snapshot the exact visible corpus selected for a reindex job. The
    /// durable job stores these logical IDs so a resume never silently picks
    /// up unrelated later records.
    pub async fn snapshot_reindex_item_ids(
        &self,
        notes: bool,
        messages: bool,
        summaries: bool,
    ) -> Result<Vec<String>> {
        let mut ids = Vec::new();
        if notes {
            // A corpus-wide identity cutover must cover hidden/pending file
            // generations too. Otherwise a later source promotion could make
            // an old-model vector visible under freshly committed metadata.
            let note_query = if messages && summaries {
                "SELECT VALUE id FROM note ORDER BY id ASC".to_string()
            } else {
                format!("SELECT VALUE id FROM note WHERE {VISIBLE_NOTE_CONDITION} ORDER BY id ASC")
            };
            ids.extend(self.reindex_ids_for_query(&note_query).await?);
        }
        if messages {
            ids.extend(
                self.reindex_ids_for_query("SELECT VALUE id FROM message ORDER BY id ASC")
                    .await?,
            );
        }
        if summaries {
            ids.extend(
                self.reindex_ids_for_query(
                    "SELECT VALUE id FROM conversation WHERE summary IS NOT NONE AND summary != '' ORDER BY id ASC",
                )
                .await?,
            );
        }
        Ok(ids)
    }

    /// Check whether a corpus-wide reindex snapshot has been widened by newly
    /// eligible records. The cutover transaction repeats this check atomically;
    /// this preflight gives the worker an actionable restart error instead of
    /// leaving a failed job with an opaque database transaction message.
    pub async fn full_reindex_scope_widened(&self, item_ids: &[String]) -> Result<bool> {
        let expected = item_ids.iter().collect::<HashSet<_>>();
        Ok(self
            .snapshot_reindex_item_ids(true, true, true)
            .await?
            .iter()
            .any(|id| !expected.contains(id)))
    }

    async fn reindex_ids_for_query(&self, query: &str) -> Result<Vec<String>> {
        let ids: Vec<RecordId> = self.db.query(query).await?.take(0)?;
        Ok(ids.iter().map(record_id_to_string).collect())
    }

    /// Load one previously snapshotted reindex item. A record removed after
    /// snapshot is reported as absent and can be reconciled as completed by
    /// the worker without widening its durable scope.
    pub async fn get_reindex_item(&self, id: &str) -> Result<Option<ReindexItem>> {
        let record = parse_record_id(id, None)?;
        match record.table.as_str() {
            "note" => {
                #[derive(Deserialize, SurrealValue)]
                struct NoteRow {
                    id: RecordId,
                    content: String,
                    #[serde(default)]
                    search_content: Option<String>,
                    #[serde(default)]
                    chunk_heading_path: Vec<String>,
                }
                let row: Option<NoteRow> = self
                    .db
                    .query("SELECT id, content, search_content, chunk_heading_path FROM $id")
                    .bind(("id", record))
                    .await?
                    .take(0)?;
                Ok(row.map(|row| ReindexItem {
                    id: record_id_to_string(&row.id),
                    text: canonical_note_reindex_text(
                        &row.content,
                        row.search_content.as_deref(),
                        &row.chunk_heading_path,
                    ),
                    source_snapshot: serde_json::json!({
                        "content": row.content,
                        "search_content": row.search_content,
                        "chunk_heading_path": row.chunk_heading_path,
                    }),
                }))
            }
            "message" => {
                #[derive(Deserialize, SurrealValue)]
                struct MessageRow {
                    id: RecordId,
                    content: String,
                    #[serde(default)]
                    content_blocks: serde_json::Value,
                }
                let row: Option<MessageRow> = self
                    .db
                    .query("SELECT id, content, content_blocks FROM $id")
                    .bind(("id", record))
                    .await?
                    .take(0)?;
                Ok(row.map(|row| ReindexItem {
                    id: record_id_to_string(&row.id),
                    text: canonical_message_reindex_text(&row.content, &row.content_blocks),
                    source_snapshot: serde_json::json!({
                        "content": row.content,
                        "content_blocks": row.content_blocks,
                    }),
                }))
            }
            "conversation" => {
                #[derive(Deserialize, SurrealValue)]
                struct ConversationRow {
                    id: RecordId,
                    #[serde(default)]
                    title: Option<String>,
                    summary: String,
                }
                let row: Option<ConversationRow> = self
                    .db
                    .query("SELECT id, title, summary FROM $id WHERE summary IS NOT NONE AND summary != ''")
                    .bind(("id", record))
                    .await?
                    .take(0)?;
                Ok(row.map(|row| ReindexItem {
                    id: record_id_to_string(&row.id),
                    text: canonical_conversation_reindex_text(row.title.as_deref(), &row.summary),
                    source_snapshot: serde_json::json!({
                        "title": row.title,
                        "summary": row.summary,
                    }),
                }))
            }
            table => Err(DbError::QueryFailed(format!(
                "{table} is not a supported reindex record table"
            ))),
        }
    }

    /// Persist a newly computed vector in the inactive reindex field. Search
    /// continues to read the old active vector until [`Self::commit_reindex`]
    /// validates and swaps the full selected generation.
    pub async fn stage_reindex_embedding(
        &self,
        item: &ReindexItem,
        embedding: Vec<f32>,
        owner: &str,
    ) -> Result<()> {
        let record = parse_record_id(&item.id, None)?;
        let query = match record.table.as_str() {
            "note" => "UPDATE $id SET reindex_embedding = $embedding, reindex_source_text = $text, reindex_source_snapshot = $snapshot, reindex_staging_owner = $owner WHERE content = $snapshot.content AND search_content = $snapshot.search_content AND chunk_heading_path = $snapshot.chunk_heading_path RETURN AFTER",
            "message" => "UPDATE $id SET reindex_embedding = $embedding, reindex_source_text = $text, reindex_source_snapshot = $snapshot, reindex_staging_owner = $owner WHERE content = $snapshot.content AND content_blocks = $snapshot.content_blocks RETURN AFTER",
            "conversation" => "UPDATE $id SET reindex_summary_embedding = $embedding, reindex_source_text = $text, reindex_source_snapshot = $snapshot, reindex_staging_owner = $owner WHERE title = $snapshot.title AND summary = $snapshot.summary RETURN AFTER",
            table => {
                return Err(DbError::QueryFailed(format!(
                    "{table} is not a supported reindex record table"
                )))
            }
        };
        #[derive(Deserialize, SurrealValue)]
        struct StagedRow {
            id: RecordId,
        }
        let staged: Option<StagedRow> = self
            .db
            .query(query)
            .bind(("id", record))
            .bind(("embedding", embedding))
            .bind(("text", item.text.clone()))
            .bind(("snapshot", item.source_snapshot.clone()))
            .bind(("owner", owner.to_string()))
            .await?
            .take(0)?;
        if staged.is_none() {
            return Err(DbError::QueryFailed(
                "reindex item changed or disappeared while its embedding was being staged".into(),
            ));
        }
        Ok(())
    }

    /// Atomically publish all staged vectors and the corresponding model
    /// identity. Any provider failure or cancellation before this method leaves
    /// every active vector and metadata field on the last known-good model.
    #[allow(clippy::too_many_arguments)]
    pub async fn commit_reindex(
        &self,
        job_id: &RecordId,
        owner: &str,
        item_ids: &[String],
        embedding: &EmbeddingIdentity,
        clear_entity_embeddings: bool,
        validate_full_scope_membership: bool,
        completed_count: u64,
    ) -> Result<()> {
        let mut notes = Vec::new();
        let mut messages = Vec::new();
        let mut conversations = Vec::new();
        for id in item_ids {
            let record = parse_record_id(id, None)?;
            match record.table.as_str() {
                "note" => notes.push(record),
                "message" => messages.push(record),
                "conversation" => conversations.push(record),
                table => {
                    return Err(DbError::QueryFailed(format!(
                        "{table} is not a supported reindex record table"
                    )))
                }
            }
        }
        let dimension = i64::try_from(embedding.dimension).map_err(|_| {
            DbError::QueryFailed("embedding dimension exceeds database integer range".into())
        })?;
        let completed_count = count_to_i64(completed_count)?;
        self.db
            .query(
                "BEGIN TRANSACTION; \
                 LET $job_valid = (SELECT VALUE count() FROM processing_job WHERE id = $job_id AND job_type = 'reindex' AND status = 'running' AND reindex_lease_owner = $owner AND reindex_lease_expires_at >= time::now() GROUP ALL)[0] = 1; \
                 LET $notes_expected = (SELECT VALUE count() FROM note WHERE id IN $notes GROUP ALL)[0]; \
                 LET $messages_expected = (SELECT VALUE count() FROM message WHERE id IN $messages GROUP ALL)[0]; \
                 LET $conversations_expected = (SELECT VALUE count() FROM conversation WHERE id IN $conversations GROUP ALL)[0]; \
                 LET $notes_widened = (SELECT VALUE count() FROM note WHERE id NOT IN $notes GROUP ALL)[0] != 0; \
                 LET $messages_widened = (SELECT VALUE count() FROM message WHERE id NOT IN $messages GROUP ALL)[0] != 0; \
                 LET $conversations_widened = (SELECT VALUE count() FROM conversation WHERE summary IS NOT NONE AND summary != '' AND id NOT IN $conversations GROUP ALL)[0] != 0; \
                 LET $notes_valid = (SELECT VALUE count() FROM note WHERE id IN $notes AND reindex_embedding IS NOT NONE AND reindex_source_snapshot.content = content AND reindex_source_snapshot.search_content = search_content AND reindex_source_snapshot.chunk_heading_path = chunk_heading_path AND reindex_staging_owner = $owner GROUP ALL)[0] = $notes_expected; \
                 LET $messages_valid = (SELECT VALUE count() FROM message WHERE id IN $messages AND reindex_embedding IS NOT NONE AND reindex_source_snapshot.content = content AND reindex_source_snapshot.content_blocks = content_blocks AND reindex_staging_owner = $owner GROUP ALL)[0] = $messages_expected; \
                 LET $conversations_valid = (SELECT VALUE count() FROM conversation WHERE id IN $conversations AND reindex_summary_embedding IS NOT NONE AND reindex_source_snapshot.title = title AND reindex_source_snapshot.summary = summary AND reindex_staging_owner = $owner GROUP ALL)[0] = $conversations_expected; \
                 IF !$job_valid OR !$notes_valid OR !$messages_valid OR !$conversations_valid THEN THROW 'reindex staging is stale, incomplete, or no longer owned' END; \
                 IF $validate_full_scope_membership AND ($notes_widened OR $messages_widened OR $conversations_widened) THEN THROW 'full reindex scope widened after snapshot; start a new reindex job' END; \
                 UPDATE note SET embedding = reindex_embedding, reindex_embedding = NONE, reindex_source_text = NONE, reindex_staging_owner = NONE WHERE id IN $notes; \
                 UPDATE message SET embedding = reindex_embedding, reindex_embedding = NONE, reindex_source_text = NONE, reindex_staging_owner = NONE WHERE id IN $messages; \
                 UPDATE conversation SET summary_embedding = reindex_summary_embedding, reindex_summary_embedding = NONE, reindex_source_text = NONE, reindex_staging_owner = NONE WHERE id IN $conversations; \
                 UPDATE entity SET embedding = NONE WHERE $clear_entity_embeddings AND embedding IS NOT NONE; \
                 UPSERT graphrag_metadata SET key = 'active_embedding', \
                     application_schema_version = $schema_version, embedding_provider = $provider, \
                     embedding_model = $model, embedding_dimension = $dimension, \
                     generation = IF generation = NONE THEN 1 ELSE generation + 1 END, last_reindex_at = time::now(), \
                     last_reindex_status = 'completed', updated_at = time::now() WHERE key = 'active_embedding'; \
                 UPDATE $job_id SET status = 'completed', completed_count = $completed_count, checkpoint = NONE, last_error = NONE, finished_at = time::now(), updated_at = time::now() \
                     WHERE job_type = 'reindex' AND status = 'running' AND reindex_lease_owner = $owner AND reindex_lease_expires_at >= time::now(); \
                 COMMIT TRANSACTION;",
            )
            .bind(("job_id", job_id.clone()))
            .bind(("owner", owner.to_string()))
            .bind(("notes", notes))
            .bind(("messages", messages))
            .bind(("conversations", conversations))
            .bind(("schema_version", i64::from(migrations::latest_version())))
            .bind(("provider", embedding.provider.clone()))
            .bind(("model", embedding.model.clone()))
            .bind(("dimension", dimension))
            .bind(("clear_entity_embeddings", clear_entity_embeddings))
            .bind((
                "validate_full_scope_membership",
                validate_full_scope_membership,
            ))
            .bind(("completed_count", completed_count))
            .await?
            .check()?;
        Ok(())
    }
}

impl Repository {
    /// Read a durable local inference result by its fully semantic cache key.
    pub async fn get_inference_cache(&self, cache_key: &str) -> Result<Option<serde_json::Value>> {
        #[derive(Deserialize, SurrealValue)]
        struct CacheValue {
            cache_value: serde_json::Value,
        }
        let row: Option<CacheValue> = self
            .db
            .query("SELECT cache_value FROM inference_cache WHERE cache_key = $cache_key LIMIT 1")
            .bind(("cache_key", cache_key.to_string()))
            .await?
            .take(0)?;
        Ok(row.map(|row| row.cache_value))
    }

    /// Store a JSON result under its semantic key. The unique index makes
    /// concurrent misses converge without changing the cached result.
    pub async fn put_inference_cache(&self, entry: InferenceCacheEntry) -> Result<()> {
        self.db
            .query(
                "UPSERT type::record('inference_cache', $cache_key) SET cache_key = $cache_key, operation = $operation, \
                 provider = $provider, model = $model, version = $version, input_hash = $input_hash, \
                 cache_value = $cache_value, updated_at = time::now(), created_at = IF created_at = NONE THEN time::now() ELSE created_at END",
            )
            .bind(("cache_key", entry.cache_key))
            .bind(("operation", entry.operation))
            .bind(("provider", entry.provider))
            .bind(("model", entry.model))
            .bind(("version", entry.version))
            .bind(("input_hash", entry.input_hash))
            .bind(("cache_value", entry.value))
            .await?
            .check()?;
        Ok(())
    }
}
