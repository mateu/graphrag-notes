//! Private extraction checkpoints for an explicitly reviewed source-policy migration.
//! The previous successful generation remains visible until every batch is ready.
use super::*;
use graphrag_core::EntityType;

const MAX_PREPARED_BYTES: usize = 16 * 1024 * 1024;
const MAX_ENTITIES_PER_CHUNK: usize = 128;

#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
struct PreparedGraph {
    note: RecordId,
    scope: Option<String>,
    #[serde(serialize_with = "serialize_entities")]
    entities: Vec<Entity>,
    entity_names: Vec<String>,
}

fn serialize_entities<S: serde::Serializer>(
    entities: &[Entity],
    serializer: S,
) -> std::result::Result<S::Ok, S::Error> {
    let values = entities
        .iter()
        .map(|entity| {
            let mut value = serde_json::to_value(entity).map_err(serde::ser::Error::custom)?;
            value["created_at"] =
                serde_json::to_value(entity.created_at).map_err(serde::ser::Error::custom)?;
            Ok(value)
        })
        .collect::<std::result::Result<Vec<_>, S::Error>>()?;
    values.serialize(serializer)
}

fn validate_entities(note: &Note, scope: Option<&str>, entities: &[Entity]) -> Result<()> {
    let mut identities = HashSet::new();
    let note_id = note
        .id
        .as_ref()
        .ok_or_else(|| DbError::InvalidRemoteRequest("migration note lacks an identity".into()))?;
    let note_scope = scope
        .map(str::to_owned)
        .unwrap_or_else(|| record_id_to_string(note_id));
    let source_scope =
        record_id_to_string(note.source_id.as_ref().ok_or_else(|| {
            DbError::InvalidRemoteRequest("migration note lacks a source".into())
        })?);
    for entity in entities {
        let scope = if matches!(entity.entity_type, EntityType::Person | EntityType::Project) {
            &note_scope
        } else {
            &source_scope
        };
        let identity =
            serde_json::json!([scope, entity.entity_type, entity.canonical_name]).to_string();
        let expected = format!(
            "extracted-v1:{}",
            graphrag_core::normalized_content_hash(&identity)
        );
        if entity.id.is_some()
            || entity.name.trim().is_empty()
            || entity.name.chars().count() > 80
            || entity.name.chars().any(char::is_control)
            || entity.canonical_name != Entity::canonicalize(&entity.name)
            || entity.identity_key.as_deref() != Some(expected.as_str())
            || entity.metadata["extraction"]["scope"].as_str() != Some(scope.as_str())
            || !identities.insert(expected)
            || (!entity.embedding.is_empty() && entity.embedding.len() != 1024)
            || entity.embedding.iter().any(|value| !value.is_finite())
        {
            return Err(DbError::InvalidRemoteRequest(
                "prepared entities must retain bounded typed identity and exact scope".into(),
            ));
        }
    }
    Ok(())
}

fn prepared(job: &RemoteUploadJob) -> Result<Vec<PreparedGraph>> {
    let value = job
        .result
        .as_ref()
        .ok_or_else(|| DbError::InvalidRemoteRequest("migration checkpoint missing".into()))?;
    if value["policy_migration_stage"]["version"] != 1
        || serde_json::to_vec(value)
            .map_err(|error| DbError::InvalidRemoteRequest(error.to_string()))?
            .len()
            > MAX_PREPARED_BYTES
    {
        return Err(DbError::InvalidRemoteRequest(
            "migration checkpoint exceeds its bound or version".into(),
        ));
    }
    let batches: Vec<PreparedGraph> =
        serde_json::from_value(value["policy_migration_stage"]["batches"].clone())
            .map_err(|error| DbError::InvalidRemoteRequest(error.to_string()))?;
    if batches.len() > MAX_REMOTE_UPLOAD_CHUNKS
        || batches.len() != job.job.completed_count as usize
        || batches.iter().enumerate().any(|(index, batch)| {
            job.job.item_ids.get(index) != Some(&record_id_to_string(&batch.note))
                || batch.entities.len() > MAX_ENTITIES_PER_CHUNK
                || batch
                    .entities
                    .iter()
                    .map(Entity::effective_identity_key)
                    .collect::<Vec<_>>()
                    != batch.entity_names
                || batch
                    .entities
                    .iter()
                    .any(|entity| entity.embedding.iter().any(|value| !value.is_finite()))
        })
    {
        return Err(DbError::InvalidRemoteRequest(
            "migration checkpoint does not match exact prepared items".into(),
        ));
    }
    Ok(batches)
}

/// The reviewed original guard is shared by first generation preparation and
/// an unprepared failed admission's explicit resume. Prepared resumptions use
/// their existing lease/generation ownership instead of the opening ready state.
pub(super) fn matches_origin(
    source: &Source,
    original: &RemoteUploadJob,
    input: &RemoteUploadInput,
) -> Result<bool> {
    let Some(intent) = &input.policy_migration else {
        return Ok(false);
    };
    let mut prior = original.input.source_provenance.clone();
    let mut supplied = input.source_provenance.clone();
    let collection = prior["metadata"]["collection_id"].as_str();
    let collection_matches = collection
        .is_some_and(|value| !value.is_empty() && supplied["metadata"]["collection_id"] == value);
    for provenance in [&mut prior, &mut supplied] {
        if let Some(metadata) = provenance["metadata"].as_object_mut() {
            metadata.remove("original_sha256");
            metadata.remove("parts");
        }
    }
    Ok(source.source_type == SourceType::Markdown
        && source.status == SourceIngestionStatus::Ready
        && source.generation == source.successful_generation
        && source.metadata["remote_upload_retired"] != true
        && source.id == original.source_id
        && source.uri.as_deref() == Some(original.source_uri.as_str())
        && original.instance_id == input.authenticated_instance_id
        && original.input.document_key == input.document_key
        && original.job.status == "completed"
        && original.input.extract_entities
        && collection_matches
        && prior == supplied
        && intent["original_policy_sha256"]
            == uploaded_processing_policy_sha256(&original.input.processing_options)?
        && intent["target_policy_sha256"]
            == uploaded_processing_policy_sha256(&input.processing_options)?
        && uploaded_processing_compatible(
            &original.input.processing_options,
            &input.processing_options,
            false,
            false,
        ))
}

impl Repository {
    pub async fn prepare_policy_migration_extraction(
        &self,
        lease: &RemoteJobLease,
        successors: &[(RecordId, RecordId, bool)],
    ) -> Result<RemoteUploadJob> {
        let _gate = self.remote_job_transition_lock.lock().await;
        let _lifecycle = self.proposal_acceptance_lock.lock().await;
        let job = self.owned_remote_upload(lease, false).await?;
        self.ensure_remote_source_current(&job).await?;
        if job.input.policy_migration.is_none() || job.phase != "migration_staged" {
            return Err(DbError::RemoteJobOwnershipLost(record_id_to_string(
                &lease.job_id,
            )));
        }
        let source = self
            .get_source(&job.source_uri)
            .await?
            .ok_or_else(|| DbError::RemoteJobSourceConflict(record_id_to_string(&lease.job_id)))?;
        if source.successful_generation >= source.generation {
            return Err(DbError::RemoteJobSourceConflict(record_id_to_string(
                &lease.job_id,
            )));
        }
        let mut scopes = Vec::new();
        let mut old_ids = HashSet::new();
        let mut new_ids = HashSet::new();
        for (old_id, new_id, _) in successors {
            if !old_ids.insert(record_id_to_string(old_id))
                || !new_ids.insert(record_id_to_string(new_id))
            {
                return Err(DbError::InvalidRemoteRequest(
                    "migration lineage must be one-to-one".into(),
                ));
            }
            let old: Note = self.db.select(old_id.clone()).await?.ok_or_else(|| {
                DbError::RemoteJobSourceConflict(record_id_to_string(&lease.job_id))
            })?;
            let new: Note = self.db.select(new_id.clone()).await?.ok_or_else(|| {
                DbError::RemoteJobSourceConflict(record_id_to_string(&lease.job_id))
            })?;
            if old.source_id != job.source_id
                || new.source_id != job.source_id
                || old.source_generation != Some(source.successful_generation)
                || new.source_generation != job.source_generation
                || !job.job.item_ids.contains(&record_id_to_string(new_id))
            {
                return Err(DbError::RemoteJobSourceConflict(record_id_to_string(
                    &lease.job_id,
                )));
            }
            // Adopt validated prior scope on hidden notes only. A failed migration
            // must not even rewrite the opening note's revision/anchor.
            scopes.push((new_id.clone(), self.note_extraction_scope(&old).await?));
        }
        let result = serde_json::json!({"policy_migration_stage":{"version":1,"batches":[]}});
        let mut response = self.db.query(format!("BEGIN TRANSACTION; {} FOR $item IN $scopes {{ LET $target = $item[0]; UPDATE $target SET extraction_scope = $item[1]; }}; UPDATE $job SET remote_phase = 'migration_extracting', completed_count = 0, checkpoint = NONE, remote_result = $result; COMMIT TRANSACTION;", guard_sql()))
            .bind(("job", lease.job_id.clone())).bind(("instance", lease.instance_id.clone())).bind(("epoch", lease.service_epoch.clone())).bind(("worker", lease.worker_token.clone()))
            .bind(("scopes", scopes)).bind(("result", result)).await?;
        check_write(response.take_errors(), lease)?;
        self.owned_remote_upload(lease, true).await
    }

    pub async fn checkpoint_policy_migration_entities(
        &self,
        lease: &RemoteJobLease,
        index: usize,
        entities: Vec<Entity>,
    ) -> Result<RemoteUploadJob> {
        let _gate = self.remote_job_transition_lock.lock().await;
        let _lifecycle = self.proposal_acceptance_lock.lock().await;
        let job = self.owned_remote_upload(lease, false).await?;
        self.ensure_remote_source_current(&job).await?;
        if job.input.policy_migration.is_none() || job.phase != "migration_extracting" {
            return Err(DbError::RemoteJobOwnershipLost(record_id_to_string(
                &lease.job_id,
            )));
        }
        let mut batches = prepared(&job)?;
        if index < batches.len() {
            return Ok(job);
        }
        if index != batches.len()
            || index >= job.job.item_ids.len()
            || entities.len() > MAX_ENTITIES_PER_CHUNK
        {
            return Err(DbError::InvalidRemoteRequest(
                "migration extraction must advance one exact bounded item".into(),
            ));
        }
        let id = parse_record_id(&job.job.item_ids[index], Some("note"))?;
        let note: Note =
            self.db.select(id.clone()).await?.ok_or_else(|| {
                DbError::RemoteJobSourceConflict(record_id_to_string(&lease.job_id))
            })?;
        if note.source_id != job.source_id || note.source_generation != job.source_generation {
            return Err(DbError::RemoteJobSourceConflict(record_id_to_string(
                &lease.job_id,
            )));
        }
        let scope = self.note_extraction_scope(&note).await?;
        validate_entities(&note, scope.as_deref(), &entities)?;
        let names = entities
            .iter()
            .map(Entity::effective_identity_key)
            .collect();
        batches.push(PreparedGraph {
            note: id,
            scope,
            entities,
            entity_names: names,
        });
        let result = serde_json::json!({"policy_migration_stage":{"version":1,"batches":batches}});
        if serde_json::to_vec(&result)
            .map_err(|error| DbError::InvalidRemoteRequest(error.to_string()))?
            .len()
            > MAX_PREPARED_BYTES
        {
            return Err(DbError::InvalidRemoteRequest(
                "prepared migration graph exceeds 16 MiB".into(),
            ));
        }
        let mut response = self.db.query(format!("BEGIN TRANSACTION; {} UPDATE $job SET completed_count += 1, checkpoint = $checkpoint, remote_result = $result; COMMIT TRANSACTION;", guard_sql()))
            .bind(("job", lease.job_id.clone())).bind(("instance", lease.instance_id.clone())).bind(("epoch", lease.service_epoch.clone())).bind(("worker", lease.worker_token.clone()))
            .bind(("checkpoint", job.job.item_ids[index].clone())).bind(("result", result)).await?;
        check_write(response.take_errors(), lease)?;
        self.owned_remote_upload(lease, true).await
    }

    pub async fn promote_policy_migration(
        &self,
        lease: &RemoteJobLease,
        successors: &[(RecordId, RecordId, bool)],
    ) -> Result<()> {
        let _gate = self.remote_job_transition_lock.lock().await;
        let _lifecycle = self.proposal_acceptance_lock.lock().await;
        let job = self.owned_remote_upload(lease, false).await?;
        self.ensure_remote_source_current(&job).await?;
        if job.input.policy_migration.is_none() || job.phase != "migration_extracting" {
            return Err(DbError::RemoteJobOwnershipLost(record_id_to_string(
                &lease.job_id,
            )));
        }
        let batches = prepared(&job)?;
        if batches.len() != job.job.item_ids.len() || job.job.failed_count != 0 {
            return Err(DbError::InvalidRemoteRequest(
                "migration graph is not fully prepared".into(),
            ));
        }
        let source = self
            .get_source(&job.source_uri)
            .await?
            .ok_or_else(|| DbError::RemoteJobSourceConflict(record_id_to_string(&lease.job_id)))?;
        if source.successful_generation >= source.generation {
            return Err(DbError::RemoteJobSourceConflict(record_id_to_string(
                &lease.job_id,
            )));
        }
        let mut old_ids = HashSet::new();
        let mut new_ids = HashSet::new();
        for (old_id, new_id, _) in successors {
            if !old_ids.insert(record_id_to_string(old_id))
                || !new_ids.insert(record_id_to_string(new_id))
            {
                return Err(DbError::InvalidRemoteRequest(
                    "migration lineage must be one-to-one".into(),
                ));
            }
            let old: Note = self.db.select(old_id.clone()).await?.ok_or_else(|| {
                DbError::RemoteJobSourceConflict(record_id_to_string(&lease.job_id))
            })?;
            let new: Note = self.db.select(new_id.clone()).await?.ok_or_else(|| {
                DbError::RemoteJobSourceConflict(record_id_to_string(&lease.job_id))
            })?;
            if old.source_id != job.source_id
                || new.source_id != job.source_id
                || old.source_generation != Some(source.successful_generation)
                || new.source_generation != job.source_generation
                || !job.job.item_ids.contains(&record_id_to_string(new_id))
            {
                return Err(DbError::RemoteJobSourceConflict(record_id_to_string(
                    &lease.job_id,
                )));
            }
        }
        for batch in &batches {
            let note: Note = self.db.select(batch.note.clone()).await?.ok_or_else(|| {
                DbError::RemoteJobSourceConflict(record_id_to_string(&lease.job_id))
            })?;
            if note.source_id != job.source_id
                || note.source_generation != job.source_generation
                || self.note_extraction_scope(&note).await? != batch.scope
            {
                return Err(DbError::RemoteJobSourceConflict(record_id_to_string(
                    &lease.job_id,
                )));
            }
            validate_entities(&note, batch.scope.as_deref(), &batch.entities)?;
        }
        for (old_id, new_id, _) in successors {
            let old: Note = self.db.select(old_id.clone()).await?.ok_or_else(|| {
                DbError::RemoteJobSourceConflict(record_id_to_string(&lease.job_id))
            })?;
            let batch = batches
                .iter()
                .find(|batch| batch.note == *new_id)
                .ok_or_else(|| {
                    DbError::RemoteJobSourceConflict(record_id_to_string(&lease.job_id))
                })?;
            if self.note_extraction_scope(&old).await? != batch.scope {
                return Err(DbError::RemoteJobSourceConflict(record_id_to_string(
                    &lease.job_id,
                )));
            }
        }
        // A prior promotion attempt may have copied relationships before its
        // graph/visibility transaction failed. Rebuild those private successor
        // dependents from the current live generation so deleted manual edges
        // and removed chat provenance cannot reappear on a later retry.
        let hidden_notes = job
            .job
            .item_ids
            .iter()
            .map(|id| parse_record_id(id, Some("note")))
            .collect::<Result<Vec<_>>>()?;
        let mut response = self.db.query(format!("BEGIN TRANSACTION; {} LET $owned_notes = (SELECT VALUE id FROM note WHERE id IN $notes AND source_id = $source AND source_generation = $generation); IF array::len($owned_notes) != array::len($notes) {{ THROW '{FENCE}'; }}; DELETE supports WHERE in IN $notes OR out IN $notes; DELETE contradicts WHERE in IN $notes OR out IN $notes; DELETE derived_from WHERE in IN $notes OR out IN $notes; DELETE related_to WHERE in IN $notes OR out IN $notes; DELETE note_from_conversation WHERE in IN $notes; DELETE note_from_message WHERE in IN $notes; COMMIT TRANSACTION;", guard_sql()))
            .bind(("job", lease.job_id.clone())).bind(("instance", lease.instance_id.clone())).bind(("epoch", lease.service_epoch.clone())).bind(("worker", lease.worker_token.clone()))
            .bind(("source", job.source_id.clone())).bind(("generation", job.source_generation))
            .bind(("notes", hidden_notes)).await?;
        check_write(response.take_errors(), lease)?;
        // Live manual relationships are then copied under the lifecycle fence.
        // Old mentions never overwrite the explicitly refreshed graph.
        self.copy_note_dependents_to_successors_with_options_locked(successors, false, false)
            .await?;
        let mut metadata = source.metadata;
        let origin = metadata
            .as_object_mut()
            .and_then(|value| value.remove("remote_upload_pending"))
            .ok_or_else(|| DbError::RemoteJobSourceConflict(record_id_to_string(&lease.job_id)))?;
        metadata["remote_upload"] = origin;
        let mut response = self.db.query(format!("BEGIN TRANSACTION; {} FOR $batch IN $batches {{ LET $note = $batch.note; LET $selected = (SELECT VALUE id FROM note WHERE id = $note AND source_id = $source AND source_generation = $generation); IF array::len($selected) != 1 {{ THROW '{FENCE}'; }}; LET $replacement_entities = $batch.entities; LET $replacement_entity_names = $batch.entity_names; {} DELETE mentions WHERE in = $note; {} IF $batch.scope != NONE {{ UPDATE $note SET extraction_scope = $batch.scope; }}; }}; UPDATE $source SET successful_generation = $generation, status = 'ready', last_error = NONE, metadata = $metadata, updated_at = time::now(), last_ingested_at = time::now(); UPDATE $job SET remote_phase = 'migration_promoted'; COMMIT TRANSACTION;", guard_sql(), super::super::notes::replacement_entities_transaction(), super::super::notes::replacement_mentions_transaction("$note")))
            .bind(("job", lease.job_id.clone())).bind(("instance", lease.instance_id.clone())).bind(("epoch", lease.service_epoch.clone())).bind(("worker", lease.worker_token.clone()))
            .bind(("source", job.source_id)).bind(("generation", job.source_generation)).bind(("metadata", metadata)).bind(("batches", batches)).await?;
        check_write(response.take_errors(), lease)?;
        // Keep proposal acceptance/undo excluded until audit retargeting and
        // old-generation cleanup have observed the new visibility. If cleanup
        // fails, the durable promoted phase retries this boundary on resume.
        let mut promoted = self
            .get_source(&job.source_uri)
            .await?
            .ok_or_else(|| DbError::RemoteJobSourceConflict(record_id_to_string(&lease.job_id)))?;
        self.complete_file_import_locked(&mut promoted)
            .await
            .map(|_| ())
    }
}
