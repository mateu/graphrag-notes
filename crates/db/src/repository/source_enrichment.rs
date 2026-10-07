//! Service-owner development slice: one reviewed vector-only source, no scan or providers.
//! Checkpoints are private source metadata; publication and policy change are atomic.
use super::*;
use sha2::{Digest, Sha256};

fn content_revision(note: &Note) -> Result<String> {
    Ok(format!(
        "{:x}",
        Sha256::digest(
            serde_json::to_vec(&(note, note.created_at, note.updated_at)).map_err(|_| conflict())?
        )
    ))
}

#[cfg(test)]
#[path = "source_enrichment_tests.rs"]
mod tests;

const KEY: &str = "entity_enrichment_v1";
const MAX_STAGE_BYTES: usize = 16 * 1024 * 1024;
const FENCE: &str = "source-enrichment-revision-conflict";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SourceEnrichmentPlan {
    pub instance_id: String,
    pub request_id: String,
    pub source_id: String,
    pub document_key: String,
    pub expected_source_revision: String,
    pub expected_content_sha256: String,
    pub expected_generation: u64,
    pub original_policy_sha256: String,
    /// Exact reviewed server policy snapshot, not provider URLs or caller extraction output.
    pub target_processing_options: serde_json::Value,
    pub confirmed: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SourceEnrichmentStatus {
    pub request_id: String,
    pub status: String,
    pub generation: u64,
    pub completed: usize,
    pub total: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Item {
    id: String,
    revision: String,
    entities: Option<Vec<Entity>>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Stage {
    version: u32,
    plan: SourceEnrichmentPlan,
    status: String,
    prior_origin: serde_json::Value,
    items: Vec<Item>,
}
fn invalid(message: &str) -> DbError {
    DbError::InvalidRemoteRequest(message.into())
}
fn conflict() -> DbError {
    DbError::MutationRevisionConflict(FENCE.into())
}
fn digest_valid(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn plan_valid(plan: &SourceEnrichmentPlan) -> Result<()> {
    for value in [&plan.instance_id, &plan.request_id, &plan.document_key] {
        if value.is_empty()
            || value.len() > 256
            || value.chars().count() > 128
            || value.trim() != value
            || value.chars().any(char::is_control)
        {
            return Err(invalid("bounded exact enrichment identities required"));
        }
    }
    if !plan.confirmed
        || plan.expected_generation == 0
        || !digest_valid(&plan.expected_source_revision)
        || !plan
            .expected_content_sha256
            .strip_prefix("sha256:")
            .is_some_and(digest_valid)
        || !digest_valid(&plan.original_policy_sha256)
        || plan.source_id.len() > 512
        || !plan.target_processing_options.is_object()
        || serde_json::to_vec(&plan.target_processing_options)
            .map_err(|_| invalid("policy serialization"))?
            .len()
            > 65536
    {
        return Err(invalid(
            "confirmed exact source, content, generation and reviewed policies required",
        ));
    }
    let id = parse_record_id(&plan.source_id, Some("source"))?;
    if record_id_to_string(&id) != plan.source_id {
        return Err(invalid("canonical source identity required"));
    }
    Ok(())
}
fn stage_shape(value: &Stage) -> Result<()> {
    plan_valid(&value.plan)?;
    if value.version != 1
        || !matches!(value.status.as_str(), "staged" | "promoted" | "rolled_back")
        || value.items.is_empty()
        || value.items.len() > MAX_REMOTE_UPLOAD_CHUNKS
    {
        return Err(conflict());
    }
    let mut ids = HashSet::new();
    let mut gap = false;
    for item in &value.items {
        if item.id.len() > 512
            || record_id_to_string(&parse_record_id(&item.id, Some("note"))?) != item.id
            || !ids.insert(item.id.clone())
            || !digest_valid(&item.revision)
        {
            return Err(conflict());
        }
        if item.entities.is_none() {
            gap = true;
        } else if gap && value.status == "staged" {
            return Err(conflict());
        }
        if value.status == "promoted" && item.entities.is_none() {
            return Err(conflict());
        }
        if let Some(entities) = &item.entities {
            if entities.len() > 128
                || entities.iter().any(|e| {
                    e.id.is_some()
                        || e.canonical_name != Entity::canonicalize(&e.name)
                        || e.embedding.iter().any(|v| !v.is_finite())
                        || (!e.embedding.is_empty() && e.embedding.len() != 1024)
                })
            {
                return Err(conflict());
            }
        }
    }
    Ok(())
}
fn stage(source: &Source, plan: &SourceEnrichmentPlan) -> Result<Stage> {
    let result: Stage =
        serde_json::from_value(source.metadata[KEY].clone()).map_err(|_| conflict())?;
    stage_shape(&result)?;
    if result.plan != *plan {
        return Err(conflict());
    }
    if serde_json::to_vec(&source.metadata[KEY])
        .map_err(|_| conflict())?
        .len()
        > MAX_STAGE_BYTES
    {
        return Err(conflict());
    }
    let mut expected_origin = result.prior_origin.clone();
    if result.status == "promoted" {
        expected_origin["extract_entities"] = serde_json::json!(true);
        expected_origin["processing_options"] = result.plan.target_processing_options.clone();
    }
    if source.metadata["remote_upload"] != expected_origin {
        return Err(conflict());
    }
    Ok(result)
}
fn status(stage: &Stage) -> SourceEnrichmentStatus {
    SourceEnrichmentStatus {
        request_id: stage.plan.request_id.clone(),
        status: stage.status.clone(),
        generation: stage.plan.expected_generation,
        completed: stage.items.iter().filter(|v| v.entities.is_some()).count(),
        total: stage.items.len(),
    }
}
fn stage_value(stage: &Stage) -> Result<serde_json::Value> {
    let value = serde_json::to_value(stage).map_err(|_| invalid("checkpoint serialization"))?;
    if serde_json::to_vec(&value)
        .map_err(|_| invalid("checkpoint serialization"))?
        .len()
        > MAX_STAGE_BYTES
    {
        return Err(invalid("enrichment checkpoint too large"));
    }
    Ok(value)
}
/// Only a completed published graph for the exact currently visible generation is fresh.
pub fn source_entity_enrichment_current(source: &Source) -> bool {
    let Ok(decoded) = serde_json::from_value::<Stage>(source.metadata[KEY].clone()) else {
        return false;
    };
    if stage_shape(&decoded).is_err() {
        return false;
    }
    source.metadata[KEY]["version"] == 1
        && source.metadata[KEY]["status"] == "promoted"
        && source.metadata[KEY]["plan"]["expected_generation"] == source.generation
        && source.generation == source.successful_generation
        && source.status == SourceIngestionStatus::Ready
        && source.metadata[KEY]["plan"]["expected_content_sha256"]
            == source.content_hash.as_deref().unwrap_or("")
        && source.metadata["remote_upload"]["extract_entities"] == true
        && source.metadata[KEY]["plan"]["target_processing_options"]
            == source.metadata["remote_upload"]["processing_options"]
}
impl Repository {
    /// Single owning embedded datastore worker; durable checkpoints resume after crash.
    /// This separate gate is never held by source writers, so replacements can fence stale work.
    pub async fn source_entity_enrichment_worker_guard(&self) -> tokio::sync::OwnedMutexGuard<()> {
        self.source_enrichment_worker_lock
            .clone()
            .lock_owned()
            .await
    }
    async fn enrichment_source(&self, plan: &SourceEnrichmentPlan) -> Result<Source> {
        plan_valid(plan)?;
        let id = parse_record_id(&plan.source_id, Some("source"))?;
        let source: Source = self.db.select(id).await?.ok_or_else(conflict)?;
        let origin = &source.metadata["remote_upload"];
        if source.source_type != SourceType::Markdown
            || source.status != SourceIngestionStatus::Ready
            || source.generation != plan.expected_generation
            || source.successful_generation != plan.expected_generation
            || source.content_hash.as_deref() != Some(plan.expected_content_sha256.as_str())
            || origin["instance_id"] != plan.instance_id
            || origin["document_key"] != plan.document_key
            || source
                .metadata
                .get("remote_upload_pending")
                .is_some_and(|pending| pending != origin)
            || source.metadata["remote_upload_retired"] == true
        {
            return Err(conflict());
        }
        Ok(source)
    }
    /// Stage only the inspected source's current chunks. No entity/mention/policy changes.
    pub async fn begin_source_entity_enrichment(
        &self,
        plan: SourceEnrichmentPlan,
    ) -> Result<SourceEnrichmentStatus> {
        let _transition = self.remote_job_transition_lock.lock().await;
        let _lifecycle = self.proposal_acceptance_lock.lock().await;
        let source = self.enrichment_source(&plan).await?;
        if source.metadata.get(KEY).is_some() {
            return stage(&source, &plan).map(|s| status(&s));
        }
        let origin = &source.metadata["remote_upload"];
        let job = self
            .get_remote_upload_job(
                &plan.instance_id,
                origin["job_id"].as_str().ok_or_else(conflict)?,
            )
            .await?
            .ok_or_else(conflict)?;
        if origin["extract_entities"] != false
            || job.input.extract_entities
            || job.job.status != "completed"
            || uploaded_source_revision(&source, &job.input.markdown)?
                != plan.expected_source_revision
            || uploaded_processing_policy_sha256(&job.input.processing_options)?
                != plan.original_policy_sha256
            || origin["processing_options"] != job.input.processing_options
            || !uploaded_processing_compatible(
                &job.input.processing_options,
                &plan.target_processing_options,
                false,
                false,
            )
            || !uploaded_processing_compatible(
                &plan.target_processing_options,
                &plan.target_processing_options,
                true,
                false,
            )
        {
            return Err(conflict());
        }
        let notes = self
            .get_source_chunks(source.id.as_ref().ok_or_else(conflict)?)
            .await?;
        if notes.is_empty() || notes.len() > MAX_REMOTE_UPLOAD_CHUNKS {
            return Err(invalid("enrichment requires 1..200 existing chunks"));
        }
        let mut items = Vec::new();
        for note in &notes {
            let id = record_id_to_string(note.id.as_ref().ok_or_else(conflict)?);
            let mentions: Vec<RecordId> = self
                .db
                .query("SELECT VALUE id FROM mentions WHERE in = $note LIMIT 1")
                .bind(("note", note.id.clone()))
                .await?
                .take(0)?;
            if !mentions.is_empty() {
                return Err(invalid(
                    "existing graph overlay requires a separate reviewed reconciliation; no automatic overwrite",
                ));
            }
            items.push(Item {
                revision: content_revision(note)?,
                id,
                entities: None,
            });
        }
        let staged = Stage {
            version: 1,
            plan,
            status: "staged".into(),
            prior_origin: origin.clone(),
            items,
        };
        let mut metadata = source.metadata.clone();
        if let Some(fields) = metadata.as_object_mut() {
            fields.remove("remote_upload_pending");
        }
        metadata[KEY] = stage_value(&staged)?;
        self.write_enrichment(&source, metadata, "IF array::len((SELECT VALUE id FROM mentions WHERE in IN $note_ids LIMIT 1)) != 0 { THROW 'source-enrichment-revision-conflict'; };", notes).await?;
        Ok(status(&staged))
    }
    /// Exact ordered current chunks for the owner worker; no scan outside this source.
    pub async fn source_entity_enrichment_notes(
        &self,
        plan: &SourceEnrichmentPlan,
    ) -> Result<Vec<Note>> {
        let source = self.enrichment_source(plan).await?;
        let staged = stage(&source, plan)?;
        let mut notes = Vec::new();
        for item in &staged.items {
            notes.push(self.enrichment_note(item, plan).await?);
        }
        Ok(notes)
    }
    /// Sequential checkpoint. Identical replay is a no-op; failed provider work never calls this.
    pub async fn checkpoint_source_entity_enrichment(
        &self,
        plan: &SourceEnrichmentPlan,
        index: usize,
        entities: Vec<Entity>,
    ) -> Result<SourceEnrichmentStatus> {
        let _transition = self.remote_job_transition_lock.lock().await;
        let _lifecycle = self.proposal_acceptance_lock.lock().await;
        let source = self.enrichment_source(plan).await?;
        let mut staged = stage(&source, plan)?;
        if staged.status != "staged" || index >= staged.items.len() || entities.len() > 128 {
            return Err(conflict());
        }
        let note = self.enrichment_note(&staged.items[index], plan).await?;
        super::remote_jobs::validate_enrichment_entities(
            &note,
            self.note_extraction_scope(&note).await?.as_deref(),
            &entities,
        )?;
        if let Some(saved) = &staged.items[index].entities {
            if serde_json::to_value(saved).map_err(|_| conflict())?
                != serde_json::to_value(&entities).map_err(|_| conflict())?
            {
                return Err(conflict());
            }
            return Ok(status(&staged));
        }
        if staged
            .items
            .iter()
            .take(index)
            .any(|v| v.entities.is_none())
        {
            return Err(invalid("checkpoint must be sequential"));
        }
        staged.items[index].entities = Some(entities);
        let mut metadata = source.metadata.clone();
        metadata[KEY] = stage_value(&staged)?;
        self.write_enrichment(&source, metadata, "", vec![note])
            .await?;
        Ok(status(&staged))
    }
    async fn enrichment_note(&self, item: &Item, plan: &SourceEnrichmentPlan) -> Result<Note> {
        let note = self.get_note(&item.id).await?.ok_or_else(conflict)?;
        if note.source_id.as_ref().map(record_id_to_string).as_deref()
            != Some(plan.source_id.as_str())
            || note.source_generation != Some(plan.expected_generation)
            || content_revision(&note)? != item.revision
        {
            return Err(conflict());
        }
        Ok(note)
    }
    /// Atomic all-chunk publication. Source/note identity, generation, vectors and timestamps survive.
    pub async fn promote_source_entity_enrichment(
        &self,
        plan: &SourceEnrichmentPlan,
    ) -> Result<SourceEnrichmentStatus> {
        let _transition = self.remote_job_transition_lock.lock().await;
        let _lifecycle = self.proposal_acceptance_lock.lock().await;
        let source = self.enrichment_source(plan).await?;
        let mut staged = stage(&source, plan)?;
        if staged.status == "promoted" && source_entity_enrichment_current(&source) {
            return Ok(status(&staged));
        }
        if staged.status != "staged" || staged.items.iter().any(|v| v.entities.is_none()) {
            return Err(invalid("all extraction checkpoints must be complete"));
        }
        let mut notes = Vec::new();
        let mut effects = String::new();
        for (index, item) in staged.items.iter().enumerate() {
            let note = self.enrichment_note(item, plan).await?;
            let entities = item.entities.as_ref().ok_or_else(conflict)?;
            super::remote_jobs::validate_enrichment_entities(
                &note,
                self.note_extraction_scope(&note).await?.as_deref(),
                entities,
            )?;
            effects.push_str(&format!("LET $note = $notes[{index}].id; IF array::len((SELECT VALUE id FROM mentions WHERE in = $note LIMIT 1)) != 0 {{ THROW '{FENCE}'; }}; LET $replacement_entities = $entity_batches[{index}]; LET $replacement_entity_names = $identity_batches[{index}]; {} {}", super::notes::replacement_entities_transaction(), super::notes::replacement_mentions_transaction("$note")));
            notes.push(note);
        }
        effects.push_str(
            &super::remote_endpoint_proposals::invalidate_reviewed_endpoint_sql("$note_ids"),
        );
        staged.status = "promoted".into();
        let mut metadata = source.metadata.clone();
        metadata["remote_upload"]["extract_entities"] = serde_json::json!(true);
        metadata["graph_enrichment_managed"] = serde_json::json!(true);
        metadata["remote_upload"]["processing_options"] = plan.target_processing_options.clone();
        metadata[KEY] = stage_value(&staged)?;
        metadata["graph_policy_revision"] = serde_json::json!(source.metadata
            ["graph_policy_revision"]
            .as_u64()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or_else(conflict)?);
        self.write_enrichment(&source, metadata, &effects, notes)
            .await?;
        Ok(status(&staged))
    }
    /// Explicit policy rollback. Refuse if any relationships/proposals now depend on these notes.
    /// It removes only this conversion's mentions; orphan entities are intentionally retained.
    pub async fn rollback_source_entity_enrichment(
        &self,
        plan: &SourceEnrichmentPlan,
        confirmed: bool,
    ) -> Result<SourceEnrichmentStatus> {
        if !confirmed {
            return Err(invalid("rollback requires separate confirmation"));
        }
        let _transition = self.remote_job_transition_lock.lock().await;
        let _lifecycle = self.proposal_acceptance_lock.lock().await;
        let source = self.enrichment_source(plan).await?;
        let mut staged = stage(&source, plan)?;
        if staged.status == "rolled_back" {
            return Ok(status(&staged));
        }
        if !matches!(staged.status.as_str(), "staged" | "promoted") {
            return Err(conflict());
        }
        if staged.status == "promoted" && !source_entity_enrichment_current(&source) {
            return Err(conflict());
        }
        let mut notes = Vec::new();
        let mut effects = String::new();
        for item in &staged.items {
            notes.push(self.enrichment_note(item, plan).await?);
        }
        for table in [
            "supports",
            "contradicts",
            "derived_from",
            "related_to",
            "proposed_edge",
        ] {
            effects.push_str(&format!("IF array::len((SELECT VALUE id FROM {table} WHERE in IN $note_ids OR out IN $note_ids LIMIT 1)) != 0 {{ THROW '{FENCE}'; }}; "));
        }
        if staged.status == "promoted" {
            for (index, item) in staged.items.iter().enumerate() {
                let count = item.entities.as_ref().ok_or_else(conflict)?.len();
                effects.push_str(&format!("LET $owned_mentions = (SELECT * FROM mentions WHERE in = $notes[{index}].id); IF array::len($owned_mentions) != {count} {{ THROW '{FENCE}'; }}; FOR $mention IN $owned_mentions {{ LET $expected_entity = array::filter($entity_batches[{index}], |$entity| $entity.identity_key = $mention.out.identity_key)[0]; IF $expected_entity = NONE OR $mention.metadata != object::extend($expected_entity.metadata, {{ aliases: array::slice($expected_entity.metadata.aliases ?? [], 0, 8) }}) {{ THROW '{FENCE}'; }}; }}; "));
            }
            effects.push_str("DELETE mentions WHERE in IN $note_ids; ");
        }
        let mut metadata = source.metadata.clone();
        metadata["remote_upload"] = staged.prior_origin.clone();
        metadata["graph_enrichment_managed"] = serde_json::json!(false);
        staged.status = "rolled_back".into();
        metadata[KEY] = stage_value(&staged)?;
        metadata["graph_policy_revision"] = serde_json::json!(source.metadata
            ["graph_policy_revision"]
            .as_u64()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or_else(conflict)?);
        self.write_enrichment(&source, metadata, &effects, notes)
            .await?;
        Ok(status(&staged))
    }
    async fn write_enrichment(
        &self,
        source: &Source,
        metadata: serde_json::Value,
        effects: &str,
        notes: Vec<Note>,
    ) -> Result<()> {
        let note_ids = notes
            .iter()
            .map(|n| n.id.clone().ok_or_else(conflict))
            .collect::<Result<Vec<_>>>()?;
        let staged: Stage =
            serde_json::from_value(metadata[KEY].clone()).map_err(|_| conflict())?;
        stage_shape(&staged)?;
        let all_ids = staged
            .items
            .iter()
            .map(|i| parse_record_id(&i.id, Some("note")))
            .collect::<Result<Vec<_>>>()?;
        let entity_batches = staged
            .items
            .iter()
            .map(|item| item.entities.clone().unwrap_or_default())
            .collect::<Vec<_>>();
        let identity_batches = entity_batches
            .iter()
            .map(|v| {
                v.iter()
                    .map(Entity::effective_identity_key)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut guards = String::new();
        for (index, _) in notes.iter().enumerate() {
            guards.push_str(
                &super::notes::editor_snapshot_guard(false)
                    .replace("$editor_source", &format!("$notes[{index}].id"))
                    .replace("$editor_expected", &format!("$notes[{index}]"))
                    .replace("$editor_matches", &format!("$note_matches_{index}")),
            );
        }
        let mut response = self.db.query(format!("BEGIN TRANSACTION; LET $matches = (SELECT VALUE id FROM source WHERE id = $source AND metadata = $expected_metadata AND generation = $generation AND successful_generation = $generation AND status = 'ready' AND content_hash = $hash AND content = $content AND title = $title); IF array::len($matches) != 1 {{ THROW '{FENCE}'; }}; LET $current_notes = (SELECT VALUE id FROM note WHERE source_id = $source AND source_generation = $generation); IF array::len($current_notes) != array::len($all_ids) OR array::len(array::difference($current_notes,$all_ids)) != 0 {{ THROW '{FENCE}'; }}; {guards} {effects} UPDATE $source SET metadata = $metadata; COMMIT TRANSACTION;"))
            .bind(("source", source.id.clone())).bind(("expected_metadata", source.metadata.clone())).bind(("generation", source.generation))
            .bind(("hash", source.content_hash.clone())).bind(("content", source.content.clone())).bind(("title", source.title.clone()))
            .bind(("metadata", metadata)).bind(("all_ids", all_ids)).bind(("notes", notes)).bind(("note_ids", note_ids)).bind(("entity_batches", entity_batches)).bind(("identity_batches", identity_batches)).await?;
        let errors = response.take_errors();
        if !errors.is_empty() {
            #[cfg(test)]
            eprintln!("synthetic enrichment transaction errors: {errors:?}");
            return Err(conflict());
        }
        Ok(())
    }
}
