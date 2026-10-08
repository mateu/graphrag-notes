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
    /// Latest acknowledged unchanged capture; prior_origin remains immutable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    current_origin: Option<serde_json::Value>,
    items: Vec<Item>,
    #[serde(default)]
    opening_graph_epoch: u64,
    #[serde(default)]
    rollback_opening_revision: Option<String>,
    #[serde(default)]
    rollback_opening_graph_epoch: Option<u64>,
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
pub(super) fn plan_valid(plan: &SourceEnrichmentPlan) -> Result<()> {
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
            let mut entity_ids = HashSet::new();
            for entity in entities {
                let scope = entity.metadata["extraction"]["scope"]
                    .as_str()
                    .ok_or_else(conflict)?;
                let identity =
                    serde_json::json!([scope, entity.entity_type, entity.canonical_name])
                        .to_string();
                let key = format!(
                    "extracted-v1:{}",
                    graphrag_core::normalized_content_hash(&identity)
                );
                if entity.identity_key.as_deref() != Some(key.as_str())
                    || !entity_ids.insert(key)
                    || entity.name.trim().is_empty()
                    || entity.name.chars().count() > 80
                    || entity.name.chars().any(char::is_control)
                    || (!matches!(
                        entity.entity_type,
                        graphrag_core::EntityType::Person | graphrag_core::EntityType::Project
                    ) && scope != value.plan.source_id)
                {
                    return Err(conflict());
                }
            }
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
fn saved_stage(source: &Source) -> Result<Stage> {
    if serde_json::to_vec(&source.metadata[KEY])
        .map_err(|_| conflict())?
        .len()
        > MAX_STAGE_BYTES
    {
        return Err(conflict());
    }
    let result: Stage =
        serde_json::from_value(source.metadata[KEY].clone()).map_err(|_| conflict())?;
    stage_shape(&result)?;
    let plan = &result.plan;
    if source.id.as_ref().map(record_id_to_string).as_deref() != Some(plan.source_id.as_str())
        || result.prior_origin["extract_entities"] != false
        || result.prior_origin["instance_id"] != plan.instance_id
        || result.prior_origin["document_key"] != plan.document_key
        || uploaded_processing_policy_sha256(&result.prior_origin["processing_options"])?
            != plan.original_policy_sha256
        || !uploaded_processing_compatible(
            &result.prior_origin["processing_options"],
            &plan.target_processing_options,
            false,
            false,
        )
    {
        return Err(conflict());
    }
    let current_origin = result
        .current_origin
        .as_ref()
        .unwrap_or(&result.prior_origin);
    if current_origin["instance_id"] != plan.instance_id
        || current_origin["document_key"] != plan.document_key
    {
        return Err(conflict());
    }
    let mut expected_origin = current_origin.clone();
    if result.status == "promoted" {
        expected_origin["extract_entities"] = serde_json::json!(true);
        expected_origin["processing_options"] = result.plan.target_processing_options.clone();
    } else {
        expected_origin["extract_entities"] = serde_json::json!(false);
        expected_origin["processing_options"] = result.prior_origin["processing_options"].clone();
    }
    if source.metadata["remote_upload"] != expected_origin {
        return Err(conflict());
    }
    Ok(result)
}
fn stage(source: &Source, plan: &SourceEnrichmentPlan) -> Result<Stage> {
    let result = saved_stage(source)?;
    if result.plan != *plan {
        return Err(conflict());
    }
    Ok(result)
}
fn origin_current(source: &Source, decoded: &Stage) -> bool {
    source.source_type == SourceType::Markdown
        && decoded.plan.expected_generation == source.generation
        && source.generation == source.successful_generation
        && source.status == SourceIngestionStatus::Ready
        && Some(decoded.plan.expected_content_sha256.as_str()) == source.content_hash.as_deref()
        && source.metadata["remote_upload_retired"] != true
        && source.metadata.get("remote_upload_pending").is_none()
        && (decoded.status != "promoted" || source.metadata["graph_enrichment_managed"] == true)
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

/// Metadata-only eligibility, not a graph freshness proof. Public readback must also
/// call Repository::source_entity_enrichment_current to validate datastore evidence.
pub fn source_entity_enrichment_metadata_eligible(source: &Source) -> bool {
    let Ok(decoded) = saved_stage(source) else {
        return false;
    };
    decoded.status == "promoted"
        && source.source_type == SourceType::Markdown
        && decoded.plan.expected_generation == source.generation
        && source.generation == source.successful_generation
        && source.status == SourceIngestionStatus::Ready
        && Some(decoded.plan.expected_content_sha256.as_str()) == source.content_hash.as_deref()
        && source.metadata["graph_enrichment_managed"] == true
        && source.metadata["remote_upload_retired"] != true
        && source.metadata["remote_upload"]["instance_id"] == decoded.plan.instance_id
        && source.metadata["remote_upload"]["document_key"] == decoded.plan.document_key
        && source.metadata["remote_upload"]["extract_entities"] == true
        && decoded.plan.target_processing_options
            == source.metadata["remote_upload"]["processing_options"]
        && !source
            .metadata
            .get("remote_upload_pending")
            .is_some_and(|pending| pending != &source.metadata["remote_upload"])
}
impl Repository {
    /// Fail closed against exact current notes, scoped typed entities and published
    /// mentions. Snapshot guards and inventory checks share one read transaction.
    pub async fn source_entity_enrichment_current(&self, source: &Source) -> Result<bool> {
        if !source_entity_enrichment_metadata_eligible(source) {
            return Ok(false);
        }
        let decoded: Stage =
            serde_json::from_value(source.metadata[KEY].clone()).map_err(|_| conflict())?;
        Ok(self
            .validate_enrichment_evidence(source, &decoded, true)
            .await
            .is_ok())
    }
    /// Rebind only the active capture snapshot, never the reviewed opening lineage.
    pub(super) async fn rebind_unchanged_enrichment_origin(
        &self,
        source: &Source,
        origin: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        let mut saved = self.validated_enrichment_origin(source).await?;
        if origin["instance_id"] != saved.plan.instance_id
            || origin["document_key"] != saved.plan.document_key
            || origin["extract_entities"] != source.metadata["remote_upload"]["extract_entities"]
            || origin["processing_options"]
                != source.metadata["remote_upload"]["processing_options"]
        {
            return Err(conflict());
        }
        saved.current_origin = Some(origin.clone());
        stage_value(&saved)
    }
    pub(super) async fn validate_source_entity_enrichment_origin(
        &self,
        source: &Source,
    ) -> Result<()> {
        self.validated_enrichment_origin(source).await.map(|_| ())
    }
    async fn validated_enrichment_origin(&self, source: &Source) -> Result<Stage> {
        let saved = saved_stage(source)?;
        if !origin_current(source, &saved) {
            return Err(conflict());
        }
        self.validate_enrichment_evidence(source, &saved, saved.status == "promoted")
            .await?;
        Ok(saved)
    }
    async fn validate_enrichment_evidence(
        &self,
        source: &Source,
        staged: &Stage,
        published: bool,
    ) -> Result<()> {
        if let Some(origin) = &staged.current_origin {
            let latest = self
                .get_remote_upload_job(
                    &staged.plan.instance_id,
                    origin["job_id"].as_str().ok_or_else(conflict)?,
                )
                .await?
                .ok_or_else(conflict)?;
            if latest.input.enrichment.is_some()
                || latest.source_id != source.id
                || latest.source_generation != Some(staged.plan.expected_generation)
                || source.uri.as_deref() != Some(latest.source_uri.as_str())
                || latest.job.scope.as_deref() != Some("unchanged")
                || !matches!(latest.phase.as_str(), "promoted" | "completed")
                || latest.request_id != origin["request_id"]
                || latest.input.document_key != staged.plan.document_key
                || latest.input.source_provenance != origin["source"]
                || latest.input.title != source.title
                || graphrag_core::normalized_content_hash(&latest.input.markdown)
                    != staged.plan.expected_content_sha256
                || !uploaded_processing_compatible(
                    &latest.input.processing_options,
                    &staged.plan.target_processing_options,
                    false,
                    false,
                )
            {
                return Err(conflict());
            }
        }
        let mut notes = Vec::new();
        let mut checks = String::new();
        for (index, item) in staged.items.iter().enumerate() {
            let note = self.enrichment_note(item, &staged.plan).await?;
            if let Some(entities) = &item.entities {
                super::remote_jobs::validate_enrichment_entities(
                    &note,
                    self.note_extraction_scope(&note).await?.as_deref(),
                    entities,
                )?;
            }
            checks.push_str(
                &super::notes::editor_snapshot_guard(false)
                    .replace("$editor_source", &format!("$notes[{index}].id"))
                    .replace("$editor_expected", &format!("$notes[{index}]"))
                    .replace("$editor_matches", &format!("$note_matches_{index}")),
            );
            if published {
                let count = item.entities.as_ref().ok_or_else(conflict)?.len();
                checks.push_str(&format!("LET $owned_mentions_{index} = (SELECT * FROM mentions WHERE in = $notes[{index}].id); IF array::len($owned_mentions_{index}) != {count} {{ THROW '{FENCE}'; }}; FOR $expected_entity IN $entity_batches[{index}] {{ LET $found = array::filter($owned_mentions_{index}, |$mention| $mention.out.identity_key = $expected_entity.identity_key); IF array::len($found) != 1 {{ THROW '{FENCE}'; }}; LET $mention = $found[0]; IF $mention.out.canonical_name != $expected_entity.canonical_name OR $mention.out.entity_type != $expected_entity.entity_type OR $mention.metadata != object::extend($expected_entity.metadata, {{ aliases: array::slice($expected_entity.metadata.aliases ?? [], 0, 8) }}) {{ THROW '{FENCE}'; }}; }}; "));
            }
            notes.push(note);
        }
        let all_ids = staged
            .items
            .iter()
            .map(|i| parse_record_id(&i.id, Some("note")))
            .collect::<Result<Vec<_>>>()?;
        let batches = staged
            .items
            .iter()
            .map(|i| i.entities.clone().unwrap_or_default())
            .collect::<Vec<_>>();
        self.db.query(format!("BEGIN TRANSACTION; LET $matches=(SELECT VALUE id FROM source WHERE id=$source AND metadata=$metadata AND generation=$generation AND successful_generation=$generation AND status='ready' AND content_hash=$hash AND content=$content AND title=$title); IF array::len($matches)!=1 {{ THROW '{FENCE}'; }}; LET $current_notes=(SELECT VALUE id FROM note WHERE source_id=$source AND source_generation=$generation); IF array::len($current_notes)!=array::len($all_ids) OR array::len(array::difference($current_notes,$all_ids))!=0 {{ THROW '{FENCE}'; }}; {checks} COMMIT TRANSACTION;"))
            .bind(("source", source.id.clone())).bind(("metadata", source.metadata.clone())).bind(("generation",source.generation))
            .bind(("hash",source.content_hash.clone())).bind(("content",source.content.clone())).bind(("title",source.title.clone()))
            .bind(("all_ids",all_ids)).bind(("notes",notes)).bind(("entity_batches",batches)).await?.check().map_err(|_| conflict())?;
        Ok(())
    }
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
    /// Current, repository-validated single-source adoption proof; not submitted receipt echo.
    pub async fn reviewed_source_enrichment_proof(
        &self,
        source: &Source,
        supplied_content: &str,
    ) -> Result<Option<serde_json::Value>> {
        let Some(raw) = source.metadata.get(KEY) else {
            return Ok(None);
        };
        let saved: Stage = match serde_json::from_value(raw.clone()) {
            Ok(x) => x,
            Err(_) => return Ok(None),
        };
        if !matches!(saved.status.as_str(), "promoted" | "rolled_back")
            || stage(source, &saved.plan).is_err()
        {
            return Ok(None);
        }
        if saved.status == "promoted" {
            if !self.source_entity_enrichment_current(source).await? {
                return Ok(None);
            }
        } else {
            self.validate_enrichment_evidence(source, &saved, false)
                .await?;
        }
        let notes = self.source_entity_enrichment_notes(&saved.plan).await?;
        let mut endpoints = serde_json::Map::new();
        let epoch = source.metadata["graph_policy_revision"]
            .as_u64()
            .ok_or_else(conflict)?;
        for note in &notes {
            let id = record_id_to_string(note.id.as_ref().ok_or_else(conflict)?);
            let inspected = self.inspect_record(&id, 0).await?;
            if !inspected.conversations.is_empty() || !inspected.messages.is_empty() {
                return Ok(None);
            }
            if inspected.provenance.graph_policy_revision != Some(epoch)
                || inspected.content != note.content
            {
                return Ok(None);
            }
            endpoints.insert(id, serde_json::json!(inspected.revision));
        }
        // Fence the exact full snapshots used above, including complete chunk inventory.
        let mut checks = String::new();
        for index in 0..notes.len() {
            checks.push_str(
                &super::notes::editor_snapshot_guard(false)
                    .replace("$editor_source", &format!("$notes[{index}].id"))
                    .replace("$editor_expected", &format!("$notes[{index}]"))
                    .replace("$editor_matches", &format!("$snapshot_{index}")),
            );
        }
        let ids = notes
            .iter()
            .map(|n| n.id.clone().ok_or_else(conflict))
            .collect::<Result<Vec<_>>>()?;
        self.db.query(format!("BEGIN TRANSACTION; IF array::len((SELECT VALUE id FROM note_from_conversation WHERE in IN $ids LIMIT 1))!=0 OR array::len((SELECT VALUE id FROM note_from_message WHERE in IN $ids LIMIT 1))!=0 {{ THROW '{FENCE}'; }}; LET $snapshot=(SELECT VALUE id FROM source WHERE id=$source AND metadata=$metadata AND content=$content AND content_hash=$hash AND title=$title AND generation=$generation AND successful_generation=$generation AND status='ready'); IF array::len($snapshot)!=1 {{ THROW '{FENCE}'; }}; LET $inventory=(SELECT VALUE id FROM note WHERE source_id=$source AND source_generation=$generation); IF array::len($inventory)!=array::len($ids) OR array::len(array::difference($inventory,$ids))!=0 {{ THROW '{FENCE}'; }}; {checks} COMMIT TRANSACTION;"))
            .bind(("source",source.id.clone())).bind(("metadata",source.metadata.clone())).bind(("content",source.content.clone())).bind(("hash",source.content_hash.clone())).bind(("title",source.title.clone())).bind(("generation",source.generation)).bind(("notes",notes)).bind(("ids",ids)).await?.check()?;
        let origin = &source.metadata["remote_upload"];
        let opening_epoch = if saved.status == "promoted" {
            saved.opening_graph_epoch
        } else {
            saved.rollback_opening_graph_epoch.ok_or_else(conflict)?
        };
        if opening_epoch.checked_add(1) != Some(epoch) {
            return Ok(None);
        }
        let revision = uploaded_source_revision(source, supplied_content)?;
        fn ordinal(origin: &serde_json::Value, key: &str) -> Option<u64> {
            let value = &origin["source"]["metadata"][key];
            if value.is_null() {
                Some(1)
            } else {
                value.as_str()?.parse().ok()
            }
        }
        let Some(part) = ordinal(origin, "part") else {
            return Ok(None);
        };
        let Some(parts) = ordinal(origin, "parts") else {
            return Ok(None);
        };
        if part == 0 || parts == 0 || part > parts {
            return Ok(None);
        }
        Ok(Some(
            serde_json::json!({"operation":if saved.status=="promoted" {"enrich"} else {"rollback"},"source_id":saved.plan.source_id,"document_key":saved.plan.document_key,"owner":saved.plan.instance_id,"source_uri":source.uri,"title":source.title,"provenance":origin["source"],"content_sha256":format!("{:x}",Sha256::digest(supplied_content.as_bytes())),"generation":source.generation,"original_request_id":saved.prior_origin["request_id"],"part":part,"parts":parts,"opening_revision":if saved.status=="promoted" {saved.plan.expected_source_revision} else {saved.rollback_opening_revision.ok_or_else(conflict)?},"final_revision":revision,"opening_graph_epoch":opening_epoch,"final_graph_epoch":epoch,"original_policy_sha256":saved.plan.original_policy_sha256,"applied_policy_sha256":uploaded_processing_policy_sha256(&origin["processing_options"])?,"extraction_enabled":saved.status=="promoted","endpoint_revisions":endpoints}),
        ))
    }
    /// Stage only the inspected source's current chunks. No entity/mention/policy changes.
    pub async fn begin_source_entity_enrichment(
        &self,
        plan: SourceEnrichmentPlan,
    ) -> Result<SourceEnrichmentStatus> {
        self.begin_source_entity_enrichment_leased(plan, None).await
    }
    pub async fn begin_source_entity_enrichment_leased(
        &self,
        plan: SourceEnrichmentPlan,
        lease: Option<&RemoteJobLease>,
    ) -> Result<SourceEnrichmentStatus> {
        self.begin_enrichment_internal(plan, lease, false).await
    }
    pub async fn preflight_source_entity_enrichment(
        &self,
        plan: SourceEnrichmentPlan,
    ) -> Result<SourceEnrichmentStatus> {
        self.begin_enrichment_internal(plan, None, true).await
    }
    pub async fn inspect_source_entity_enrichment(
        &self,
        plan: &SourceEnrichmentPlan,
    ) -> Result<SourceEnrichmentStatus> {
        let source = self.enrichment_source(plan).await?;
        let saved = stage(&source, plan)?;
        self.validate_enrichment_evidence(&source, &saved, false)
            .await?;
        Ok(status(&saved))
    }
    async fn begin_enrichment_internal(
        &self,
        plan: SourceEnrichmentPlan,
        lease: Option<&RemoteJobLease>,
        dry_run: bool,
    ) -> Result<SourceEnrichmentStatus> {
        let _transition = self.remote_job_transition_lock.lock().await;
        let _lifecycle = self.proposal_acceptance_lock.lock().await;
        let source = self.enrichment_source(&plan).await?;
        if source.metadata.get(KEY).is_some() {
            let saved = stage(&source, &plan)?;
            if saved.status == "promoted" {
                if !self.source_entity_enrichment_current(&source).await? {
                    return Err(conflict());
                }
            } else {
                self.validate_enrichment_evidence(&source, &saved, false)
                    .await?;
            }
            if lease.is_some() && saved.status == "promoted" && !dry_run {
                let notes = self.source_entity_enrichment_notes(&plan).await?;
                self.write_enrichment(&source, source.metadata.clone(), "", notes, lease, None)
                    .await?;
            }
            return Ok(status(&saved));
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
            current_origin: None,
            items,
            opening_graph_epoch: source.metadata["graph_policy_revision"]
                .as_u64()
                .unwrap_or(0),
            rollback_opening_revision: None,
            rollback_opening_graph_epoch: None,
        };
        let mut metadata = source.metadata.clone();
        if let Some(fields) = metadata.as_object_mut() {
            fields.remove("remote_upload_pending");
        }
        metadata[KEY] = stage_value(&staged)?;
        if dry_run {
            return Ok(status(&staged));
        }
        self.write_enrichment(&source, metadata, "IF array::len((SELECT VALUE id FROM mentions WHERE in IN $note_ids LIMIT 1)) != 0 { THROW 'source-enrichment-revision-conflict'; };", notes, lease, None).await?;
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
        self.checkpoint_source_entity_enrichment_leased(plan, index, entities, None)
            .await
    }
    pub async fn checkpoint_source_entity_enrichment_leased(
        &self,
        plan: &SourceEnrichmentPlan,
        index: usize,
        entities: Vec<Entity>,
        lease: Option<&RemoteJobLease>,
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
        self.write_enrichment(&source, metadata, "", vec![note], lease, None)
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
        self.promote_source_entity_enrichment_leased(plan, None)
            .await
    }
    pub async fn promote_source_entity_enrichment_leased(
        &self,
        plan: &SourceEnrichmentPlan,
        lease: Option<&RemoteJobLease>,
    ) -> Result<SourceEnrichmentStatus> {
        let _transition = self.remote_job_transition_lock.lock().await;
        let _lifecycle = self.proposal_acceptance_lock.lock().await;
        let source = self.enrichment_source(plan).await?;
        let mut staged = stage(&source, plan)?;
        if staged.status == "promoted" {
            if !self.source_entity_enrichment_current(&source).await? {
                return Err(conflict());
            }
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
            &super::remote_endpoint_proposals::invalidate_source_reviewed_endpoints_sql("$source"),
        );
        staged.status = "promoted".into();
        let mut metadata = source.metadata.clone();
        metadata["remote_upload"]["extract_entities"] = serde_json::json!(true);
        metadata["graph_enrichment_managed"] = serde_json::json!(true);
        metadata["remote_upload"]["processing_options"] = plan.target_processing_options.clone();
        if staged.current_origin.is_some() {
            staged.current_origin = Some(metadata["remote_upload"].clone());
        }
        metadata[KEY] = stage_value(&staged)?;
        metadata["graph_policy_revision"] = serde_json::json!(source.metadata
            ["graph_policy_revision"]
            .as_u64()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or_else(conflict)?);
        self.write_enrichment(&source, metadata, &effects, notes, lease, None)
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
        self.rollback_source_entity_enrichment_leased(plan, confirmed, None, None)
            .await
    }
    pub async fn rollback_source_entity_enrichment_leased(
        &self,
        plan: &SourceEnrichmentPlan,
        confirmed: bool,
        lease: Option<&RemoteJobLease>,
        reviewed_revision: Option<&str>,
    ) -> Result<SourceEnrichmentStatus> {
        if !confirmed {
            return Err(invalid("rollback requires separate confirmation"));
        }
        if lease.is_some() && reviewed_revision.is_none() {
            return Err(invalid(
                "leased rollback requires its reviewed source revision",
            ));
        }
        let _transition = self.remote_job_transition_lock.lock().await;
        let _lifecycle = self.proposal_acceptance_lock.lock().await;
        let source = self.enrichment_source(plan).await?;
        let mut staged = stage(&source, plan)?;
        let rollback_opening_revision =
            if staged.status != "rolled_back" || reviewed_revision.is_some() {
                let original_job = self
                    .get_remote_upload_job(
                        &plan.instance_id,
                        source.metadata["remote_upload"]["job_id"]
                            .as_str()
                            .ok_or_else(conflict)?,
                    )
                    .await?
                    .ok_or_else(conflict)?;
                Some(uploaded_source_revision(
                    &source,
                    &original_job.input.markdown,
                )?)
            } else {
                None
            };
        if reviewed_revision
            .is_some_and(|expected| rollback_opening_revision.as_deref() != Some(expected))
        {
            return Err(conflict());
        }
        if staged.status == "rolled_back" {
            self.validate_enrichment_evidence(&source, &staged, false)
                .await?;
            if lease.is_some() {
                let notes = self.source_entity_enrichment_notes(plan).await?;
                self.write_enrichment(
                    &source,
                    source.metadata.clone(),
                    "",
                    notes,
                    lease,
                    reviewed_revision,
                )
                .await?;
            }
            return Ok(status(&staged));
        }
        if !matches!(staged.status.as_str(), "staged" | "promoted") {
            return Err(conflict());
        }
        if staged.status == "promoted" && !self.source_entity_enrichment_current(&source).await? {
            return Err(conflict());
        }
        let mut notes = Vec::new();
        let mut effects = String::new();
        for item in &staged.items {
            notes.push(self.enrichment_note(item, plan).await?);
        }
        // Relationship rows are always live dependencies. Proposal records are
        // audit history once explicitly rejected or superseded; every other
        // (including future/unknown) state remains fail-closed.
        for table in ["supports", "contradicts", "derived_from", "related_to"] {
            effects.push_str(&format!("IF array::len((SELECT VALUE id FROM {table} WHERE in IN $note_ids OR out IN $note_ids LIMIT 1)) != 0 {{ THROW '{FENCE}'; }}; "));
        }
        effects.push_str(&format!("IF array::len((SELECT VALUE id FROM proposed_edge WHERE status != 'rejected' AND status != 'superseded' AND (in IN $note_ids OR out IN $note_ids) LIMIT 1)) != 0 {{ THROW '{FENCE}'; }}; "));
        // Retire reviewed decisions pinned to detached source-linked notes too.
        // The dependency guards above remain deliberately limited to owned notes.
        effects.push_str(
            &super::remote_endpoint_proposals::invalidate_source_reviewed_endpoints_sql("$source"),
        );
        if staged.status == "promoted" {
            for (index, item) in staged.items.iter().enumerate() {
                let count = item.entities.as_ref().ok_or_else(conflict)?.len();
                effects.push_str(&format!("LET $owned_mentions = (SELECT * FROM mentions WHERE in = $notes[{index}].id); IF array::len($owned_mentions) != {count} {{ THROW '{FENCE}'; }}; FOR $mention IN $owned_mentions {{ LET $expected_entity = array::filter($entity_batches[{index}], |$entity| $entity.identity_key = $mention.out.identity_key)[0]; IF $expected_entity = NONE OR $mention.metadata != object::extend($expected_entity.metadata, {{ aliases: array::slice($expected_entity.metadata.aliases ?? [], 0, 8) }}) {{ THROW '{FENCE}'; }}; }}; "));
            }
            effects.push_str("DELETE mentions WHERE in IN $note_ids; ");
        }
        let mut metadata = source.metadata.clone();
        metadata["remote_upload"] = staged
            .current_origin
            .clone()
            .unwrap_or_else(|| staged.prior_origin.clone());
        metadata["remote_upload"]["extract_entities"] = serde_json::json!(false);
        metadata["remote_upload"]["processing_options"] =
            staged.prior_origin["processing_options"].clone();
        if staged.current_origin.is_some() {
            staged.current_origin = Some(metadata["remote_upload"].clone());
        }
        metadata["graph_enrichment_managed"] = serde_json::json!(false);
        staged.rollback_opening_revision = rollback_opening_revision;
        staged.rollback_opening_graph_epoch = Some(
            source.metadata["graph_policy_revision"]
                .as_u64()
                .unwrap_or(0),
        );
        staged.status = "rolled_back".into();
        metadata[KEY] = stage_value(&staged)?;
        metadata["graph_policy_revision"] = serde_json::json!(source.metadata
            ["graph_policy_revision"]
            .as_u64()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or_else(conflict)?);
        self.write_enrichment(&source, metadata, &effects, notes, lease, reviewed_revision)
            .await?;
        Ok(status(&staged))
    }
    async fn write_enrichment(
        &self,
        source: &Source,
        metadata: serde_json::Value,
        effects: &str,
        notes: Vec<Note>,
        lease: Option<&RemoteJobLease>,
        reviewed_revision: Option<&str>,
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
        let lease_guard = if lease.is_some() {
            "LET $owner = (UPDATE $job SET updated_at=time::now() WHERE job_type='remote_upload' AND remote_enrichment_job=true AND remote_instance_id=$instance AND status='running' AND remote_service_epoch=$epoch AND remote_worker_token=$worker AND remote_cancel_requested=false AND remote_input.enrichment.plan=$plan AND remote_input.enrichment.rollback=$rollback AND remote_input.expected_source_revision=$expected_revision RETURN VALUE id); IF array::len($owner)!=1 { THROW 'remote-upload-worker-fence'; }; "
        } else {
            ""
        };
        let terminal = if lease.is_some() && staged.status != "staged" {
            "UPDATE $job SET status='completed', remote_phase=$phase, total_count=array::len($all_ids), completed_count=array::len($all_ids), remote_result=$outcome, last_error=NONE, finished_at=time::now(), remote_service_epoch=NONE, remote_worker_token=NONE; "
        } else if lease.is_some() {
            "UPDATE $job SET remote_phase='enrichment_staged', total_count=array::len($all_ids), completed_count=$completed; "
        } else {
            ""
        };
        let outcome = serde_json::json!({"enrichment":status(&staged), "plan":staged.plan});
        let mut response = self.db.query(format!("BEGIN TRANSACTION; {lease_guard} LET $matches = (SELECT VALUE id FROM source WHERE id = $source AND metadata = $expected_metadata AND generation = $generation AND successful_generation = $generation AND status = 'ready' AND content_hash = $hash AND content = $content AND title = $title); IF array::len($matches) != 1 {{ THROW '{FENCE}'; }}; LET $current_notes = (SELECT VALUE id FROM note WHERE source_id = $source AND source_generation = $generation); IF array::len($current_notes) != array::len($all_ids) OR array::len(array::difference($current_notes,$all_ids)) != 0 {{ THROW '{FENCE}'; }}; {guards} {effects} UPDATE $source SET metadata = $metadata; {terminal} COMMIT TRANSACTION;"))
            .bind(("job", lease.map(|l|l.job_id.clone()))).bind(("instance",lease.map(|l|l.instance_id.clone()))).bind(("epoch",lease.map(|l|l.service_epoch.clone()))).bind(("worker",lease.map(|l|l.worker_token.clone()))).bind(("plan",serde_json::to_value(&staged.plan).map_err(|_|conflict())?)).bind(("phase",format!("enrichment_{}",staged.status))).bind(("completed",staged.items.iter().filter(|i|i.entities.is_some()).count() as i64)).bind(("outcome",outcome))
            .bind(("rollback", staged.status == "rolled_back"))
            .bind(("expected_revision", reviewed_revision.unwrap_or(&staged.plan.expected_source_revision)))
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
