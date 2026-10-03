//! Portable logical backup creation, verification, and fresh-target restore.
//!
//! Archives are a manifest plus a streaming JSONL payload. They intentionally
//! contain application records, never a copied live RocksDB directory.

use anyhow::{bail, Context, Result};
use graphrag_core::{PortableBackupManifest, PortableEmbeddingIdentity, PortableRecord};
use graphrag_db::{init_persistent, migrations, Repository, PORTABLE_TABLES};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

const MANIFEST_FILE: &str = "manifest.json";
const RECORDS_FILE: &str = "records.jsonl";
const EXPORT_PAGE_SIZE: usize = 256;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct BackupSummary {
    pub path: PathBuf,
    pub schema_version: u32,
    pub records: u64,
    pub record_counts: BTreeMap<String, u64>,
    pub includes_embeddings: bool,
    pub dry_run: bool,
}

/// Create a new portable archive. `path` must not already exist: replacing an
/// existing backup in place could turn an interrupted export into a plausible
/// but incomplete recovery artifact.
pub async fn create_backup(
    repo: &Repository,
    path: &Path,
    include_embeddings: bool,
) -> Result<BackupSummary> {
    if path.exists() {
        bail!(
            "refusing to overwrite existing backup {}; choose a new path",
            path.display()
        );
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        bail!("backup parent {} does not exist", parent.display());
    }
    let staging = sibling_staging_path(path, "backup")?;
    fs::create_dir(&staging)
        .with_context(|| format!("create backup staging directory {}", staging.display()))?;

    let created = async {
        let mut manifest =
            PortableBackupManifest::new(migrations::latest_version(), include_embeddings);
        if include_embeddings {
            let metadata = repo
                .portable_embedding_metadata()
                .await?
                .context("--include-embeddings requires initialized embedding model metadata")?;
            manifest.embedding_identity = Some(PortableEmbeddingIdentity {
                provider: metadata.embedding.provider,
                model: metadata.embedding.model,
                dimension: metadata.embedding.dimension,
            });
        }

        let records_path = staging.join(RECORDS_FILE);
        let payload = write_records(repo, &records_path, include_embeddings).await?;
        manifest.payload = payload;
        manifest.record_counts = count_records(&records_path)?;
        manifest.validate_format().map_err(anyhow::Error::msg)?;
        write_manifest(&staging.join(MANIFEST_FILE), &manifest)?;
        Ok::<_, anyhow::Error>(manifest)
    }
    .await;

    match created {
        Ok(manifest) => {
            // Verify the finalized artifact before exposing it at the requested
            // path. A failed verification leaves no completed backup behind.
            verify_backup(&staging)?;
            fs::rename(&staging, path).with_context(|| {
                format!(
                    "publish verified backup from {} to {}",
                    staging.display(),
                    path.display()
                )
            })?;
            Ok(summary_from_manifest(path, &manifest, false))
        }
        Err(error) => {
            let _ = fs::remove_dir_all(&staging);
            Err(error)
        }
    }
}

/// Validate an archive without opening or changing a database.
pub fn verify_backup(path: &Path) -> Result<BackupSummary> {
    let manifest = read_manifest(path)?;
    validate_manifest_schema(&manifest)?;
    let payload_path = path.join(&manifest.payload.path);
    verify_payload(&payload_path, &manifest)?;
    Ok(summary_from_manifest(path, &manifest, false))
}

/// Export the portable record stream as a standalone JSONL file with a
/// sidecar `<path>.manifest.json`. This is the same validated logical format
/// as `backup create`, merely packaged for tooling that expects a JSONL file.
pub async fn export_jsonl(repo: &Repository, path: &Path) -> Result<BackupSummary> {
    if path.exists() || jsonl_manifest_path(path).exists() {
        bail!(
            "refusing to overwrite existing export {} or its manifest sidecar",
            path.display()
        );
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        bail!("export parent {} does not exist", parent.display());
    }
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("export path must have a UTF-8 file name")?;
    let staging = sibling_staging_path(path, "export")?;
    fs::create_dir(&staging)?;
    let result = async {
        let staged_payload = staging.join(filename);
        let mut manifest = PortableBackupManifest::new(migrations::latest_version(), false);
        manifest.payload = write_records(repo, &staged_payload, false).await?;
        manifest.payload.path = filename.to_string();
        manifest.record_counts = count_records(&staged_payload)?;
        manifest.validate_format().map_err(anyhow::Error::msg)?;
        verify_payload(&staged_payload, &manifest)?;
        write_manifest(&staging.join(MANIFEST_FILE), &manifest)?;
        Ok::<_, anyhow::Error>(manifest)
    }
    .await;
    match result {
        Ok(manifest) => {
            fs::rename(staging.join(filename), path)?;
            fs::rename(staging.join(MANIFEST_FILE), jsonl_manifest_path(path))?;
            fs::remove_dir(&staging)?;
            Ok(summary_from_manifest(path, &manifest, false))
        }
        Err(error) => {
            let _ = fs::remove_dir_all(&staging);
            Err(error)
        }
    }
}

/// Verify a standalone JSONL export and its sidecar manifest without opening
/// a database.
#[cfg(test)]
pub fn verify_jsonl(path: &Path) -> Result<BackupSummary> {
    let manifest = read_manifest_file(&jsonl_manifest_path(path))?;
    validate_manifest_schema(&manifest)?;
    let name = path.file_name().and_then(|name| name.to_str());
    if name != Some(manifest.payload.path.as_str()) {
        bail!("JSONL manifest payload name does not match the export path");
    }
    verify_payload(path, &manifest)?;
    Ok(summary_from_manifest(path, &manifest, false))
}

fn verify_payload(payload_path: &Path, manifest: &PortableBackupManifest) -> Result<()> {
    if !payload_path.is_file() {
        bail!(
            "portable backup payload {} is missing",
            payload_path.display()
        );
    }

    let validation = inspect_records(payload_path, manifest)?;
    if validation.bytes != manifest.payload.bytes {
        bail!(
            "portable backup byte count mismatch: manifest {}, payload {}",
            manifest.payload.bytes,
            validation.bytes
        );
    }
    if validation.sha256 != manifest.payload.sha256 {
        bail!("portable backup checksum mismatch");
    }
    if validation.records != manifest.payload.records {
        bail!(
            "portable backup record count mismatch: manifest {}, payload {}",
            manifest.payload.records,
            validation.records
        );
    }
    if validation.record_counts != manifest.record_counts {
        bail!("portable backup table counts do not match the manifest");
    }
    validate_references(&validation.ids, &validation.references)?;
    Ok(())
}

/// Restore a verified backup into a newly created database directory. Restore
/// always stages a sibling database and renames it into place only after all
/// records have loaded and their counts validate; the requested target remains
/// untouched on any failure.
pub async fn restore_backup(
    backup_path: &Path,
    target_db_path: &Path,
    dry_run: bool,
) -> Result<BackupSummary> {
    let manifest = read_manifest(backup_path)?;
    validate_manifest_schema(&manifest)?;
    let payload_path = backup_path.join(&manifest.payload.path);
    verify_payload(&payload_path, &manifest)?;
    restore_verified_payload(&manifest, &payload_path, target_db_path, dry_run).await
}

/// Restore a verified standalone JSONL export into a fresh staged database.
pub async fn import_jsonl(
    path: &Path,
    target_db_path: &Path,
    dry_run: bool,
) -> Result<BackupSummary> {
    let manifest = read_manifest_file(&jsonl_manifest_path(path))?;
    validate_manifest_schema(&manifest)?;
    if path.file_name().and_then(|name| name.to_str()) != Some(manifest.payload.path.as_str()) {
        bail!("JSONL manifest payload name does not match the import path");
    }
    verify_payload(path, &manifest)?;
    restore_verified_payload(&manifest, path, target_db_path, dry_run).await
}

async fn restore_verified_payload(
    manifest: &PortableBackupManifest,
    payload_path: &Path,
    target_db_path: &Path,
    dry_run: bool,
) -> Result<BackupSummary> {
    let verified = summary_from_manifest(payload_path, manifest, false);
    ensure_absent_target(target_db_path)?;
    if dry_run {
        return Ok(BackupSummary {
            path: target_db_path.to_path_buf(),
            dry_run: true,
            ..verified
        });
    }

    let staging = sibling_staging_path(target_db_path, "restore")?;
    let restore_result = async {
        let db = init_persistent(&staging)
            .await
            .with_context(|| format!("open fresh staged restore database {}", staging.display()))?;
        let repo = Repository::new(db);
        restore_records(&repo, payload_path).await?;
        let restored = count_repository_records(&repo).await?;
        if restored != verified.record_counts {
            bail!("restored table counts do not match the verified backup manifest");
        }
        drop(repo);
        // Embedded RocksDB releases its directory lock asynchronously.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        Ok::<_, anyhow::Error>(())
    }
    .await;

    match restore_result {
        Ok(()) => {
            fs::rename(&staging, target_db_path).with_context(|| {
                format!(
                    "activate staged restore from {} to {}",
                    staging.display(),
                    target_db_path.display()
                )
            })?;
            Ok(BackupSummary {
                path: target_db_path.to_path_buf(),
                dry_run: false,
                ..verified
            })
        }
        Err(error) => {
            // The requested target was never created. Leave only the staging
            // path when cleanup itself fails, making the recovery location
            // explicit instead of silently deleting diagnostic evidence.
            let _ = fs::remove_dir_all(&staging);
            Err(error)
        }
    }
}

async fn write_records(
    repo: &Repository,
    path: &Path,
    include_embeddings: bool,
) -> Result<graphrag_core::PortablePayload> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("create portable payload {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    let mut hasher = Sha256::new();
    let mut bytes = 0_u64;
    let mut records = 0_u64;

    for table in PORTABLE_TABLES {
        // Compatibility metadata describes vector generations. Restoring it
        // without the vectors would falsely advertise a usable corpus, so it
        // travels only with an explicitly identified vector export.
        if *table == "graphrag_metadata" && !include_embeddings {
            continue;
        }
        let mut offset = 0_usize;
        loop {
            let page = repo
                .portable_records_page(table, offset, EXPORT_PAGE_SIZE)
                .await?;
            if page.is_empty() {
                break;
            }
            offset = offset.saturating_add(page.len());
            for mut record in page {
                sanitize_portable_record(table, &mut record, include_embeddings);
                if *table == "source" {
                    strip_source_title_path(&mut record);
                }
                let line = serde_json::to_vec(&PortableRecord {
                    table: (*table).to_string(),
                    record,
                })?;
                writer.write_all(&line)?;
                writer.write_all(b"\n")?;
                hasher.update(&line);
                hasher.update(b"\n");
                bytes = bytes.saturating_add(u64::try_from(line.len() + 1)?);
                records = records.saturating_add(1);
            }
        }
    }
    writer.flush()?;
    Ok(graphrag_core::PortablePayload {
        path: RECORDS_FILE.to_string(),
        sha256: format!("{:x}", hasher.finalize()),
        bytes,
        records,
    })
}

/// Immutable capture outcomes and caller provenance are user content, like
/// note bodies. Walking them would change the identity of a committed request
/// and make its original response unrecoverable after restore. Authentication
/// credentials are never part of these database fields.
fn opaque_client_fields(
    table: &str,
    record: &serde_json::Value,
) -> &'static [(&'static str, &'static str)] {
    match table {
        "remote_capture_receipt" | "remote_mutation_receipt" => &[("", "payload"), ("", "result")],
        "processing_job" if record["remote_instance_id"].is_string() => &[
            ("", "remote_input"),
            ("", "remote_admission"),
            ("", "remote_result"),
        ],
        "source"
            if record["source_type"] == "markdown"
                && record["uri"]
                    .as_str()
                    .is_some_and(|uri| uri.starts_with("mcp://upload/")) =>
        {
            &[
                ("/metadata/remote_upload", "source"),
                ("/metadata/remote_upload_pending", "source"),
            ]
        }
        "source"
            if record["source_type"] == "manual"
                && record["uri"]
                    .as_str()
                    .is_some_and(|uri| uri.starts_with("mcp://capture/")) =>
        {
            &[("/metadata/remote_capture", "source")]
        }
        _ => &[],
    }
}

fn take_opaque_client_fields(
    table: &str,
    record: &mut serde_json::Value,
) -> Vec<(&'static str, &'static str, serde_json::Value)> {
    opaque_client_fields(table, record)
        .iter()
        .filter_map(|&(parent, key)| {
            record
                .pointer_mut(parent)?
                .as_object_mut()?
                .remove(key)
                .map(|value| (parent, key, value))
        })
        .collect()
}

fn sanitize_portable_record(table: &str, record: &mut serde_json::Value, include_embeddings: bool) {
    let opaque = take_opaque_client_fields(table, record);
    sanitize_record(record, include_embeddings);
    for (parent, key, value) in opaque {
        // These known parent fields are retained by the generic sanitizer.
        if let Some(object) = record
            .pointer_mut(parent)
            .and_then(serde_json::Value::as_object_mut)
        {
            object.insert(key.into(), value);
        }
    }
}

fn sanitize_record(value: &mut serde_json::Value, include_embeddings: bool) {
    match value {
        serde_json::Value::Array(values) => {
            for value in values {
                sanitize_record(value, include_embeddings);
            }
        }
        serde_json::Value::Object(values) => {
            values.retain(|key, value| {
                let lower = key.to_ascii_lowercase();
                if (!include_embeddings
                    && matches!(key.as_str(), "embedding" | "summary_embedding"))
                    // Staging vectors are transient implementation state and
                    // must never leak into a portable archive, even when a
                    // caller elects to include a labelled active vector set.
                    || matches!(key.as_str(), "reindex_embedding" | "reindex_summary_embedding")
                    || matches!(lower.as_str(), "secret" | "api_key" | "token" | "password")
                {
                    return false;
                }
                if matches!(key.as_str(), "uri" | "normalized_uri" | "source_uri")
                    && value.as_str().is_some_and(is_local_absolute_path)
                {
                    return false;
                }
                sanitize_record(value, include_embeddings);
                true
            });
        }
        _ => {}
    }
}

fn is_local_absolute_path(value: &str) -> bool {
    value.starts_with("file://")
        || Path::new(value).is_absolute()
        || (value.len() >= 3
            && value.as_bytes()[1] == b':'
            && matches!(value.as_bytes()[2], b'/' | b'\\'))
}

fn strip_source_title_path(record: &mut serde_json::Value) {
    let Some(object) = record.as_object_mut() else {
        return;
    };
    if object
        .get("title")
        .and_then(serde_json::Value::as_str)
        .is_some_and(is_local_absolute_path)
    {
        object.remove("title");
    }
}

fn read_manifest(path: &Path) -> Result<PortableBackupManifest> {
    read_manifest_file(&path.join(MANIFEST_FILE))
}

fn read_manifest_file(manifest_path: &Path) -> Result<PortableBackupManifest> {
    let reader = File::open(manifest_path)
        .with_context(|| format!("open portable backup manifest {}", manifest_path.display()))?;
    serde_json::from_reader(reader)
        .with_context(|| format!("parse portable backup manifest {}", manifest_path.display()))
}

fn write_manifest(path: &Path, manifest: &PortableBackupManifest) -> Result<()> {
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    serde_json::to_writer_pretty(BufWriter::new(file), manifest)?;
    Ok(())
}

fn validate_manifest_schema(manifest: &PortableBackupManifest) -> Result<()> {
    manifest.validate_format().map_err(anyhow::Error::msg)?;
    if manifest.schema_version > migrations::latest_version() {
        bail!(
            "portable backup requires application schema {}, but this binary supports {}",
            manifest.schema_version,
            migrations::latest_version()
        );
    }
    Ok(())
}

#[derive(Default)]
struct RecordInspection {
    bytes: u64,
    sha256: String,
    records: u64,
    record_counts: BTreeMap<String, u64>,
    ids: BTreeSet<String>,
    references: Vec<(String, String)>,
}

fn inspect_records(path: &Path, manifest: &PortableBackupManifest) -> Result<RecordInspection> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut inspection = RecordInspection::default();
    let mut hasher = Sha256::new();
    let mut line = Vec::new();
    loop {
        line.clear();
        let read = reader.read_until(b'\n', &mut line)?;
        if read == 0 {
            break;
        }
        hasher.update(&line);
        inspection.bytes = inspection.bytes.saturating_add(u64::try_from(read)?);
        let payload = line.strip_suffix(b"\n").unwrap_or(&line);
        if payload.is_empty() {
            bail!("portable payload contains an empty JSONL record");
        }
        let record: PortableRecord =
            serde_json::from_slice(payload).context("portable payload contains invalid JSONL")?;
        if !PORTABLE_TABLES.contains(&record.table.as_str()) {
            bail!(
                "portable payload contains unsupported table {}",
                record.table
            );
        }
        validate_record_embeddings(&record.table, &record.record, manifest)?;
        let id = canonical_record_id(record.record.get("id"))
            .context("portable payload record is missing a usable id")?;
        if !id.starts_with(&format!("{}:", record.table)) {
            bail!(
                "portable payload record {id} is not in table {}",
                record.table
            );
        }
        if !inspection.ids.insert(id.clone()) {
            bail!("portable payload contains duplicate logical record id {id}");
        }
        inspection
            .references
            .extend(record_references(&record.table, &record.record)?);
        *inspection.record_counts.entry(record.table).or_default() += 1;
        inspection.records = inspection.records.saturating_add(1);
    }
    inspection.sha256 = format!("{:x}", hasher.finalize());
    Ok(inspection)
}

fn validate_record_embeddings(
    table: &str,
    record: &serde_json::Value,
    manifest: &PortableBackupManifest,
) -> Result<()> {
    fn visit(value: &serde_json::Value, dimension: Option<usize>) -> Result<()> {
        match value {
            serde_json::Value::Array(values) => {
                for value in values {
                    visit(value, dimension)?;
                }
            }
            serde_json::Value::Object(values) => {
                for (key, value) in values {
                    if matches!(key.as_str(), "embedding" | "summary_embedding") {
                        let expected = dimension.context(
                            "portable payload contains embeddings without a manifest identity",
                        )?;
                        let vector = value.as_array().context("embedding is not an array")?;
                        if vector.len() != expected {
                            bail!(
                                "portable payload embedding dimension {} does not match manifest {}",
                                vector.len(),
                                expected
                            );
                        }
                    } else {
                        visit(value, dimension)?;
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }
    // A caller metadata key named `embedding` is opaque text, not a stored
    // vector. Apply vector validation only to application-owned fields.
    let mut owned_fields = record.clone();
    take_opaque_client_fields(table, &mut owned_fields);
    visit(
        &owned_fields,
        manifest
            .embedding_identity
            .as_ref()
            .map(|identity| identity.dimension),
    )
}

fn canonical_record_id(value: Option<&serde_json::Value>) -> Option<String> {
    let value = value?;
    if let Some(value) = value.as_str() {
        return Some(value.to_string());
    }
    let object = value.as_object()?;
    let table = object.get("tb").or_else(|| object.get("table"))?.as_str()?;
    let key = object.get("id").or_else(|| object.get("key"))?;
    let key = key
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| key.to_string());
    Some(format!("{table}:{key}"))
}

fn record_references(table: &str, record: &serde_json::Value) -> Result<Vec<(String, String)>> {
    let expected: &[(&str, Option<&str>)] = match table {
        "note" => &[("source_id", Some("source"))],
        "message" => &[("conversation_id", Some("conversation"))],
        "supports" | "contradicts" | "derived_from" | "related_to" => &[
            ("in", Some("note")),
            ("out", Some("note")),
            ("proposal_id", Some("proposed_edge")),
        ],
        "proposed_edge" => &[
            ("in", Some("note")),
            ("out", Some("note")),
            ("resulting_edge_id", None),
        ],
        "mentions" => &[("in", Some("note")), ("out", Some("entity"))],
        "note_from_conversation" => &[("in", Some("note")), ("out", Some("conversation"))],
        "note_from_message" => &[("in", Some("note")), ("out", Some("message"))],
        "remote_capture_receipt" => &[("note_id", Some("note")), ("source_id", Some("source"))],
        "remote_mutation_receipt" => &[("target", None)],
        "processing_job" => &[("remote_source_id", Some("source"))],
        _ => &[],
    };
    let object = record
        .as_object()
        .context("portable record must be a JSON object")?;
    let historical_proposal = table == "proposed_edge"
        && matches!(
            object.get("status").and_then(serde_json::Value::as_str),
            Some("rejected" | "superseded")
        );
    let mut references = Vec::new();
    if table == "processing_job" {
        if object.get("job_type").and_then(serde_json::Value::as_str) != Some("remote_upload")
            || object
                .get("remote_instance_id")
                .and_then(serde_json::Value::as_str)
                .is_none()
        {
            bail!("portable processing_job must be a remote-owned upload job");
        }
        let items = object
            .get("item_ids")
            .and_then(serde_json::Value::as_array)
            .context("portable uploaded job item_ids must be an array")?;
        for value in items
            .iter()
            .chain(object.get("checkpoint").filter(|value| !value.is_null()))
        {
            let id = value
                .as_str()
                .context("portable uploaded job items/checkpoint must be canonical note IDs")?;
            graphrag_db::parse_portable_record_id(id, Some("note")).context(
                "portable uploaded job has a malformed or wrong-table historical note ID",
            )?;
        }
    }
    for (field, expected_table) in expected {
        let Some(value) = object.get(*field) else {
            continue;
        };
        if value.is_null() {
            continue;
        }
        let id = canonical_record_id(Some(value))
            .with_context(|| format!("portable {table} record has invalid reference in {field}"))?;
        if expected_table
            .is_some_and(|expected_table| !id.starts_with(&format!("{expected_table}:")))
        {
            bail!(
                "portable {table}.{field} must reference a {} record",
                expected_table.expect("checked above")
            );
        }
        if (historical_proposal && matches!(*field, "in" | "out"))
            || table == "remote_capture_receipt"
            || table == "remote_mutation_receipt"
            || table == "processing_job"
        {
            // Terminal proposals and successful capture receipts retain IDs
            // after source/note cleanup. Validate their shape without requiring
            // retired records to exist or recreating them during restore.
            // Resulting edges and every active proposal remain strict.
            graphrag_db::parse_portable_record_id(&id, *expected_table).with_context(|| {
                format!("portable {table} record has invalid reference in {field}")
            })?;
            continue;
        }
        references.push((format!("{table}.{field}"), id));
    }
    Ok(references)
}

fn validate_references(ids: &BTreeSet<String>, references: &[(String, String)]) -> Result<()> {
    for (field, id) in references {
        if !ids.contains(id) {
            bail!("portable backup has dangling reference {field} -> {id}");
        }
    }
    Ok(())
}

async fn restore_records(repo: &Repository, backup_path: &Path) -> Result<()> {
    let file = File::open(backup_path)?;
    let reader = BufReader::new(file);
    for line in reader.lines() {
        let line = line?;
        let record: PortableRecord = serde_json::from_str(&line)?;
        repo.restore_portable_record(&record.table, record.record)
            .await?;
    }
    Ok(())
}

fn jsonl_manifest_path(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.manifest.json", path.display()))
}

async fn count_repository_records(repo: &Repository) -> Result<BTreeMap<String, u64>> {
    let mut counts = BTreeMap::new();
    for table in PORTABLE_TABLES {
        let mut offset = 0;
        loop {
            let page = repo
                .portable_records_page(table, offset, EXPORT_PAGE_SIZE)
                .await?;
            if page.is_empty() {
                break;
            }
            offset += page.len();
        }
        if offset != 0 {
            counts.insert((*table).to_string(), u64::try_from(offset)?);
        }
    }
    Ok(counts)
}

fn count_records(path: &Path) -> Result<BTreeMap<String, u64>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut counts = BTreeMap::new();
    for line in reader.lines() {
        let record: PortableRecord = serde_json::from_str(&line?)?;
        *counts.entry(record.table).or_default() += 1;
    }
    Ok(counts)
}

fn ensure_absent_target(path: &Path) -> Result<()> {
    if path.exists() {
        bail!(
            "refusing to restore over existing target {}; choose a fresh, nonexistent --db-path",
            path.display()
        );
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        bail!("restore target parent {} does not exist", parent.display());
    }
    Ok(())
}

fn sibling_staging_path(path: &Path, operation: &str) -> Result<PathBuf> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("backup path must have a UTF-8 file name")?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_nanos();
    Ok(parent.join(format!(".{name}.{operation}-{nonce}")))
}

fn summary_from_manifest(
    path: &Path,
    manifest: &PortableBackupManifest,
    dry_run: bool,
) -> BackupSummary {
    BackupSummary {
        path: path.to_path_buf(),
        schema_version: manifest.schema_version,
        records: manifest.payload.records,
        record_counts: manifest.record_counts.clone(),
        includes_embeddings: manifest.includes_embeddings,
        dry_run,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use graphrag_core::{
        record_id_to_string, ChatConversation, ChatMessage, EdgeType, Entity, EntityType, Note,
        ProposedEdgeStatus, Source, SourceType,
    };
    use graphrag_db::{init_memory, repository::EdgeProposalDraft};
    use tempfile::tempdir;

    async fn populated_repo() -> Repository {
        let repo = Repository::new(init_memory().await.unwrap());
        let source = repo
            .create_source(Source::manual().with_title("portable fixture"))
            .await
            .unwrap();
        let first = repo
            .create_note(Note::new("portable first note").with_source(source.id.unwrap()))
            .await
            .unwrap();
        let second = repo
            .create_note(Note::new("portable second note"))
            .await
            .unwrap();
        let mut entity = Entity::new("Portable Entity", EntityType::Concept);
        entity.metadata = serde_json::json!({});
        let entity = repo.upsert_entity(entity).await.unwrap();
        repo.link_note_to_entity(first.id.as_ref().unwrap(), entity.id.as_ref().unwrap())
            .await
            .unwrap();
        repo.create_edge(
            first.id.as_ref().unwrap(),
            second.id.as_ref().unwrap(),
            EdgeType::Supports,
            Some(0.9),
        )
        .await
        .unwrap();
        repo
    }

    #[tokio::test]
    async fn portable_backup_round_trip_is_streamed_verified_and_restorable() {
        let temp = tempdir().unwrap();
        let backup_path = temp.path().join("backup");
        let repo = populated_repo().await;
        let created = create_backup(&repo, &backup_path, false).await.unwrap();
        assert_eq!(created.records, 6);
        assert!(!created.includes_embeddings);
        let payload = fs::read_to_string(backup_path.join(RECORDS_FILE)).unwrap();
        assert!(!payload.contains("\"embedding\""));
        assert_eq!(verify_backup(&backup_path).unwrap(), created);

        let restored = Repository::new(init_memory().await.unwrap());
        restore_records(&restored, &backup_path.join(RECORDS_FILE))
            .await
            .unwrap();
        assert_eq!(
            count_repository_records(&restored).await.unwrap(),
            created.record_counts
        );
        assert_eq!(
            restored
                .fulltext_search("portable", 10)
                .await
                .unwrap()
                .len(),
            2
        );
    }

    #[tokio::test]
    async fn uploaded_job_backup_preserves_replay_input_and_private_client_provenance() {
        use graphrag_db::{ProcessingJobStatus, RemoteJobLease, RemoteUploadInput};
        let temp = tempdir().unwrap();
        let repo = Repository::new(init_memory().await.unwrap());
        let input = RemoteUploadInput {
            authenticated_instance_id: "hermes-backup".into(),
            request_id: "upload-backup".into(),
            payload_fingerprint: "d".repeat(64),
            document_key: "logical-document".into(),
            markdown: "# Synthetic uploaded source\n\nportableuploadlexeme".into(),
            title: Some("Uploaded backup".into()),
            source_provenance: serde_json::json!({"uri":"file:///client-only/daily.md","metadata":{"token":"client vocabulary","embedding":"user label"}}),
            extract_entities: false,
            processing_options: serde_json::json!({"provider":"fixture","chunk_size":200}),
        };
        let admitted = repo.admit_remote_upload(input.clone()).await.unwrap();
        let id = admitted.result["job_id"].as_str().unwrap();
        let claimed = repo
            .claim_remote_upload_job(&input.authenticated_instance_id, id, "old-epoch", "worker")
            .await
            .unwrap();
        let lease = RemoteJobLease {
            job_id: claimed.job.id.clone().unwrap(),
            instance_id: input.authenticated_instance_id.clone(),
            service_epoch: "old-epoch".into(),
            worker_token: "worker".into(),
        };
        let source = repo.begin_remote_upload_generation(&lease).await.unwrap();
        repo.stage_remote_upload_notes(&lease, vec![Note::new(input.markdown.clone())])
            .await
            .unwrap();
        repo.reconcile_remote_upload(&lease, &[]).await.unwrap();
        let original = repo.finish_remote_upload_job(&lease, ProcessingJobStatus::Completed, None, Some(serde_json::json!({"source_id":record_id_to_string(source.id.as_ref().unwrap())}))).await.unwrap();
        // A second version prunes the original generated note; the first
        // completed job's immutable item IDs remain historical audit data.
        let mut next_input = input.clone();
        next_input.request_id = "upload-backup-new".into();
        next_input.payload_fingerprint = "e".repeat(64);
        next_input.markdown = "# Refreshed uploaded source\n\nnewportableuploadlexeme".into();
        let next = repo.admit_remote_upload(next_input.clone()).await.unwrap();
        let next_claim = repo
            .claim_remote_upload_job(
                &input.authenticated_instance_id,
                next.result["job_id"].as_str().unwrap(),
                "new-epoch",
                "new-worker",
            )
            .await
            .unwrap();
        let next_lease = RemoteJobLease {
            job_id: next_claim.job.id.unwrap(),
            instance_id: input.authenticated_instance_id.clone(),
            service_epoch: "new-epoch".into(),
            worker_token: "new-worker".into(),
        };
        repo.begin_remote_upload_generation(&next_lease)
            .await
            .unwrap();
        repo.stage_remote_upload_notes(&next_lease, vec![Note::new(next_input.markdown.clone())])
            .await
            .unwrap();
        repo.reconcile_remote_upload(&next_lease, &[])
            .await
            .unwrap();
        repo.finish_remote_upload_job(
            &next_lease,
            ProcessingJobStatus::Completed,
            None,
            Some(serde_json::json!({"refreshed":true})),
        )
        .await
        .unwrap();
        for old_id in &original.job.item_ids {
            assert!(repo.get_note(old_id).await.unwrap().is_none());
        }
        let mut host = Source::manual();
        host.uri = Some("file:///private/server/hidden.md".into());
        host.normalized_uri = host.uri.clone();
        host.metadata = serde_json::json!({"token":"real-host-secret"});
        repo.create_source(host).await.unwrap();
        let archive = temp.path().join("remote-jobs");
        let summary = create_backup(&repo, &archive, false).await.unwrap();
        assert_eq!(verify_backup(&archive).unwrap(), summary);
        let payload = fs::read_to_string(archive.join(RECORDS_FILE)).unwrap();
        assert!(payload.contains("file:///client-only/daily.md"));
        assert!(payload.contains("client vocabulary"));
        assert!(!payload.contains("/private/server"));
        assert!(!payload.contains("real-host-secret"));
        assert!(!payload.contains("token_sha256"));
        let target = temp.path().join("restored-jobs");
        restore_backup(&archive, &target, false).await.unwrap();
        let restored = Repository::new(init_persistent(&target).await.unwrap());
        let replay = restored
            .find_remote_upload_admission(
                &input.authenticated_instance_id,
                &input.request_id,
                &input.payload_fingerprint,
            )
            .await
            .unwrap()
            .unwrap();
        assert!(replay.replayed);
        assert_eq!(replay.result, admitted.result);
        let job = restored
            .get_remote_upload_job(&input.authenticated_instance_id, id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(job.input, input);
        assert_eq!(job.result, original.result);
        assert!(job.service_epoch.is_none());
        assert!(job.worker_token.is_none());
        for (case, wrong) in ["source:wrong-table", "note:⟨broken", "note:u'invalid-uuid'"]
            .into_iter()
            .enumerate()
        {
            for field in ["item_ids", "checkpoint"] {
                let invalid_archive = temp.path().join(format!("invalid-upload-{case}-{field}"));
                fs::create_dir(&invalid_archive).unwrap();
                let mut records = payload
                    .lines()
                    .map(|line| serde_json::from_str::<PortableRecord>(line).unwrap())
                    .collect::<Vec<_>>();
                let record = &mut records
                    .iter_mut()
                    .find(|record| record.table == "processing_job")
                    .unwrap()
                    .record;
                record[field] = if field == "item_ids" {
                    serde_json::json!([wrong])
                } else {
                    serde_json::json!(wrong)
                };
                let invalid_payload = records
                    .iter()
                    .map(|record| format!("{}\n", serde_json::to_string(record).unwrap()))
                    .collect::<String>();
                fs::write(invalid_archive.join(RECORDS_FILE), &invalid_payload).unwrap();
                let mut manifest = read_manifest(&archive).unwrap();
                manifest.payload.bytes = invalid_payload.len() as u64;
                manifest.payload.sha256 =
                    format!("{:x}", Sha256::digest(invalid_payload.as_bytes()));
                write_manifest(&invalid_archive.join(MANIFEST_FILE), &manifest).unwrap();
                assert!(
                    verify_backup(&invalid_archive).is_err(),
                    "verify accepted malformed upload {field}"
                );
                let invalid_target = temp.path().join(format!("invalid-restore-{case}-{field}"));
                assert!(restore_backup(&invalid_archive, &invalid_target, false)
                    .await
                    .is_err());
                assert!(!invalid_target.exists());
            }
        }
        let restored_source = restored
            .get_source(&record_id_to_string(source.id.as_ref().unwrap()))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            restored_source.metadata["remote_upload"]["source"],
            input.source_provenance
        );
    }

    fn remote_capture_fixture() -> graphrag_db::RemoteCaptureInput {
        graphrag_db::RemoteCaptureInput {
            authenticated_instance_id: "openclaw-office".into(),
            request_id: "backup-retry".into(),
            payload_fingerprint: "a".repeat(64),
            content: "Portable shared capture body.".into(),
            title: Some("Shared capture".into()),
            tags: vec!["shared".into()],
            source_provenance: serde_json::json!({
                "uri": "file:///client-owned/notes.md",
                "label": "Client document",
                "metadata": {
                    "token": "opaque client vocabulary",
                    "embedding": "a user label, not a vector",
                },
            }),
        }
    }

    #[tokio::test]
    async fn remote_capture_backup_preserves_exact_replay_and_client_provenance() {
        let temp = tempdir().unwrap();
        let repo = Repository::new(init_memory().await.unwrap());
        let input = remote_capture_fixture();
        let original = repo
            .capture_remote_note(input.clone(), vec![], vec![])
            .await
            .unwrap();
        let original_inspection = repo
            .inspect_record(original.result["id"].as_str().unwrap(), 0)
            .await
            .unwrap();
        let receipt = repo
            .portable_records_page("remote_capture_receipt", 0, 10)
            .await
            .unwrap()
            .remove(0);
        let mut host_source = Source::manual().with_title("ordinary host source");
        host_source.uri = Some("file:///private/server/folder.md".into());
        host_source.normalized_uri = host_source.uri.clone();
        host_source.metadata = serde_json::json!({
            "token": "server-secret-is-stripped",
            "api_key": "server-provider-key-is-stripped",
            // Matching object names alone must not bypass ordinary sanitization.
            "remote_capture": {"source": {"uri": "file:///private/server/nested.md", "password": "host-password-is-stripped"}},
        });
        repo.create_source(host_source).await.unwrap();
        let path = temp.path().join("shared-capture");
        let summary = create_backup(&repo, &path, false).await.unwrap();
        assert_eq!(verify_backup(&path).unwrap(), summary);
        let payload = fs::read_to_string(path.join(RECORDS_FILE)).unwrap();
        let records = payload
            .lines()
            .map(|line| serde_json::from_str::<PortableRecord>(line).unwrap())
            .collect::<Vec<_>>();
        let archived = &records
            .iter()
            .find(|record| record.table == "remote_capture_receipt")
            .unwrap()
            .record;
        assert_eq!(archived["payload"], receipt["payload"]);
        assert_eq!(archived["result"], original.result);
        for secret_or_host_path in [
            "/private/server",
            "server-secret-is-stripped",
            "server-provider-key-is-stripped",
            "host-password-is-stripped",
        ] {
            assert!(!payload.contains(secret_or_host_path));
        }
        // Authentication material is not accepted by capture storage; only the
        // trusted instance ID and opaque caller provenance form the journal.
        assert!(!payload.contains("token_sha256"));
        assert!(!payload.contains("Authorization"));
        let target = temp.path().join("shared-restored");
        restore_backup(&path, &target, false).await.unwrap();
        let restored = Repository::new(init_persistent(&target).await.unwrap());
        let replay = restored
            .find_remote_capture_receipt(
                &input.authenticated_instance_id,
                &input.request_id,
                &input.payload_fingerprint,
            )
            .await
            .unwrap()
            .unwrap();
        assert!(replay.replayed);
        assert_eq!(replay.result, original.result);
        let replay = restored
            .capture_remote_note(input, vec![], vec![])
            .await
            .unwrap();
        assert!(replay.replayed);
        assert_eq!(replay.result, original.result);
        assert_eq!(restored.get_stats().await.unwrap().note_count, 1);
        let inspection = restored
            .inspect_record(original.result["id"].as_str().unwrap(), 0)
            .await
            .unwrap();
        assert_eq!(inspection.provenance, original_inspection.provenance);
        assert_eq!(inspection.revision, original_inspection.revision);
    }

    #[tokio::test]
    async fn remote_mutation_backup_preserves_deleted_target_replay_and_opaque_payload() {
        let temp = tempdir().unwrap();
        let repo = Repository::new(init_memory().await.unwrap());
        let capture = remote_capture_fixture();
        let saved = repo
            .capture_remote_note(capture.clone(), vec![], vec![])
            .await
            .unwrap();
        let id = saved.result["id"].as_str().unwrap();
        let input = graphrag_db::RemoteMutationInput {
            instance_id: "hermes-reviewer".into(),
            request_id: "delete-original".into(),
            operation: "delete".into(),
            payload_fingerprint: "b".repeat(64),
            payload: serde_json::json!({"id":id,"confirmed":true,"token":"opaque user text","uri":"file:///client-owned/context.md"}),
            result: serde_json::json!({"id":id,"operation":"delete","original":"exact original outcome","embedding":"opaque user vocabulary"}),
        };
        let guard = repo.mutation_guard().await;
        let note = repo.mutation_note_snapshot(&guard, id).await.unwrap().note;
        let original = repo
            .apply_remote_mutation(
                &guard,
                input.clone(),
                graphrag_db::RemoteMutationEffect::Delete {
                    expected: Box::new(note),
                },
            )
            .await
            .unwrap();
        drop(guard);
        let path = temp.path().join("mutation-backup");
        let created = create_backup(&repo, &path, false).await.unwrap();
        assert_eq!(verify_backup(&path).unwrap(), created);
        let records = fs::read_to_string(path.join(RECORDS_FILE)).unwrap();
        assert!(records.contains("opaque user text"));
        assert!(records.contains("file:///client-owned/context.md"));
        let target = temp.path().join("mutation-restored");
        restore_backup(&path, &target, false).await.unwrap();
        let restored = Repository::new(init_persistent(&target).await.unwrap());
        assert_eq!(restored.get_stats().await.unwrap().note_count, 0);
        let replay = restored
            .find_remote_mutation_receipt(&input)
            .await
            .unwrap()
            .unwrap();
        assert!(replay.replayed);
        assert_eq!(replay.result, original.result);
        let captured = restored
            .find_remote_capture_receipt(
                &capture.authenticated_instance_id,
                &capture.request_id,
                &capture.payload_fingerprint,
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(captured.result, saved.result);
    }

    #[tokio::test]
    async fn deleted_remote_capture_records_keep_historical_receipts_after_backup_restore() {
        let temp = tempdir().unwrap();
        let repo = Repository::new(init_memory().await.unwrap());
        let input = remote_capture_fixture();
        let original = repo
            .capture_remote_note(input.clone(), vec![], vec![])
            .await
            .unwrap();
        let id = original.result["id"].as_str().unwrap();
        let source_id = original.result["provenance"]["source_id"].as_str().unwrap();
        let source = repo.get_source(source_id).await.unwrap().unwrap();
        repo.delete_note(id).await.unwrap();
        repo.delete_source(&source).await.unwrap();
        assert!(repo.get_source(source_id).await.unwrap().is_none());
        let path = temp.path().join("historical-capture");
        let summary = create_backup(&repo, &path, false).await.unwrap();
        assert_eq!(summary.records, 1);
        assert_eq!(verify_backup(&path).unwrap(), summary);
        let target = temp.path().join("historical-restored");
        restore_backup(&path, &target, false).await.unwrap();
        let restored = Repository::new(init_persistent(&target).await.unwrap());
        let replay = restored
            .capture_remote_note(input, vec![], vec![])
            .await
            .unwrap();
        assert!(replay.replayed);
        assert_eq!(replay.result, original.result);
        assert_eq!(restored.get_stats().await.unwrap().note_count, 0);
        assert_eq!(restored.get_stats().await.unwrap().source_count, 0);
        for invalid in ["note:`unterminated", "source:wrong-table"] {
            let mut receipt = repo
                .portable_records_page("remote_capture_receipt", 0, 10)
                .await
                .unwrap()
                .remove(0);
            receipt["note_id"] = serde_json::json!(invalid);
            assert!(record_references("remote_capture_receipt", &receipt).is_err());
        }
    }

    #[tokio::test]
    async fn schema_fifteen_v1_backup_remains_restorable_without_capture_table() {
        let temp = tempdir().unwrap();
        let path = temp.path().join("schema-fifteen");
        let repo = populated_repo().await;
        create_backup(&repo, &path, false).await.unwrap();
        // The pre-service v1 format is identical and simply has no receipt
        // records. Its recorded application schema must remain accepted.
        let mut manifest = read_manifest(&path).unwrap();
        manifest.schema_version = 15;
        assert_eq!(manifest.format_version, 1);
        assert!(!manifest
            .record_counts
            .contains_key("remote_capture_receipt"));
        fs::write(
            path.join(MANIFEST_FILE),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        verify_backup(&path).unwrap();
        let target = temp.path().join("old-restored");
        restore_backup(&path, &target, false).await.unwrap();
        let restored = Repository::new(init_persistent(&target).await.unwrap());
        assert_eq!(
            count_repository_records(&restored).await.unwrap(),
            manifest.record_counts
        );
        manifest.schema_version = migrations::latest_version() + 1;
        assert!(validate_manifest_schema(&manifest)
            .unwrap_err()
            .to_string()
            .contains("supports"));
    }

    #[tokio::test]
    async fn empty_backup_round_trip_is_valid_and_restores_a_fresh_schema() {
        let temp = tempdir().unwrap();
        let backup_path = temp.path().join("empty");
        let created = create_backup(
            &Repository::new(init_memory().await.unwrap()),
            &backup_path,
            false,
        )
        .await
        .unwrap();
        assert_eq!(created.records, 0);
        assert_eq!(verify_backup(&backup_path).unwrap(), created);
        let target = temp.path().join("empty-restored");
        let restored = restore_backup(&backup_path, &target, false).await.unwrap();
        assert_eq!(restored.records, 0);
        assert!(target.is_dir());
    }

    #[tokio::test]
    async fn chat_and_graph_records_round_trip_with_provenance_and_proposals() {
        let temp = tempdir().unwrap();
        let repo = populated_repo().await;
        let note = repo
            .create_note(Note::new("provenance note"))
            .await
            .unwrap();
        let conversation: ChatConversation = serde_json::from_value(serde_json::json!({
            "uuid": "portable-chat", "name": "Portable chat", "summary": "chat summary",
            "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z"
        }))
        .unwrap();
        let conversation_id = repo
            .upsert_conversation(&conversation, None, serde_json::json!({}), None)
            .await
            .unwrap();
        let message: ChatMessage = serde_json::from_value(serde_json::json!({
            "uuid": "portable-message", "sender": "human", "text": "portable chat message", "content": []
        }))
        .unwrap();
        let message_id = repo
            .upsert_message(&conversation_id, &conversation.uuid, 0, &message, None)
            .await
            .unwrap();
        repo.link_note_to_conversation(note.id.as_ref().unwrap(), &conversation_id)
            .await
            .unwrap();
        repo.link_note_to_message(note.id.as_ref().unwrap(), &message_id)
            .await
            .unwrap();
        let other = repo
            .create_note(Note::new("proposal endpoint"))
            .await
            .unwrap();
        let proposal = repo
            .upsert_edge_proposal(EdgeProposalDraft {
                from_id: note.id.clone().unwrap(),
                to_id: other.id.clone().unwrap(),
                edge_type: EdgeType::RelatedTo,
                confidence: 0.7,
                reason: "portable graph fixture".into(),
                generator: "test".into(),
                generator_version: Some("1".into()),
                model: None,
            })
            .await
            .unwrap();
        let backup_path = temp.path().join("chat-graph");
        let created = create_backup(&repo, &backup_path, false).await.unwrap();
        assert!(created
            .record_counts
            .get("conversation")
            .is_some_and(|count| *count == 1));
        assert!(created
            .record_counts
            .get("message")
            .is_some_and(|count| *count == 1));
        assert!(created
            .record_counts
            .get("proposed_edge")
            .is_some_and(|count| *count == 1));
        let target = temp.path().join("chat-graph-restored");
        restore_backup(&backup_path, &target, false).await.unwrap();
        let restored = Repository::new(init_persistent(&target).await.unwrap());
        assert_eq!(
            count_repository_records(&restored).await.unwrap(),
            created.record_counts
        );
        // Portable restore creates records at their exported logical IDs; it
        // does not best-effort remap graph/provenance endpoints.
        assert!(restored
            .get_note(&record_id_to_string(note.id.as_ref().unwrap()))
            .await
            .unwrap()
            .is_some());
        let restored_proposal = restored
            .get_edge_proposal(proposal.id.as_ref().unwrap())
            .await
            .unwrap()
            .unwrap();
        let expected_endpoints = [note.id.unwrap(), other.id.unwrap()];
        assert!(expected_endpoints.contains(&restored_proposal.from_id));
        assert!(expected_endpoints.contains(&restored_proposal.to_id));
        assert_ne!(restored_proposal.from_id, restored_proposal.to_id);
    }

    #[tokio::test]
    async fn source_prune_preserves_terminal_proposal_history_through_backup_restore() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut plan = repo
            .begin_file_import(
                SourceType::Markdown,
                "pruned.md".into(),
                "file:///portable/pruned.md".into(),
                "generated endpoint".into(),
                "generated-content-hash".into(),
                false,
            )
            .await
            .unwrap();
        let removed = repo
            .create_note(
                Note::new("generated endpoint")
                    .with_source(plan.source.id.as_ref().unwrap().clone())
                    .with_source_generation(plan.source.generation),
            )
            .await
            .unwrap()
            .id
            .unwrap();
        repo.complete_file_import(&mut plan.source).await.unwrap();

        let mut originals = Vec::new();
        for (status, label) in [
            (ProposedEdgeStatus::Pending, "pending"),
            (ProposedEdgeStatus::Accepted, "accepted"),
            (ProposedEdgeStatus::Rejected, "rejected"),
        ] {
            let target = repo
                .create_note(Note::new(format!("surviving {label} endpoint")))
                .await
                .unwrap()
                .id
                .unwrap();
            let proposal = repo
                .upsert_edge_proposal(EdgeProposalDraft {
                    from_id: removed.clone(),
                    to_id: target,
                    edge_type: EdgeType::RelatedTo,
                    confidence: 0.8,
                    reason: format!("original {label} relationship evidence"),
                    generator: "portable-lifecycle-fixture".into(),
                    generator_version: Some("1".into()),
                    model: Some("historical-model".into()),
                })
                .await
                .unwrap();
            let id = proposal.id.as_ref().unwrap();
            let reviewed = match status {
                ProposedEdgeStatus::Accepted => repo
                    .accept_edge_proposal(
                        id,
                        Some("original-reviewer".into()),
                        Some("original acceptance reason".into()),
                        true,
                    )
                    .await
                    .unwrap(),
                ProposedEdgeStatus::Rejected => repo
                    .reject_edge_proposal(
                        id,
                        Some("original-reviewer".into()),
                        Some("original rejection reason".into()),
                    )
                    .await
                    .unwrap(),
                _ => proposal,
            };
            originals.push(reviewed);
        }

        let deletion = repo.delete_source(&plan.source).await.unwrap();
        assert_eq!(deletion.notes, 1);
        assert_eq!(deletion.note_edges, 1);
        assert_eq!(deletion.proposals, 2);
        let mut history = Vec::new();
        for original in originals {
            let audit = repo
                .get_edge_proposal(original.id.as_ref().unwrap())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(audit.from_id, original.from_id);
            assert_eq!(audit.to_id, original.to_id);
            assert_eq!(audit.reason, original.reason);
            assert_eq!(audit.reviewer, original.reviewer);
            assert_eq!(audit.action_reason, original.action_reason);
            assert_eq!(audit.reviewed_at, original.reviewed_at);
            assert_eq!(audit.resulting_edge_id, None);
            if original.status == ProposedEdgeStatus::Rejected {
                assert_eq!(audit.status, ProposedEdgeStatus::Rejected);
            } else {
                assert_eq!(audit.status, ProposedEdgeStatus::Superseded);
                assert_eq!(
                    audit.supersession_reason.as_deref(),
                    Some("proposal endpoint removed by source lifecycle")
                );
                assert!(audit.superseded_at.is_some());
            }
            history.push(audit);
        }

        let temp = tempdir().unwrap();
        let backup_path = temp.path().join("pruned-source");
        let summary = create_backup(&repo, &backup_path, false).await.unwrap();
        assert_eq!(verify_backup(&backup_path).unwrap(), summary);
        assert_eq!(summary.record_counts.get("note"), Some(&3));
        assert_eq!(summary.record_counts.get("proposed_edge"), Some(&3));
        let restored = Repository::new(init_memory().await.unwrap());
        restore_records(&restored, &backup_path.join(RECORDS_FILE))
            .await
            .unwrap();
        assert!(restored
            .get_note(&record_id_to_string(&removed))
            .await
            .unwrap()
            .is_none());
        assert!(restored.list_note_edges(10).await.unwrap().is_empty());
        for audit in history {
            let id = audit.id.as_ref().unwrap();
            let restored_audit = restored.get_edge_proposal(id).await.unwrap().unwrap();
            assert_eq!(
                serde_json::to_value(restored_audit).unwrap(),
                serde_json::to_value(&audit).unwrap()
            );
            assert!(restored
                .accept_edge_proposal(id, None, None, true)
                .await
                .is_err());
        }
        let second_path = temp.path().join("restored-history");
        assert_eq!(
            create_backup(&restored, &second_path, false)
                .await
                .unwrap()
                .record_counts,
            summary.record_counts
        );
    }

    #[test]
    fn only_terminal_proposal_endpoints_may_reference_retired_notes() {
        let surviving_ids = BTreeSet::from(["note:surviving".to_string()]);
        for status in ["pending", "accepting", "accepted", "unknown"] {
            let record = serde_json::json!({
                "status": status, "in": "note:retired", "out": "note:surviving"
            });
            let references = record_references("proposed_edge", &record).unwrap();
            assert!(validate_references(&surviving_ids, &references)
                .unwrap_err()
                .to_string()
                .contains("dangling reference proposed_edge.in"));
        }
        let missing_status = serde_json::json!({
            "in": "note:retired", "out": "note:surviving"
        });
        assert!(validate_references(
            &surviving_ids,
            &record_references("proposed_edge", &missing_status).unwrap()
        )
        .is_err());
        for status in ["rejected", "superseded"] {
            let record = serde_json::json!({
                "status": status, "in": "note:retired", "out": "note:also-retired"
            });
            let original = record.clone();
            validate_references(
                &surviving_ids,
                &record_references("proposed_edge", &record).unwrap(),
            )
            .unwrap();
            assert_eq!(record, original, "validation preserves the audit payload");
        }
    }

    #[test]
    fn terminal_proposals_still_require_valid_ids_and_existing_resulting_edges() {
        for status in ["rejected", "superseded"] {
            for field in ["in", "out"] {
                for malformed in [
                    serde_json::json!("entity:retired"),
                    serde_json::json!("note:"),
                    serde_json::json!("note: "),
                    serde_json::json!("note"),
                    serde_json::json!("note:`unterminated"),
                    serde_json::json!("note:⟨unterminated"),
                    serde_json::json!("note:u'not-a-uuid'"),
                    serde_json::json!("note:`encoded` trailing"),
                    serde_json::json!(42),
                    serde_json::json!({"unexpected": "note:retired"}),
                ] {
                    let mut record = serde_json::json!({
                        "status": status, "in": "note:retired", "out": "note:also-retired"
                    });
                    record[field] = malformed;
                    assert!(record_references("proposed_edge", &record).is_err());
                }
            }
            let record = serde_json::json!({
                "status": status, "in": "note:retired", "out": "note:also-retired",
                "resulting_edge_id": "related_to:missing"
            });
            let references = record_references("proposed_edge", &record).unwrap();
            assert!(validate_references(&BTreeSet::new(), &references)
                .unwrap_err()
                .to_string()
                .contains("dangling reference proposed_edge.resulting_edge_id"));
            validate_references(
                &BTreeSet::from(["related_to:missing".to_string()]),
                &references,
            )
            .unwrap();
        }
    }

    #[tokio::test]
    async fn verify_and_restore_both_reject_malformed_encoded_terminal_endpoints() {
        let repo = populated_repo().await;
        let notes = repo.list_notes(10).await.unwrap();
        let proposal = repo
            .upsert_edge_proposal(EdgeProposalDraft {
                from_id: notes[0].id.clone(),
                to_id: notes[1].id.clone(),
                edge_type: EdgeType::RelatedTo,
                confidence: 0.8,
                reason: "terminal archive validation".into(),
                generator: "test".into(),
                generator_version: None,
                model: None,
            })
            .await
            .unwrap();
        repo.reject_edge_proposal(proposal.id.as_ref().unwrap(), None, None)
            .await
            .unwrap();
        let temp = tempdir().unwrap();
        let backup_path = temp.path().join("terminal-archive");
        create_backup(&repo, &backup_path, false).await.unwrap();
        let payload_path = backup_path.join(RECORDS_FILE);
        let original_records = fs::read_to_string(&payload_path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<PortableRecord>(line).unwrap())
            .collect::<Vec<_>>();
        let manifest_path = backup_path.join(MANIFEST_FILE);
        let original_manifest: PortableBackupManifest =
            serde_json::from_reader(File::open(&manifest_path).unwrap()).unwrap();
        for status in ["rejected", "superseded"] {
            for field in ["in", "out"] {
                let mut records = original_records.clone();
                let record = &mut records
                    .iter_mut()
                    .find(|record| record.table == "proposed_edge")
                    .unwrap()
                    .record;
                record["status"] = serde_json::json!(status);
                record[field] = serde_json::json!("note:`unterminated");
                let payload = records
                    .iter()
                    .map(serde_json::to_string)
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .unwrap()
                    .join("\n")
                    + "\n";
                fs::write(&payload_path, &payload).unwrap();
                let mut manifest = original_manifest.clone();
                manifest.payload.sha256 = format!("{:x}", Sha256::digest(payload.as_bytes()));
                manifest.payload.bytes = payload.len() as u64;
                fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
                assert!(verify_backup(&backup_path)
                    .unwrap_err()
                    .to_string()
                    .contains(&format!("invalid reference in {field}")));
                let restored = Repository::new(init_memory().await.unwrap());
                assert!(restore_records(&restored, &payload_path).await.is_err());
            }
        }
    }

    #[tokio::test]
    async fn default_export_removes_local_paths_and_secret_fields() {
        let temp = tempdir().unwrap();
        let repo = Repository::new(init_memory().await.unwrap());
        let mut source = Source::manual().with_title("private source");
        source.uri = Some("/private/operator/notes.md".into());
        source.normalized_uri = Some("file:///private/operator/notes.md".into());
        repo.create_source(source).await.unwrap();
        let path = temp.path().join("sanitized");
        create_backup(&repo, &path, false).await.unwrap();
        let payload = fs::read_to_string(path.join(RECORDS_FILE)).unwrap();
        assert!(!payload.contains("/private/operator"));
        let mut secret_shape =
            serde_json::json!({"api_key": "never-export", "nested": {"token": "also-never"}});
        sanitize_record(&mut secret_shape, false);
        assert_eq!(secret_shape, serde_json::json!({"nested": {}}));
    }

    #[tokio::test]
    async fn export_paginates_a_large_fixture_without_changing_counts() {
        let temp = tempdir().unwrap();
        let repo = Repository::new(init_memory().await.unwrap());
        for number in 0..(EXPORT_PAGE_SIZE * 2 + 1) {
            repo.create_note(Note::new(format!("paged portable note {number}")))
                .await
                .unwrap();
        }
        let path = temp.path().join("paged");
        let summary = create_backup(&repo, &path, false).await.unwrap();
        assert_eq!(
            summary.record_counts.get("note"),
            Some(&((EXPORT_PAGE_SIZE * 2 + 1) as u64))
        );
        assert_eq!(verify_backup(&path).unwrap(), summary);
    }

    #[tokio::test]
    async fn verify_rejects_checksum_corruption_and_dangling_references() {
        let temp = tempdir().unwrap();
        let backup_path = temp.path().join("backup");
        create_backup(&populated_repo().await, &backup_path, false)
            .await
            .unwrap();
        let payload_path = backup_path.join(RECORDS_FILE);
        let payload = fs::read_to_string(&payload_path).unwrap();
        fs::write(
            &payload_path,
            payload.replacen("portable first", "portable worst", 1),
        )
        .unwrap();
        assert!(verify_backup(&backup_path)
            .unwrap_err()
            .to_string()
            .contains("checksum"));

        let manifest_path = backup_path.join(MANIFEST_FILE);
        let mut manifest: PortableBackupManifest =
            serde_json::from_reader(File::open(&manifest_path).unwrap()).unwrap();
        manifest.format_version = graphrag_core::PORTABLE_BACKUP_FORMAT_VERSION + 1;
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        assert!(verify_backup(&backup_path)
            .unwrap_err()
            .to_string()
            .contains("unsupported"));

        let dangling_path = temp.path().join("dangling-backup");
        create_backup(&populated_repo().await, &dangling_path, false)
            .await
            .unwrap();
        let payload_path = dangling_path.join(RECORDS_FILE);
        let mut records = fs::read_to_string(&payload_path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<PortableRecord>(line).unwrap())
            .collect::<Vec<_>>();
        records
            .iter_mut()
            .find(|record| record.table == "note")
            .unwrap()
            .record["source_id"] = serde_json::json!("source:missing");
        let payload = records
            .iter()
            .map(serde_json::to_string)
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap()
            .join("\n")
            + "\n";
        fs::write(&payload_path, &payload).unwrap();
        let manifest_path = dangling_path.join(MANIFEST_FILE);
        let mut manifest: PortableBackupManifest =
            serde_json::from_reader(File::open(&manifest_path).unwrap()).unwrap();
        manifest.payload.sha256 = format!("{:x}", Sha256::digest(payload.as_bytes()));
        manifest.payload.bytes = payload.len() as u64;
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        assert!(verify_backup(&dangling_path)
            .unwrap_err()
            .to_string()
            .contains("dangling reference"));
    }

    #[tokio::test]
    async fn restore_dry_run_and_existing_target_never_mutate_the_target() {
        let temp = tempdir().unwrap();
        let backup_path = temp.path().join("backup");
        create_backup(&populated_repo().await, &backup_path, false)
            .await
            .unwrap();
        let target = temp.path().join("fresh-db");
        let dry_run = restore_backup(&backup_path, &target, true).await.unwrap();
        assert!(dry_run.dry_run);
        assert!(!target.exists());

        fs::create_dir(&target).unwrap();
        assert!(restore_backup(&backup_path, &target, false)
            .await
            .unwrap_err()
            .to_string()
            .contains("refusing to restore"));
    }

    #[tokio::test]
    async fn restore_stages_then_activates_a_fresh_persistent_target() {
        let temp = tempdir().unwrap();
        let backup_path = temp.path().join("backup");
        let created = create_backup(&populated_repo().await, &backup_path, false)
            .await
            .unwrap();
        let target = temp.path().join("fresh-db");
        let restored = restore_backup(&backup_path, &target, false).await.unwrap();
        assert_eq!(restored.record_counts, created.record_counts);
        assert!(target.is_dir());

        let reopened = Repository::new(init_persistent(&target).await.unwrap());
        assert_eq!(
            count_repository_records(&reopened).await.unwrap(),
            created.record_counts
        );
    }

    #[tokio::test]
    async fn jsonl_export_and_import_use_the_same_verified_fresh_target_flow() {
        let temp = tempdir().unwrap();
        let export_path = temp.path().join("notes.jsonl");
        let created = export_jsonl(&populated_repo().await, &export_path)
            .await
            .unwrap();
        assert!(export_path.is_file());
        assert!(jsonl_manifest_path(&export_path).is_file());
        assert_eq!(verify_jsonl(&export_path).unwrap(), created);

        let target = temp.path().join("jsonl-restored-db");
        let imported = import_jsonl(&export_path, &target, false).await.unwrap();
        assert_eq!(imported.record_counts, created.record_counts);
        let restored = Repository::new(init_persistent(&target).await.unwrap());
        assert_eq!(
            count_repository_records(&restored).await.unwrap(),
            created.record_counts
        );
    }

    #[tokio::test]
    async fn embeddings_require_and_record_model_identity() {
        let temp = tempdir().unwrap();
        let repo = Repository::new(init_memory().await.unwrap());
        repo.record_embedding_metadata(
            &graphrag_db::compatibility::EmbeddingIdentity::new("fixture", "model", 1024),
            None,
        )
        .await
        .unwrap();
        repo.create_note(Note::new("vector note").with_embedding(vec![0.1; 1024]))
            .await
            .unwrap();
        let backup_path = temp.path().join("vectors");
        let summary = create_backup(&repo, &backup_path, true).await.unwrap();
        assert!(summary.includes_embeddings);
        let manifest = read_manifest(&backup_path).unwrap();
        assert_eq!(
            manifest
                .embedding_identity
                .as_ref()
                .map(|identity| identity.dimension),
            Some(1024)
        );
        assert!(manifest.record_counts.contains_key("graphrag_metadata"));
        assert!(fs::read_to_string(backup_path.join(RECORDS_FILE))
            .unwrap()
            .contains("\"embedding\""));
    }

    #[tokio::test]
    async fn verify_rejects_embeddings_with_a_manifest_dimension_mismatch() {
        let temp = tempdir().unwrap();
        let repo = Repository::new(init_memory().await.unwrap());
        repo.record_embedding_metadata(
            &graphrag_db::compatibility::EmbeddingIdentity::new("fixture", "model", 1024),
            None,
        )
        .await
        .unwrap();
        repo.create_note(Note::new("dimension fixture").with_embedding(vec![0.1; 1024]))
            .await
            .unwrap();
        let path = temp.path().join("wrong-dimension");
        create_backup(&repo, &path, true).await.unwrap();
        let payload_path = path.join(RECORDS_FILE);
        let mut records = fs::read_to_string(&payload_path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<PortableRecord>(line).unwrap())
            .collect::<Vec<_>>();
        let vector = records
            .iter_mut()
            .find(|record| record.table == "note")
            .unwrap()
            .record["embedding"]
            .as_array_mut()
            .unwrap();
        vector.pop();
        let payload = records
            .iter()
            .map(serde_json::to_string)
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap()
            .join("\n")
            + "\n";
        fs::write(&payload_path, &payload).unwrap();
        let manifest_path = path.join(MANIFEST_FILE);
        let mut manifest: PortableBackupManifest =
            serde_json::from_reader(File::open(&manifest_path).unwrap()).unwrap();
        manifest.payload.sha256 = format!("{:x}", Sha256::digest(payload.as_bytes()));
        manifest.payload.bytes = payload.len() as u64;
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        assert!(verify_backup(&path)
            .unwrap_err()
            .to_string()
            .contains("dimension"));
    }
}
