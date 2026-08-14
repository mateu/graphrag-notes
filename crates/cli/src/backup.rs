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

    let validation = inspect_records(&payload_path, &manifest)?;
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
                sanitize_record(&mut record, include_embeddings);
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
    let reader = File::open(&manifest_path)
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
        validate_record_embeddings(&record.record, manifest)?;
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
    visit(
        record,
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
        _ => &[],
    };
    let object = record
        .as_object()
        .context("portable record must be a JSON object")?;
    let mut references = Vec::new();
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
    use graphrag_core::{EdgeType, Entity, EntityType, Note, Source};
    use graphrag_db::init_memory;
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
}
