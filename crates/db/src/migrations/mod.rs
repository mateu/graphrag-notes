//! Application-managed database schema migrations.
//!
//! Migrations are additive and immutable. The embedded RocksDB engine prevents
//! multiple processes from opening the same database, while the async mutex
//! below serializes initialization within one process.

mod v001_initial;
mod v002_embedding_metadata;
mod v003_source_lifecycle;
mod v004_edge_proposals;
mod v005_proposal_supersession_audit;
mod v006_inference_cache;
mod v007_processing_jobs;
mod v008_markdown_chunks;
mod v009_entity_graph_aliases;
mod v010_reindex_staging;
mod v011_reindex_job_identity;
mod v012_reindex_item_fingerprints;
mod v013_reindex_ownership;
mod v014_reindex_input_snapshots;
mod v015_chat_metadata;
mod v016_remote_capture_receipts;
mod v017_remote_mutation_receipts;
mod v018_remote_upload_jobs;
mod v019_entity_identity;
mod v020_source_policy_migration;
mod v021_note_search;
mod v022_source_entity_enrichment;
mod v023_remote_job_family;
mod v024_entity_lexeme_index;
mod v025_mention_lexeme_evidence;
mod v026_provenance_endpoint_indexes;
mod v027_accepted_edge_frontier;

use crate::{DbConnection, DbError, Result};
use graphrag_core::record_id_to_string;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;
use surrealdb::types::RecordId;
use surrealdb_types::SurrealValue;
use tokio::sync::Mutex;
use tracing::info;

pub const LATEST_SCHEMA_VERSION: u32 = 27;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppliedMigration {
    pub version: u32,
    pub name: String,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Migration {
    pub(super) version: u32,
    pub(super) name: &'static str,
    pub(super) sql: &'static str,
}

const MIGRATIONS: &[Migration] = &[
    v001_initial::MIGRATION,
    v002_embedding_metadata::MIGRATION,
    v003_source_lifecycle::MIGRATION,
    v004_edge_proposals::MIGRATION,
    v005_proposal_supersession_audit::MIGRATION,
    v006_inference_cache::MIGRATION,
    v007_processing_jobs::MIGRATION,
    v008_markdown_chunks::MIGRATION,
    v009_entity_graph_aliases::MIGRATION,
    v010_reindex_staging::MIGRATION,
    v011_reindex_job_identity::MIGRATION,
    v012_reindex_item_fingerprints::MIGRATION,
    v013_reindex_ownership::MIGRATION,
    v014_reindex_input_snapshots::MIGRATION,
    v015_chat_metadata::MIGRATION,
    v016_remote_capture_receipts::MIGRATION,
    v017_remote_mutation_receipts::MIGRATION,
    v018_remote_upload_jobs::MIGRATION,
    v019_entity_identity::MIGRATION,
    v020_source_policy_migration::MIGRATION,
    v021_note_search::MIGRATION,
    v022_source_entity_enrichment::MIGRATION,
    v023_remote_job_family::MIGRATION,
    v024_entity_lexeme_index::MIGRATION,
    v025_mention_lexeme_evidence::MIGRATION,
    v026_provenance_endpoint_indexes::MIGRATION,
    v027_accepted_edge_frontier::MIGRATION,
];

// This table must exist before the first migration can be inspected. It is
// deliberately bootstrapped outside the numbered application migrations; the
// rows within it are the authoritative application migration history.
const MIGRATION_HISTORY_SCHEMA: &str = r#"
DEFINE TABLE IF NOT EXISTS schema_migration SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS version ON schema_migration TYPE int;
DEFINE FIELD IF NOT EXISTS name ON schema_migration TYPE string;
DEFINE FIELD IF NOT EXISTS checksum ON schema_migration TYPE string;
DEFINE FIELD IF NOT EXISTS applied_at ON schema_migration TYPE datetime DEFAULT time::now();
DEFINE INDEX IF NOT EXISTS idx_schema_migration_version ON schema_migration FIELDS version UNIQUE;

-- A durable start marker prevents a crashed or partially failed migration from
-- being rerun against a schema that may already have been changed by DDL.
DEFINE TABLE IF NOT EXISTS schema_migration_attempt SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS version ON schema_migration_attempt TYPE int;
DEFINE FIELD IF NOT EXISTS name ON schema_migration_attempt TYPE string;
DEFINE FIELD IF NOT EXISTS checksum ON schema_migration_attempt TYPE string;
DEFINE FIELD IF NOT EXISTS started_at ON schema_migration_attempt TYPE datetime DEFAULT time::now();
DEFINE INDEX IF NOT EXISTS idx_schema_migration_attempt_version ON schema_migration_attempt FIELDS version UNIQUE;
"#;

#[derive(Debug, Deserialize, SurrealValue)]
struct MigrationRecord {
    version: i64,
    name: String,
    checksum: String,
}

#[derive(Debug, Deserialize, SurrealValue)]
struct MigrationAttemptRecord {
    version: i64,
    name: String,
    checksum: String,
}

static MIGRATION_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn migration_lock() -> &'static Mutex<()> {
    MIGRATION_LOCK.get_or_init(|| Mutex::new(()))
}

/// Apply every pending application migration in order.
pub async fn apply_all(db: &DbConnection) -> Result<()> {
    apply_migrations(db, MIGRATIONS).await
}

/// Return the latest migration version understood by this binary.
pub const fn latest_version() -> u32 {
    LATEST_SCHEMA_VERSION
}

/// Return applied migrations after validating their history.
pub async fn applied_migrations(db: &DbConnection) -> Result<Vec<AppliedMigration>> {
    let migrations = load_applied_migrations(db).await?;
    validate_history(&migrations, MIGRATIONS)?;
    Ok(migrations
        .into_iter()
        .map(|migration| AppliedMigration {
            version: migration.version as u32,
            name: migration.name,
        })
        .collect())
}

/// Return the current application schema version, or zero for an uninitialized
/// database. Callers normally see the latest version because initialization
/// applies pending migrations first.
pub async fn current_version(db: &DbConnection) -> Result<u32> {
    Ok(applied_migrations(db)
        .await?
        .last()
        .map(|migration| migration.version)
        .unwrap_or(0))
}

async fn apply_migrations(db: &DbConnection, migrations: &[Migration]) -> Result<()> {
    validate_registry(migrations)?;

    let _guard = migration_lock().lock().await;
    ensure_migration_history(db).await?;
    let applied = load_applied_migrations(db).await?;
    reject_incomplete_attempts(db, &applied).await?;
    validate_history(&applied, migrations)?;

    let applied_versions: BTreeMap<u32, &str> = applied
        .iter()
        .map(|migration| (migration.version as u32, migration.name.as_str()))
        .collect();

    for migration in migrations {
        if applied_versions.contains_key(&migration.version) {
            continue;
        }

        info!(
            version = migration.version,
            name = migration.name,
            "Applying database schema migration"
        );
        apply_one(db, *migration).await?;
    }

    Ok(())
}

/// Reject a malformed binary migration registry before it can modify a database.
///
/// `validate_history` validates persisted rows, but an empty database has no
/// history to compare. Checking the registry independently prevents a gap,
/// duplicate, or out-of-order entry from being partly applied first.
fn validate_registry(migrations: &[Migration]) -> Result<()> {
    for (index, migration) in migrations.iter().enumerate() {
        let expected = u32::try_from(index + 1).expect("migration registry exceeds u32::MAX");
        if migration.version != expected {
            return Err(DbError::MigrationHistory(format!(
                "migration registry must be ordered and contiguous from version 1; expected version {}, found {} ({})",
                expected, migration.version, migration.name
            )));
        }
    }

    Ok(())
}

async fn ensure_migration_history(db: &DbConnection) -> Result<()> {
    let response =
        db.query(MIGRATION_HISTORY_SCHEMA)
            .await
            .map_err(|error| DbError::MigrationFailed {
                version: 0,
                name: "migration_history".to_string(),
                reason: error.to_string(),
            })?;
    response.check().map_err(|error| DbError::MigrationFailed {
        version: 0,
        name: "migration_history".to_string(),
        reason: error.to_string(),
    })?;
    Ok(())
}

async fn load_applied_migrations(db: &DbConnection) -> Result<Vec<MigrationRecord>> {
    Ok(db
        .query("SELECT version, name, checksum FROM schema_migration ORDER BY version ASC")
        .await?
        .take(0)?)
}

async fn reject_incomplete_attempts(db: &DbConnection, applied: &[MigrationRecord]) -> Result<()> {
    let attempts: Vec<MigrationAttemptRecord> = db
        .query("SELECT version, name, checksum FROM schema_migration_attempt ORDER BY version ASC")
        .await?
        .take(0)?;

    for attempt in attempts {
        if let Some(record) = applied
            .iter()
            .find(|record| record.version == attempt.version)
        {
            if record.name == attempt.name && record.checksum == attempt.checksum {
                db.query("DELETE schema_migration_attempt WHERE version = $version")
                    .bind(("version", attempt.version))
                    .await?
                    .check()?;
                continue;
            }
        }

        return Err(DbError::MigrationHistory(format!(
            "migration {} ({}) was interrupted or failed after it began; inspect the database before clearing its attempt record",
            attempt.version, attempt.name
        )));
    }

    Ok(())
}

fn validate_history(applied: &[MigrationRecord], migrations: &[Migration]) -> Result<()> {
    let latest = migrations
        .last()
        .map(|migration| migration.version)
        .unwrap_or(0);
    let expected: BTreeMap<u32, &Migration> = migrations
        .iter()
        .map(|migration| (migration.version, migration))
        .collect();

    let mut previous = 0_u32;
    for record in applied {
        let version = u32::try_from(record.version).map_err(|_| {
            DbError::MigrationHistory(format!(
                "migration version {} is not a positive u32",
                record.version
            ))
        })?;

        if version == 0 {
            return Err(DbError::MigrationHistory(
                "migration version 0 is invalid".to_string(),
            ));
        }
        if version > latest {
            return Err(DbError::UnsupportedSchemaVersion { version, latest });
        }
        if version != previous + 1 {
            return Err(DbError::MigrationHistory(format!(
                "expected migration {}, found {}",
                previous + 1,
                version
            )));
        }

        let expected_migration = expected.get(&version).ok_or_else(|| {
            DbError::MigrationHistory(format!("migration {} is not registered", version))
        })?;
        if record.name != expected_migration.name {
            return Err(DbError::MigrationHistory(format!(
                "migration {} is named {:?}, expected {:?}",
                version, record.name, expected_migration.name
            )));
        }
        if record.checksum != checksum(**expected_migration) {
            return Err(DbError::MigrationHistory(format!(
                "migration {} checksum does not match its registered SQL",
                version
            )));
        }

        previous = version;
    }

    Ok(())
}

fn checksum(migration: Migration) -> String {
    format!("{:x}", Sha256::digest(migration.sql.as_bytes()))
}

async fn apply_one(db: &DbConnection, migration: Migration) -> Result<()> {
    db.query(
        "INSERT INTO schema_migration_attempt (version, name, checksum, started_at) \
         VALUES ($version, $name, $checksum, time::now())",
    )
    .bind(("version", i64::from(migration.version)))
    .bind(("name", migration.name))
    .bind(("checksum", checksum(migration)))
    .await
    .map_err(|error| migration_failed(migration, error))?
    .check()
    .map_err(|error| migration_failed(migration, error))?;

    let definition = db
        .query(migration.sql)
        .await
        .map_err(|error| migration_failed(migration, error))?;
    definition
        .check()
        .map_err(|error| migration_failed(migration, error))?;

    if migration.version == v004_edge_proposals::MIGRATION.version {
        backfill_note_edge_dedupe_keys(db)
            .await
            .map_err(|error| migration_failed(migration, error))?;
    }

    if migration.version == v021_note_search::MIGRATION.version {
        v021_note_search::backfill(db)
            .await
            .map_err(|error| migration_failed(migration, error))?;
    }

    if migration.version == v024_entity_lexeme_index::MIGRATION.version {
        v024_entity_lexeme_index::backfill(db)
            .await
            .map_err(|error| migration_failed(migration, error))?;
    }

    if migration.version == 25 {
        v025_mention_lexeme_evidence::backfill(db)
            .await
            .map_err(|error| migration_failed(migration, error))?;
    }

    let recorded = db
        .query(
            "INSERT INTO schema_migration (version, name, checksum, applied_at) \
             VALUES ($version, $name, $checksum, time::now())",
        )
        .bind(("version", i64::from(migration.version)))
        .bind(("name", migration.name))
        .bind(("checksum", checksum(migration)))
        .await
        .map_err(|error| migration_failed(migration, error))?;
    recorded
        .check()
        .map_err(|error| migration_failed(migration, error))?;

    db.query("DELETE schema_migration_attempt WHERE version = $version")
        .bind(("version", i64::from(migration.version)))
        .await
        .map_err(|error| migration_failed(migration, error))?
        .check()
        .map_err(|error| migration_failed(migration, error))?;

    Ok(())
}

/// Normalize legacy note-edge rows before their unique dedupe keys are
/// introduced. The runner keeps the lowest record id deterministically,
/// canonicalizing only the symmetric `related_to` endpoints.
async fn backfill_note_edge_dedupe_keys(db: &DbConnection) -> Result<()> {
    #[derive(Deserialize, SurrealValue)]
    struct RelatedEdgeRow {
        id: RecordId,
        r#in: RecordId,
        out: RecordId,
    }

    for (table, symmetric) in [
        ("related_to", true),
        ("supports", false),
        ("contradicts", false),
        ("derived_from", false),
    ] {
        let edges: Vec<RelatedEdgeRow> = db
            .query(format!("SELECT id, in, out FROM {table} ORDER BY id ASC"))
            .await?
            .take(0)?;
        let mut retained_keys = BTreeSet::new();
        for edge in edges {
            let (from, to) =
                if symmetric && record_id_to_string(&edge.r#in) > record_id_to_string(&edge.out) {
                    (edge.out, edge.r#in)
                } else {
                    (edge.r#in, edge.out)
                };
            let dedupe_key = format!(
                "{table}:{}:{}",
                record_id_to_string(&from),
                record_id_to_string(&to)
            );
            if !retained_keys.insert(dedupe_key.clone()) {
                db.query("DELETE $id")
                    .bind(("id", edge.id))
                    .await?
                    .check()?;
                continue;
            }
            db.query("UPDATE $id SET in = $from, out = $to, dedupe_key = $dedupe_key")
                .bind(("id", edge.id))
                .bind(("from", from))
                .bind(("to", to))
                .bind(("dedupe_key", dedupe_key))
                .await?
                .check()?;
        }
        db.query(format!(
            "DEFINE INDEX IF NOT EXISTS idx_{table}_dedupe ON {table} FIELDS dedupe_key UNIQUE"
        ))
        .await?
        .check()?;
    }
    Ok(())
}

fn migration_failed(migration: Migration, error: impl std::fmt::Display) -> DbError {
    DbError::MigrationFailed {
        version: migration.version,
        name: migration.name.to_string(),
        reason: error.to_string(),
    }
}

/// Historical-schema constructors, excluded from normal production builds.
/// Changes here are covered by the conditional PR migration/schema gate.
#[cfg(any(test, feature = "test-support"))]
pub mod test_support {
    use super::{apply_migrations, DbConnection, DbError, Result, MIGRATIONS};
    use surrealdb::engine::local::Mem;
    use surrealdb::Surreal;
    use uuid::Uuid;

    /// Create a fresh, isolated database with only the registered migrations
    /// through `version`. Later definitions have never been installed, so
    /// production initialization can apply them even when they use plain DEFINE.
    pub async fn init_memory_at_version(version: u32) -> Result<DbConnection> {
        let prefix_len = MIGRATIONS.partition_point(|migration| migration.version <= version);
        if version != 0
            && MIGRATIONS[..prefix_len]
                .last()
                .map(|migration| migration.version)
                != Some(version)
        {
            return Err(DbError::UnsupportedSchemaVersion {
                version,
                latest: super::latest_version(),
            });
        }
        let db = DbConnection::new(Surreal::new::<Mem>(()).await?);
        let namespace = format!("graphrag_fixture_{}", Uuid::new_v4().simple());
        db.use_ns(namespace).use_db("notes").await?;
        apply_migrations(&db, &MIGRATIONS[..prefix_len]).await?;
        Ok(db)
    }

    /// Seed a small historical fixture from typed logical records, retaining
    /// only top-level fields defined by its actual schema. This is deliberately not the
    /// production portable restore, which synthesizes latest-schema authority.
    pub async fn copy_fixture_records(source: &DbConnection, target: &DbConnection) -> Result<()> {
        let schema = target
            .query("INFO FOR DB")
            .await?
            .take::<Option<serde_json::Value>>(0)?
            .ok_or_else(|| DbError::SchemaInit("missing fixture database schema".into()))?;
        let tables = schema["tables"].as_object().ok_or_else(|| {
            DbError::SchemaInit("fixture database has no table definitions".into())
        })?;
        for table in crate::PORTABLE_TABLES {
            if !tables.contains_key(*table) {
                continue;
            }
            let info = target
                .query(format!("INFO FOR TABLE {table}"))
                .await?
                .take::<Option<serde_json::Value>>(0)?
                .ok_or_else(|| DbError::SchemaInit(format!("missing fixture table {table}")))?;
            let fields = info["fields"].as_object().ok_or_else(|| {
                DbError::SchemaInit(format!("fixture table {table} has no field definitions"))
            })?;
            let mut records: Vec<surrealdb_types::Object> = source
                .query(format!("SELECT * FROM {table}"))
                .await?
                .take(0)?;
            for record in &mut records {
                record.retain(|field, _| field == "id" || fields.contains_key(field));
            }
            if !records.is_empty() {
                target
                    .query(format!("INSERT INTO {table} $records"))
                    .bind(("records", records))
                    .await?
                    .check()?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Repository;
    use graphrag_core::{record_id_to_string, ChatExport, EdgeType, Note, Source, SourceType};
    use surrealdb::engine::local::Mem;
    use surrealdb::types::RecordId;
    use surrealdb::Surreal;

    async fn raw_memory_db() -> DbConnection {
        let db = Surreal::new::<Mem>(()).await.unwrap();
        db.use_ns("graphrag").use_db("notes").await.unwrap();
        DbConnection::new(db)
    }

    #[tokio::test]
    async fn historical_fixture_latest_minus_one_upgrades_without_redefining_objects() {
        // Keep this relative to the registry: a new plain-DEFINE migration must
        // be absent from the fixture, not removed by another hand-written list.
        let previous = MIGRATIONS[MIGRATIONS.len() - 2].version;
        let db = test_support::init_memory_at_version(previous)
            .await
            .unwrap();
        assert_eq!(current_version(&db).await.unwrap(), previous);
        assert_eq!(
            applied_migrations(&db).await.unwrap().len(),
            MIGRATIONS.len() - 1
        );
        db.query("CREATE note:historical SET content = 'older corpus';")
            .await
            .unwrap()
            .check()
            .unwrap();

        for _ in 0..2 {
            crate::schema::initialize_schema(&db).await.unwrap();
        }
        assert_eq!(current_version(&db).await.unwrap(), latest_version());
        assert_eq!(
            applied_migrations(&db).await.unwrap().len(),
            MIGRATIONS.len()
        );
        let content: Vec<String> = db
            .query("SELECT VALUE content FROM note:historical")
            .await
            .unwrap()
            .take(0)
            .unwrap();
        assert_eq!(content, ["older corpus"]);
    }

    #[tokio::test]
    async fn historical_fixture_copy_preserves_typed_data_without_later_fields_or_tables() {
        let source = test_support::init_memory_at_version(latest_version())
            .await
            .unwrap();
        source.query("CREATE source:historical SET source_type = 'manual', status = 'ready', title = 'kept'; CREATE note:historical SET content = 'older corpus', source_id = source:historical;")
            .await.unwrap().check().unwrap();
        let before: Vec<serde_json::Value> = source
            .query("SELECT source_id, created_at FROM note:historical")
            .await
            .unwrap()
            .take(0)
            .unwrap();
        let target = test_support::init_memory_at_version(1).await.unwrap();
        test_support::copy_fixture_records(&source, &target)
            .await
            .unwrap();
        assert_eq!(current_version(&target).await.unwrap(), 1);
        let sources: Vec<serde_json::Value> = target
            .query("SELECT * FROM source:historical")
            .await
            .unwrap()
            .take(0)
            .unwrap();
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0]["title"], "kept");
        assert!(sources[0]
            .get("status")
            .is_none_or(serde_json::Value::is_null));
        crate::schema::initialize_schema(&target).await.unwrap();
        let after: Vec<serde_json::Value> = target
            .query("SELECT source_id, created_at FROM note:historical")
            .await
            .unwrap()
            .take(0)
            .unwrap();
        assert_eq!(before, after);
    }

    #[tokio::test]
    async fn historical_fixture_boundaries_are_validated_and_namespaces_are_isolated() {
        let empty = test_support::init_memory_at_version(0).await.unwrap();
        assert_eq!(current_version(&empty).await.unwrap(), 0);
        let latest = test_support::init_memory_at_version(latest_version())
            .await
            .unwrap();
        assert_eq!(current_version(&latest).await.unwrap(), latest_version());
        // These clients must not share even their migration history.
        assert_eq!(current_version(&empty).await.unwrap(), 0);
        assert!(matches!(
            test_support::init_memory_at_version(latest_version() + 1).await,
            Err(DbError::UnsupportedSchemaVersion { .. })
        ));
    }

    #[tokio::test]
    async fn fresh_database_reaches_latest_schema_version() {
        let db = raw_memory_db().await;
        apply_all(&db).await.unwrap();

        assert_eq!(current_version(&db).await.unwrap(), LATEST_SCHEMA_VERSION);
        let migrations = applied_migrations(&db).await.unwrap();
        assert_eq!(migrations.len(), LATEST_SCHEMA_VERSION as usize);
        assert_eq!(migrations[0].name, "initial_schema");
        assert_eq!(migrations[1].name, "embedding_metadata");
        assert_eq!(migrations[2].name, "source_lifecycle");
        assert_eq!(migrations[3].name, "edge_proposals");
    }

    #[tokio::test]
    async fn v025_mention_evidence_tracks_backfill_rebind_rename_delete_and_restore() {
        let db = raw_memory_db().await;
        apply_migrations(&db, &MIGRATIONS[..24]).await.unwrap();
        db.query("CREATE entity:a CONTENT {name: 'Atlas', canonical_name: 'atlas', entity_type: 'concept'}; CREATE entity:b CONTENT {name: 'Boreal', canonical_name: 'boreal', entity_type: 'concept'}; CREATE note:a CONTENT {content: 'a'}; CREATE note:b CONTENT {content: 'b'}; CREATE mentions:a SET in = note:a, out = entity:a, metadata = {aliases: ['First Alias']};").await.unwrap().check().unwrap();
        apply_all(&db).await.unwrap();
        for _ in 0..2 {
            v025_mention_lexeme_evidence::backfill(&db).await.unwrap();
        }
        let lexemes: Vec<String> = db
            .query("SELECT VALUE lexeme FROM mention_lexeme ORDER BY lexeme")
            .await
            .unwrap()
            .take(0)
            .unwrap();
        assert_eq!(lexemes, ["atlas", "first alias"]);
        db.query("UPDATE mentions:a SET in = note:b, out = entity:b, metadata = {aliases: ['Second Alias']}; UPDATE entity:b SET canonical_name = 'renamed boreal';").await.unwrap().check().unwrap();
        let lexemes: Vec<String> = db
            .query("SELECT VALUE lexeme FROM mention_lexeme ORDER BY lexeme")
            .await
            .unwrap()
            .take(0)
            .unwrap();
        assert_eq!(lexemes, ["renamed boreal", "second alias"]);
        let notes: Vec<RecordId> = db
            .query("SELECT VALUE note FROM mention_lexeme")
            .await
            .unwrap()
            .take(0)
            .unwrap();
        assert!(notes.iter().all(|id| *id == RecordId::new("note", "b")));
        db.query("DELETE mentions:a")
            .await
            .unwrap()
            .check()
            .unwrap();
        let rows: Vec<RecordId> = db
            .query("SELECT VALUE id FROM mention_lexeme")
            .await
            .unwrap()
            .take(0)
            .unwrap();
        assert!(rows.is_empty());
        // Portable restore and extraction both write the underlying relation;
        // database events must not depend on a particular repository method.
        db.query(
            "CREATE mentions:a SET in = note:a, out = entity:a, metadata = {aliases: ['Restored']}",
        )
        .await
        .unwrap()
        .check()
        .unwrap();
        let lexemes: Vec<String> = db
            .query("SELECT VALUE lexeme FROM mention_lexeme ORDER BY lexeme")
            .await
            .unwrap()
            .take(0)
            .unwrap();
        assert_eq!(lexemes, ["atlas", "restored"]);
        db.query("DELETE entity:a").await.unwrap().check().unwrap();
        let rows: Vec<RecordId> = db
            .query("SELECT VALUE id FROM mention_lexeme")
            .await
            .unwrap()
            .take(0)
            .unwrap();
        assert!(rows.is_empty());
    }

    #[tokio::test]
    async fn v024_backfills_entity_and_mention_alias_lexemes_idempotently() {
        let db = raw_memory_db().await;
        apply_migrations(&db, &MIGRATIONS[..23]).await.unwrap();
        db.query("CREATE entity:lexeme CONTENT { name: 'Atlas Engine', canonical_name: 'atlas engine', entity_type: 'concept', metadata: { aliases: ['Atlas Prime'] } }; CREATE note:lexeme CONTENT { content: 'fictional' }; CREATE mentions:lexeme SET in = note:lexeme, out = entity:lexeme, metadata = { aliases: ['Atlas Evidence'] };")
            .await
            .unwrap()
            .check()
            .unwrap();
        apply_all(&db).await.unwrap();
        let lexemes: Vec<String> = db
            .query("SELECT VALUE lexeme FROM entity_lexeme WHERE entity = entity:lexeme ORDER BY lexeme")
            .await
            .unwrap()
            .take(0)
            .unwrap();
        assert_eq!(lexemes, ["atlas engine", "atlas evidence", "atlas prime"]);
        v024_entity_lexeme_index::backfill(&db).await.unwrap();
        apply_all(&db).await.unwrap();
        let repeated: Vec<String> = db
            .query("SELECT VALUE lexeme FROM entity_lexeme WHERE entity = entity:lexeme ORDER BY lexeme")
            .await
            .unwrap()
            .take(0)
            .unwrap();
        assert_eq!(repeated, lexemes);
    }

    #[tokio::test]
    async fn v024_backfill_cursor_preserves_mixed_numeric_primary_order() {
        use surrealdb::types::{Number, RecordIdKey, Value};
        let db = raw_memory_db().await;
        apply_migrations(&db, &MIGRATIONS[..23]).await.unwrap();
        let mut keys = (1..=128)
            .map(|n| vec![Value::Number(Number::Int(1)), Value::Number(Number::Int(n))])
            .collect::<Vec<_>>();
        keys.push(vec![
            Value::Number(Number::Float(1.0)),
            Value::Number(Number::Int(0)),
        ]);
        keys.push(vec![
            Value::Number(Number::Float(1.0)),
            Value::Number(Number::Int(127)),
        ]);
        for (index, key) in keys.into_iter().enumerate() {
            db.query("CREATE $id SET name = $name, canonical_name = $name, entity_type = 'concept', metadata = {}")
                .bind(("id", RecordId::new("entity", RecordIdKey::Array(key.into()))))
                .bind(("name", format!("cursor{index}")))
                .await.unwrap().check().unwrap();
        }
        apply_all(&db).await.unwrap();
        for _ in 0..2 {
            let entities: Vec<RecordId> = db
                .query("SELECT VALUE entity FROM entity_lexeme")
                .await
                .unwrap()
                .take(0)
                .unwrap();
            assert_eq!(
                entities.len(),
                130,
                "every native record key must be indexed"
            );
            v024_entity_lexeme_index::backfill(&db).await.unwrap();
        }
        db.query("UPDATE entity SET name = 'Shared Compiler', canonical_name = 'shared compiler'")
            .await
            .unwrap()
            .check()
            .unwrap();
        v024_entity_lexeme_index::backfill(&db).await.unwrap();
        let shared: Vec<RecordId> = db
            .query("SELECT VALUE entity FROM entity_lexeme WHERE lexeme = 'shared compiler'")
            .await
            .unwrap()
            .take(0)
            .unwrap();
        assert_eq!(
            shared.len(),
            130,
            "shared spellings must retain every exact-native identity"
        );
    }

    #[tokio::test]
    async fn v019_preserves_legacy_ids_and_old_backup_identity_without_conflating_scopes() {
        use graphrag_core::{Entity, EntityType};
        let db = raw_memory_db().await;
        apply_migrations(&db, &MIGRATIONS[..18]).await.unwrap();
        db.query("CREATE entity:legacy CONTENT {name: 'Atlas', canonical_name: 'atlas', entity_type: 'concept', metadata: {aliases: ['Old Atlas']}}; CREATE note:legacy CONTENT {content: 'Atlas original evidence'}; CREATE mentions:legacy SET in = note:legacy, out = entity:legacy;").await.unwrap().check().unwrap();
        apply_all(&db).await.unwrap();
        let repo = Repository::new(db.clone());
        let legacy = repo
            .get_entities_for_note("note:legacy")
            .await
            .unwrap()
            .remove(0);
        assert_eq!(legacy.id, Some(RecordId::new("entity", "legacy")));
        assert_eq!(legacy.entity_type, EntityType::Concept);
        assert_eq!(legacy.identity_key.as_deref(), Some("legacy:atlas"));
        let mut manual = Entity::new("Atlas", EntityType::Project);
        manual.metadata = serde_json::json!({});
        assert_eq!(repo.upsert_entity(manual).await.unwrap().id, legacy.id);
        for scope in ["fictional-source-a", "fictional-source-b"] {
            let mut scoped = Entity::new("Atlas", EntityType::Project);
            scoped.identity_key = Some(scope.into());
            scoped.metadata = serde_json::json!({"aliases": ["Atlas launch"]});
            assert_ne!(repo.upsert_entity(scoped).await.unwrap().id, legacy.id);
        }
        let all = repo.portable_records_page("entity", 0, 20).await.unwrap();
        assert_eq!(all.len(), 3);
        let restored_db = crate::init_memory().await.unwrap();
        let restored = Repository::new(restored_db.clone());
        for mut record in all {
            if record["id"] == "entity:legacy" {
                // A pre-v019 archive omits this field. Restore keeps its ID
                // and uses the migration field's backwards-compatible value.
                record.as_object_mut().unwrap().remove("identity_key");
            }
            restored
                .restore_portable_record("entity", record)
                .await
                .unwrap();
        }
        let entities: Vec<Entity> = restored_db
            .query("SELECT * FROM entity ORDER BY identity_key")
            .await
            .unwrap()
            .take(0)
            .unwrap();
        assert_eq!(entities.len(), 3);
        assert!(entities.iter().any(|entity| entity.id == legacy.id
            && entity.identity_key.as_deref() == Some("legacy:atlas")));
        apply_all(&db).await.unwrap();
        assert_eq!(
            repo.get_entities_for_note("legacy").await.unwrap()[0].id,
            legacy.id
        );
    }

    #[test]
    fn historic_v001_checksum_is_immutable() {
        // This guards the v001 baseline used by databases that recorded the
        // migration before later additive schema changes existed. Altering it
        // would make startup reject those databases before upgrades can run.
        assert_eq!(
            checksum(v001_initial::MIGRATION),
            "df8157d6c1b27c25a97eefdc8025d3c50e977cdc62b8b47fef1074056e05dd53"
        );
    }

    #[tokio::test]
    async fn reapplying_current_schema_is_a_noop() {
        let db = raw_memory_db().await;
        apply_all(&db).await.unwrap();
        apply_all(&db).await.unwrap();

        assert_eq!(
            applied_migrations(&db).await.unwrap().len(),
            LATEST_SCHEMA_VERSION as usize
        );
    }

    #[tokio::test]
    async fn v023_backfills_remote_job_family_from_v022_history_and_journal() {
        let db = raw_memory_db().await;
        apply_migrations(&db, &MIGRATIONS[..22]).await.unwrap();
        assert_eq!(current_version(&db).await.unwrap(), 22);
        let v022_before = load_applied_migrations(&db)
            .await
            .unwrap()
            .into_iter()
            .find(|migration| migration.version == 22)
            .unwrap();
        assert_eq!(
            v022_before.checksum,
            "79024f458f630b3ac1bf940be358fac684e31bfedcb994834bf7cd6c9375bfb2"
        );

        db.query(
            "CREATE processing_job:ordinary SET job_type = 'remote_upload', status = 'completed', total_count = 1, completed_count = 1, failed_count = 0, remote_instance_id = 'owner', remote_request_id = 'ordinary', remote_payload_fingerprint = 'ordinary', remote_input = {}, remote_admission = {}, remote_source_uri = 'file:///ordinary', remote_phase = 'completed'; \
             CREATE processing_job:damaged SET job_type = 'remote_upload', status = 'failed', total_count = 1, completed_count = 0, failed_count = 1, remote_instance_id = 'owner', remote_request_id = 'damaged', remote_payload_fingerprint = 'damaged', remote_input = { enrichment: { damaged: true } }, remote_admission = {}, remote_source_uri = 'file:///damaged', remote_phase = 'failed'; \
             CREATE processing_job:null_input SET job_type = 'remote_upload', status = 'failed', total_count = 1, completed_count = 0, failed_count = 1, remote_instance_id = 'owner', remote_request_id = 'null-input', remote_payload_fingerprint = 'null-input', remote_input = { enrichment: NULL }, remote_admission = {}, remote_source_uri = 'file:///null-input', remote_phase = 'failed'; \
             CREATE processing_job:missing_input SET job_type = 'remote_upload', status = 'failed', total_count = 1, completed_count = 0, failed_count = 1, remote_instance_id = 'owner', remote_request_id = 'missing-input', remote_payload_fingerprint = 'missing-input', remote_input = {}, remote_admission = {}, remote_source_uri = 'file:///missing-input', remote_phase = 'failed'; \
             CREATE processing_job:cross_owner SET job_type = 'remote_upload', status = 'completed', total_count = 1, completed_count = 1, failed_count = 0, remote_instance_id = 'owner', remote_request_id = 'cross-owner', remote_payload_fingerprint = 'cross-owner', remote_input = {}, remote_admission = {}, remote_source_uri = 'file:///cross-owner', remote_phase = 'completed'; \
             CREATE processing_job:other_operation SET job_type = 'remote_upload', status = 'completed', total_count = 1, completed_count = 1, failed_count = 0, remote_instance_id = 'owner', remote_request_id = 'other-operation', remote_payload_fingerprint = 'other-operation', remote_input = {}, remote_admission = {}, remote_source_uri = 'file:///other-operation', remote_phase = 'completed'; \
             CREATE remote_mutation_receipt:null_input SET instance_id = 'owner', request_id = 'null-input', operation = 'enrich_source', target = source:null_input, payload_fingerprint = 'null-input', payload = {}, result = {}, created_at = time::now(), updated_at = time::now(); \
             CREATE remote_mutation_receipt:missing_input SET instance_id = 'owner', request_id = 'missing-input', operation = 'rollback_enrichment', target = source:missing_input, payload_fingerprint = 'missing-input', payload = {}, result = {}, created_at = time::now(), updated_at = time::now(); \
             CREATE remote_mutation_receipt:cross_owner SET instance_id = 'other-owner', request_id = 'cross-owner', operation = 'enrich_source', target = source:cross_owner, payload_fingerprint = 'cross-owner', payload = {}, result = {}, created_at = time::now(), updated_at = time::now(); \
             CREATE remote_mutation_receipt:other_operation SET instance_id = 'owner', request_id = 'other-operation', operation = 'edit_source', target = source:other_operation, payload_fingerprint = 'other-operation', payload = {}, result = {}, created_at = time::now(), updated_at = time::now();",
        )
        .await
        .unwrap()
        .check()
        .unwrap();

        apply_all(&db).await.unwrap();
        let families: Vec<serde_json::Value> = db
            .query(
                "SELECT type::string(id) AS id, remote_enrichment_job FROM processing_job \
                 WHERE job_type = 'remote_upload' ORDER BY id",
            )
            .await
            .unwrap()
            .take(0)
            .unwrap();
        assert_eq!(
            serde_json::Value::Array(families),
            serde_json::json!([
                {"id": "processing_job:cross_owner", "remote_enrichment_job": false},
                {"id": "processing_job:damaged", "remote_enrichment_job": true},
                {"id": "processing_job:missing_input", "remote_enrichment_job": true},
                {"id": "processing_job:null_input", "remote_enrichment_job": true},
                {"id": "processing_job:ordinary", "remote_enrichment_job": false},
                {"id": "processing_job:other_operation", "remote_enrichment_job": false},
            ])
        );

        let repo = Repository::new(db.clone());
        let ordinary = repo
            .list_remote_upload_job_statuses("owner", 20, false)
            .await
            .unwrap();
        assert_eq!(ordinary.len(), 3);
        assert!(ordinary
            .iter()
            .all(|status| status.enrichment == Some(false)));
        let all = repo
            .list_remote_upload_job_statuses("owner", 20, true)
            .await
            .unwrap();
        assert_eq!(all.len(), 6);
        let mut enrichment_ids = all
            .iter()
            .filter(|status| status.enrichment == Some(true))
            .map(|status| record_id_to_string(status.job.id.as_ref().unwrap()))
            .collect::<Vec<_>>();
        enrichment_ids.sort();
        assert_eq!(
            enrichment_ids,
            vec![
                "processing_job:damaged",
                "processing_job:missing_input",
                "processing_job:null_input",
            ]
        );

        assert!(db
            .query("UPDATE processing_job:ordinary SET remote_enrichment_job = true")
            .await
            .unwrap()
            .check()
            .is_err());
        let ordinary_family: Vec<bool> = db
            .query("SELECT VALUE remote_enrichment_job FROM processing_job:ordinary")
            .await
            .unwrap()
            .take(0)
            .unwrap();
        assert_eq!(ordinary_family, vec![false]);

        let history_before_retry = load_applied_migrations(&db).await.unwrap();
        apply_all(&db).await.unwrap();
        let history_after_retry = load_applied_migrations(&db).await.unwrap();
        assert_eq!(history_after_retry.len(), history_before_retry.len());
        for (before, after) in history_before_retry.iter().zip(&history_after_retry) {
            assert_eq!(after.version, before.version);
            assert_eq!(after.name, before.name);
            assert_eq!(after.checksum, before.checksum);
        }
        assert_eq!(
            history_before_retry
                .into_iter()
                .find(|migration| migration.version == 22)
                .unwrap()
                .checksum,
            v022_before.checksum
        );
    }

    #[tokio::test]
    async fn prior_schema_registry_rejects_policy_checkpoint_before_any_worker_write() {
        let db = crate::init_memory().await.unwrap();
        db.query("CREATE processing_job:migration SET job_type = 'remote_upload', status = 'queued', total_count = 1, completed_count = 1, failed_count = 0, remote_phase = 'migration_extracting', remote_migration_contract_version = 1, remote_result = {policy_migration_stage: {version: 1, batches: [{fixture: 'exact private checkpoint'}]}};")
            .await.unwrap().check().unwrap();
        let before: Vec<serde_json::Value> = db
            .query("SELECT * FROM processing_job:migration")
            .await
            .unwrap()
            .take(0)
            .unwrap();
        // Run the exact migration prefix understood by rc.3. It must refuse
        // schema20 before job claims or unknown-phase failure settlement can
        // overwrite any prepared bytes.
        let error = apply_migrations(&db, &MIGRATIONS[..19]).await.unwrap_err();
        assert!(error.to_string().contains("20"));
        let after: Vec<serde_json::Value> = db
            .query("SELECT * FROM processing_job:migration")
            .await
            .unwrap()
            .take(0)
            .unwrap();
        assert_eq!(after, before);
        assert_eq!(after[0]["status"], "queued");
        assert!(after[0]["remote_result"]["policy_migration_stage"].is_object());
    }

    #[tokio::test]
    async fn future_schema_version_is_rejected() {
        let db = raw_memory_db().await;
        apply_all(&db).await.unwrap();
        db.query(
            "INSERT INTO schema_migration (version, name, checksum, applied_at) \
             VALUES (99, 'future_schema', 'future', time::now())",
        )
        .await
        .unwrap()
        .check()
        .unwrap();

        let error = apply_all(&db).await.unwrap_err();
        assert!(matches!(
            error,
            DbError::UnsupportedSchemaVersion {
                version: 99,
                latest: LATEST_SCHEMA_VERSION
            }
        ));
    }

    #[tokio::test]
    async fn failed_migration_is_not_recorded() {
        let db = raw_memory_db().await;
        apply_all(&db).await.unwrap();
        let invalid = Migration {
            version: 5,
            name: "invalid_test_migration",
            sql: "DEFINE TABLE invalid_test_probe SCHEMAFULL; THIS IS NOT VALID SURREALQL;",
        };

        let error = apply_one(&db, invalid).await.unwrap_err();
        assert!(matches!(error, DbError::MigrationFailed { version: 5, .. }));
        assert_eq!(
            applied_migrations(&db).await.unwrap().len(),
            LATEST_SCHEMA_VERSION as usize
        );

        let retry = apply_migrations(
            &db,
            &[
                v001_initial::MIGRATION,
                v002_embedding_metadata::MIGRATION,
                v003_source_lifecycle::MIGRATION,
                v004_edge_proposals::MIGRATION,
                invalid,
            ],
        )
        .await
        .unwrap_err();
        assert!(matches!(retry, DbError::MigrationHistory(_)));
    }

    #[tokio::test]
    async fn changed_migration_sql_is_rejected_by_its_checksum() {
        let db = raw_memory_db().await;
        apply_all(&db).await.unwrap();
        db.query("UPDATE schema_migration SET checksum = 'tampered' WHERE version = 1")
            .await
            .unwrap()
            .check()
            .unwrap();

        let error = applied_migrations(&db).await.unwrap_err();
        assert!(matches!(error, DbError::MigrationHistory(_)));
    }

    #[tokio::test]
    async fn a_database_at_v001_can_upgrade_without_losing_data() {
        let db = raw_memory_db().await;
        apply_migrations(&db, &[v001_initial::MIGRATION])
            .await
            .unwrap();
        db.query("CREATE note CONTENT { content: 'legacy note' }")
            .await
            .unwrap()
            .check()
            .unwrap();

        let v002 = Migration {
            version: 2,
            name: "test_additive_upgrade",
            sql: "DEFINE TABLE IF NOT EXISTS migration_upgrade_probe SCHEMAFULL;",
        };
        apply_migrations(&db, &[v001_initial::MIGRATION, v002])
            .await
            .unwrap();

        let notes: Vec<serde_json::Value> = db.select("note").await.unwrap();
        assert_eq!(notes.len(), 1);
        let migrations = load_applied_migrations(&db).await.unwrap();
        assert_eq!(migrations.len(), 2);
    }

    #[tokio::test]
    async fn v016_adds_capture_receipts_without_rewriting_existing_knowledge() {
        let db = raw_memory_db().await;
        apply_migrations(&db, &MIGRATIONS[..15]).await.unwrap();
        db.query("CREATE note SET content = 'pre-service manual note', tags = ['preserved']; \
                  CREATE source SET source_type = 'manual', metadata = { legacy: { value: 'preserved' } }")
            .await.unwrap().check().unwrap();
        let notes_before: Vec<serde_json::Value> = db.select("note").await.unwrap();
        let sources_before: Vec<serde_json::Value> = db.select("source").await.unwrap();
        let history_before = load_applied_migrations(&db).await.unwrap();
        apply_migrations(&db, &MIGRATIONS[..16]).await.unwrap();
        let notes_after: Vec<serde_json::Value> = db.select("note").await.unwrap();
        let sources_after: Vec<serde_json::Value> = db.select("source").await.unwrap();
        assert_eq!(notes_after, notes_before);
        assert_eq!(sources_after, sources_before);
        assert_eq!(current_version(&db).await.unwrap(), 16);
        let history_after = load_applied_migrations(&db).await.unwrap();
        assert_eq!(history_after.len(), 16);
        for (before, after) in history_before.iter().zip(history_after.iter()) {
            assert_eq!(before.version, after.version);
            assert_eq!(before.checksum, after.checksum);
        }
        let receipts: Vec<serde_json::Value> = db.select("remote_capture_receipt").await.unwrap();
        assert!(receipts.is_empty());
    }

    #[tokio::test]
    async fn v017_adds_mutation_journal_without_rewriting_capture_receipts() {
        let db = raw_memory_db().await;
        apply_migrations(&db, &MIGRATIONS[..16]).await.unwrap();
        db.query("CREATE remote_capture_receipt SET instance_id='legacy',request_id='capture',payload_fingerprint='original',payload={content:'preserved'},result={revision:'original'},note_id=note:retired,source_id=source:retired,created_at=time::now(),updated_at=time::now(); CREATE note SET content='existing knowledge';").await.unwrap().check().unwrap();
        let receipts: Vec<serde_json::Value> = db.select("remote_capture_receipt").await.unwrap();
        let notes: Vec<serde_json::Value> = db.select("note").await.unwrap();
        let history = load_applied_migrations(&db).await.unwrap();
        apply_migrations(&db, &MIGRATIONS[..17]).await.unwrap();
        assert_eq!(current_version(&db).await.unwrap(), 17);
        assert_eq!(
            db.select::<Vec<serde_json::Value>>("remote_capture_receipt")
                .await
                .unwrap(),
            receipts
        );
        assert_eq!(
            db.select::<Vec<serde_json::Value>>("note").await.unwrap(),
            notes
        );
        assert!(db
            .select::<Vec<serde_json::Value>>("remote_mutation_receipt")
            .await
            .unwrap()
            .is_empty());
        let after = load_applied_migrations(&db).await.unwrap();
        for (before, after) in history.iter().zip(after.iter()) {
            assert_eq!(before.checksum, after.checksum);
        }
    }

    #[tokio::test]
    async fn v015_preserves_existing_records_and_accepts_librarian_chat_metadata() {
        let db = raw_memory_db().await;
        apply_migrations(&db, &MIGRATIONS[..14]).await.unwrap();
        let repo = Repository::new(db.clone());
        let conversation = ChatExport::from_json(
            r#"[{
                "uuid": "metadata-upgrade-conversation",
                "name": "Atlas upgrade discussion",
                "summary": "Retain the original chat while upgrading its metadata schema.",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:05:00Z",
                "chat_messages": []
            }]"#,
        )
        .unwrap()
        .conversations
        .remove(0);
        let conversation_id = repo
            .upsert_conversation(&conversation, None, serde_json::json!({}), None)
            .await
            .unwrap();
        let source = repo
            .create_source(Source::from_file("legacy-atlas.md", SourceType::Markdown).unwrap())
            .await
            .unwrap();
        let source_id = source.id.unwrap();
        let note = repo
            .create_note(Note::new("Keep this existing note.").with_source(source_id.clone()))
            .await
            .unwrap();
        // This is the same nonempty metadata written by LibrarianAgent for
        // conversations and Source::chat_export; v014 rejects its child keys.
        let metadata = serde_json::json!({
            "conversation_id": &conversation.uuid,
            "created_at": &conversation.created_at,
            "summary": &conversation.summary,
        });
        assert!(repo
            .upsert_conversation(&conversation, None, metadata.clone(), None)
            .await
            .is_err());

        apply_all(&db).await.unwrap();
        let upgraded_id = repo
            .upsert_conversation(&conversation, None, metadata.clone(), None)
            .await
            .unwrap();
        assert_eq!(upgraded_id, conversation_id);
        let persisted_metadata: Vec<serde_json::Value> = db
            .query("SELECT VALUE metadata FROM conversation WHERE id = $id")
            .bind(("id", conversation_id))
            .await
            .unwrap()
            .take(0)
            .unwrap();
        assert_eq!(persisted_metadata, vec![metadata.clone()]);
        let chat_source = repo
            .create_source(
                Source::chat_export(conversation.display_title(), None)
                    .with_metadata(metadata.clone()),
            )
            .await
            .unwrap();
        assert_eq!(chat_source.metadata, metadata);
        let retained_source = repo
            .get_source(&record_id_to_string(&source_id))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(retained_source.title.as_deref(), Some("legacy-atlas.md"));
        assert_eq!(retained_source.metadata, serde_json::json!({}));
        let retained_note = repo
            .get_note(&record_id_to_string(note.id.as_ref().unwrap()))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(retained_note.content, "Keep this existing note.");
        assert_eq!(retained_note.source_id, Some(source_id));
        assert_eq!(current_version(&db).await.unwrap(), latest_version());
    }

    #[tokio::test]
    async fn v004_backfills_and_deduplicates_legacy_note_edge_keys_before_indexes() {
        #[derive(Deserialize, SurrealValue)]
        struct IdRow {
            id: RecordId,
        }
        #[derive(Deserialize, SurrealValue)]
        struct RelatedRow {
            id: RecordId,
            r#in: RecordId,
            out: RecordId,
            dedupe_key: String,
        }

        let db = raw_memory_db().await;
        apply_migrations(
            &db,
            &[
                v001_initial::MIGRATION,
                v002_embedding_metadata::MIGRATION,
                v003_source_lifecycle::MIGRATION,
            ],
        )
        .await
        .unwrap();
        let created: Vec<IdRow> = db
            .query("CREATE note CONTENT { content: 'legacy related note' } RETURN id")
            .await
            .unwrap()
            .take(0)
            .unwrap();
        let first = created[0].id.clone();
        let second: Vec<IdRow> = db
            .query("CREATE note CONTENT { content: 'another legacy related note' } RETURN id")
            .await
            .unwrap()
            .take(0)
            .unwrap();
        let second = second.into_iter().next().unwrap();
        db.query("CREATE related_to SET in = $in, out = $out, confidence = 0.8")
            .bind(("in", first.clone()))
            .bind(("out", second.id.clone()))
            .await
            .unwrap()
            .check()
            .unwrap();
        db.query("CREATE related_to SET in = $in, out = $out, confidence = 0.9")
            .bind(("in", second.id.clone()))
            .bind(("out", first.clone()))
            .await
            .unwrap()
            .check()
            .unwrap();
        for table in ["supports", "contradicts", "derived_from"] {
            for _ in 0..2 {
                db.query(format!("CREATE {table} SET in = $in, out = $out"))
                    .bind(("in", first.clone()))
                    .bind(("out", second.id.clone()))
                    .await
                    .unwrap()
                    .check()
                    .unwrap();
            }
        }

        apply_migrations(&db, MIGRATIONS).await.unwrap();
        let rows: Vec<RelatedRow> = db
            .query("SELECT id, in, out, dedupe_key FROM related_to")
            .await
            .unwrap()
            .take(0)
            .unwrap();
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        let from = record_id_to_string(&row.r#in);
        let to = record_id_to_string(&row.out);
        assert!(from < to);
        assert_eq!(row.dedupe_key, format!("related_to:{from}:{to}"));
        for table in ["supports", "contradicts", "derived_from"] {
            #[derive(Deserialize, SurrealValue)]
            struct DirectedRow {
                dedupe_key: String,
            }
            let rows: Vec<DirectedRow> = db
                .query(format!("SELECT dedupe_key FROM {table}"))
                .await
                .unwrap()
                .take(0)
                .unwrap();
            assert_eq!(rows.len(), 1, "{table} should retain one directed row");
            assert_eq!(
                rows[0].dedupe_key,
                format!(
                    "{table}:{}:{}",
                    record_id_to_string(&first),
                    record_id_to_string(&second.id)
                )
            );
        }

        let repo = Repository::new(db);
        let proposal = repo
            .upsert_gardener_proposal(
                &first,
                &second.id,
                0.95,
                "legacy pair re-scan".into(),
                None,
                None,
            )
            .await
            .unwrap();
        assert!(repo
            .accept_edge_proposal(&proposal.id.clone().unwrap(), None, None, false)
            .await
            .is_err());
        assert_eq!(
            repo.get_edge_proposal(&proposal.id.unwrap())
                .await
                .unwrap()
                .unwrap()
                .status,
            graphrag_core::ProposedEdgeStatus::Superseded
        );
        let edges = repo.list_note_edges(10).await.unwrap();
        assert_eq!(edges.len(), 4);
        assert!(edges
            .iter()
            .any(|edge| edge.edge_type == EdgeType::RelatedTo.to_string()));
    }

    #[tokio::test]
    async fn concurrent_initialization_records_each_migration_once() {
        let db = raw_memory_db().await;
        let first = apply_all(&db);
        let second = apply_all(&db);
        let (first, second) = tokio::join!(first, second);
        first.unwrap();
        second.unwrap();

        assert_eq!(
            applied_migrations(&db).await.unwrap().len(),
            LATEST_SCHEMA_VERSION as usize
        );
    }

    #[test]
    fn malformed_migration_registry_is_rejected_before_application() {
        let v003 = Migration {
            version: 3,
            name: "gap",
            sql: "",
        };
        let duplicate_v001 = Migration {
            version: 1,
            name: "duplicate",
            sql: "",
        };

        assert!(matches!(
            validate_registry(&[v001_initial::MIGRATION, v003]),
            Err(DbError::MigrationHistory(_))
        ));
        assert!(matches!(
            validate_registry(&[v001_initial::MIGRATION, duplicate_v001]),
            Err(DbError::MigrationHistory(_))
        ));
    }
}
