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

pub const LATEST_SCHEMA_VERSION: u32 = 5;

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
        backfill_related_to_dedupe_keys(db)
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

/// Normalize the legacy unconstrained `related_to` table before its unique
/// canonical key is introduced. The migration runner owns this data rewrite
/// because it needs deterministic, row-by-row deduplication across a
/// symmetric pair while retaining one existing edge's audit metadata.
async fn backfill_related_to_dedupe_keys(db: &DbConnection) -> Result<()> {
    #[derive(Deserialize, SurrealValue)]
    struct RelatedEdgeRow {
        id: RecordId,
        r#in: RecordId,
        out: RecordId,
    }

    let edges: Vec<RelatedEdgeRow> = db
        .query("SELECT id, in, out FROM related_to ORDER BY id ASC")
        .await?
        .take(0)?;
    let mut retained_keys = BTreeSet::new();
    for edge in edges {
        let (from, to) = if record_id_to_string(&edge.r#in) <= record_id_to_string(&edge.out) {
            (edge.r#in, edge.out)
        } else {
            (edge.out, edge.r#in)
        };
        let dedupe_key = format!(
            "related_to:{}:{}",
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

    db.query(
        "DEFINE INDEX IF NOT EXISTS idx_related_to_dedupe ON related_to FIELDS dedupe_key UNIQUE",
    )
    .await?
    .check()?;
    Ok(())
}

fn migration_failed(migration: Migration, error: impl std::fmt::Display) -> DbError {
    DbError::MigrationFailed {
        version: migration.version,
        name: migration.name.to_string(),
        reason: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Repository;
    use graphrag_core::{record_id_to_string, EdgeType};
    use surrealdb::engine::local::Mem;
    use surrealdb::types::RecordId;
    use surrealdb::Surreal;

    async fn raw_memory_db() -> DbConnection {
        let db = Surreal::new::<Mem>(()).await.unwrap();
        db.use_ns("graphrag").use_db("notes").await.unwrap();
        db
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
    async fn v004_canonicalizes_legacy_related_to_pairs_before_adding_the_unique_key() {
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
        repo.accept_edge_proposal(&proposal.id.unwrap(), None, None, false)
            .await
            .unwrap();
        assert_eq!(repo.list_note_edges(10).await.unwrap().len(), 1);
        assert_eq!(
            repo.list_note_edges(10).await.unwrap()[0].edge_type,
            EdgeType::RelatedTo.to_string()
        );
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
