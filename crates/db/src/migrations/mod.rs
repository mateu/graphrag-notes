//! Application-managed database schema migrations.
//!
//! Migrations are additive and immutable. The embedded RocksDB engine prevents
//! multiple processes from opening the same database, while the async mutex
//! below serializes initialization within one process.

mod v001_initial;

use crate::{DbConnection, DbError, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::sync::OnceLock;
use surrealdb_types::SurrealValue;
use tokio::sync::Mutex;
use tracing::info;

pub const LATEST_SCHEMA_VERSION: u32 = 1;

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

const MIGRATIONS: &[Migration] = &[v001_initial::MIGRATION];

// This table must exist before the first migration can be inspected. It is
// deliberately bootstrapped outside the numbered application migrations; the
// rows within it are the authoritative application migration history.
const MIGRATION_HISTORY_SCHEMA: &str = r#"
DEFINE TABLE IF NOT EXISTS schema_migration SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS version ON schema_migration TYPE int;
DEFINE FIELD IF NOT EXISTS name ON schema_migration TYPE string;
DEFINE FIELD IF NOT EXISTS applied_at ON schema_migration TYPE datetime DEFAULT time::now();
DEFINE INDEX IF NOT EXISTS idx_schema_migration_version ON schema_migration FIELDS version UNIQUE;
"#;

#[derive(Debug, Deserialize, SurrealValue)]
struct MigrationRecord {
    version: i64,
    name: String,
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
    let _guard = migration_lock().lock().await;
    ensure_migration_history(db).await?;
    let applied = load_applied_migrations(db).await?;
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
        .query("SELECT version, name FROM schema_migration ORDER BY version ASC")
        .await?
        .take(0)?)
}

fn validate_history(applied: &[MigrationRecord], migrations: &[Migration]) -> Result<()> {
    let latest = migrations
        .last()
        .map(|migration| migration.version)
        .unwrap_or(0);
    let expected: BTreeMap<u32, &str> = migrations
        .iter()
        .map(|migration| (migration.version, migration.name))
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

        let expected_name = expected.get(&version).ok_or_else(|| {
            DbError::MigrationHistory(format!("migration {} is not registered", version))
        })?;
        if record.name != *expected_name {
            return Err(DbError::MigrationHistory(format!(
                "migration {} is named {:?}, expected {:?}",
                version, record.name, expected_name
            )));
        }

        previous = version;
    }

    Ok(())
}

async fn apply_one(db: &DbConnection, migration: Migration) -> Result<()> {
    let definition = db
        .query(migration.sql)
        .await
        .map_err(|error| DbError::MigrationFailed {
            version: migration.version,
            name: migration.name.to_string(),
            reason: error.to_string(),
        })?;
    definition
        .check()
        .map_err(|error| DbError::MigrationFailed {
            version: migration.version,
            name: migration.name.to_string(),
            reason: error.to_string(),
        })?;

    let recorded = db
        .query(
            "INSERT INTO schema_migration (version, name, applied_at) \
             VALUES ($version, $name, time::now())",
        )
        .bind(("version", i64::from(migration.version)))
        .bind(("name", migration.name))
        .await
        .map_err(|error| DbError::MigrationFailed {
            version: migration.version,
            name: migration.name.to_string(),
            reason: error.to_string(),
        })?;
    recorded.check().map_err(|error| DbError::MigrationFailed {
        version: migration.version,
        name: migration.name.to_string(),
        reason: error.to_string(),
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use surrealdb::engine::local::Mem;
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
        assert_eq!(migrations.len(), 1);
        assert_eq!(migrations[0].name, "initial_schema");
    }

    #[tokio::test]
    async fn reapplying_current_schema_is_a_noop() {
        let db = raw_memory_db().await;
        apply_all(&db).await.unwrap();
        apply_all(&db).await.unwrap();

        assert_eq!(applied_migrations(&db).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn future_schema_version_is_rejected() {
        let db = raw_memory_db().await;
        apply_all(&db).await.unwrap();
        db.query(
            "INSERT INTO schema_migration (version, name, applied_at) \
             VALUES (99, 'future_schema', time::now())",
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
            version: 2,
            name: "invalid_test_migration",
            sql: "THIS IS NOT VALID SURREALQL;",
        };

        let error = apply_one(&db, invalid).await.unwrap_err();
        assert!(matches!(error, DbError::MigrationFailed { version: 2, .. }));
        assert_eq!(applied_migrations(&db).await.unwrap().len(), 1);
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
    async fn concurrent_initialization_records_each_migration_once() {
        let db = raw_memory_db().await;
        let first = apply_all(&db);
        let second = apply_all(&db);
        let (first, second) = tokio::join!(first, second);
        first.unwrap();
        second.unwrap();

        assert_eq!(applied_migrations(&db).await.unwrap().len(), 1);
    }
}
