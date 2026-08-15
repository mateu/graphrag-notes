//! Database layer for GraphRAG Notes
//!
//! Provides SurrealDB integration with schema management and queries.

pub mod compatibility;
pub mod error;
pub mod fusion;
pub mod migrations;
pub mod repository;
pub mod schema;

pub use error::{DbError, Result};
pub use repository::{
    parse_record_id, InferenceCacheEntry, ProcessingJob, ProcessingJobStatus, ProcessingJobType,
    ProcessingJobUpdate, Repository, SourceDeleteSummary, SourceImportAction, SourceImportPlan,
    PORTABLE_TABLES,
};

use std::ops::Deref;
#[cfg(feature = "rocksdb")]
use std::path::Path;
use std::sync::Arc;
#[cfg(feature = "rocksdb")]
use surrealdb::engine::local::RocksDb;
use surrealdb::engine::local::{Db, Mem};
use surrealdb::Surreal;
use tokio::sync::Mutex;
use uuid::Uuid;

/// A database client together with process-local coordination scoped to that
/// client identity. Cloning a connection preserves the same coordination;
/// separately opened stores receive independent locks.
#[derive(Clone, Debug)]
pub struct DbConnection {
    client: Surreal<Db>,
    proposal_lifecycle_lock: Arc<Mutex<()>>,
}

impl DbConnection {
    /// Wrap a configured SurrealDB client in the application connection type.
    pub fn new(client: Surreal<Db>) -> Self {
        Self {
            client,
            proposal_lifecycle_lock: Arc::new(Mutex::new(())),
        }
    }

    pub(crate) fn proposal_lifecycle_lock(&self) -> Arc<Mutex<()>> {
        Arc::clone(&self.proposal_lifecycle_lock)
    }
}

impl Deref for DbConnection {
    type Target = Surreal<Db>;

    fn deref(&self) -> &Self::Target {
        &self.client
    }
}

/// Initialize database with RocksDB (persistent)
#[cfg(feature = "rocksdb")]
pub async fn init_persistent(path: impl AsRef<Path>) -> Result<DbConnection> {
    let db = connect_persistent(path).await?;
    setup_database(&db).await?;
    Ok(db)
}

/// Open an existing persistent database without applying application
/// migrations.  Operational diagnostics use this so inspection cannot mutate
/// a user's schema or data.  Opening an absent path is intentionally left to
/// callers: `graphrag doctor` reports that state without creating a store.
#[cfg(feature = "rocksdb")]
pub async fn connect_persistent(path: impl AsRef<Path>) -> Result<DbConnection> {
    let db = DbConnection::new(Surreal::new::<RocksDb>(path.as_ref()).await?);
    db.use_ns("graphrag").use_db("notes").await?;
    Ok(db)
}

/// Initialize database in-memory (for testing)
pub async fn init_memory() -> Result<DbConnection> {
    let db = DbConnection::new(Surreal::new::<Mem>(()).await?);
    // SurrealDB's in-memory engine can be shared by concurrently running
    // clients. Give each test/ephemeral client its own namespace so a
    // transaction in one caller cannot observe or conflict with another
    // caller's fixtures. Persistent stores retain the stable application
    // namespace through `setup_database` below.
    let namespace = format!("graphrag_memory_{}", Uuid::new_v4().simple());
    setup_database_at(&db, &namespace, "notes").await?;
    Ok(db)
}

/// Setup database namespace, database, and schema
async fn setup_database(db: &DbConnection) -> Result<()> {
    setup_database_at(db, "graphrag", "notes").await
}

async fn setup_database_at(db: &DbConnection, namespace: &str, database: &str) -> Result<()> {
    db.use_ns(namespace).use_db(database).await?;
    schema::initialize_schema(db).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_init_memory() {
        let db = init_memory().await.expect("Failed to init memory db");
        // Just verify it connects
        let _: Vec<serde_json::Value> = db.select("note").await.unwrap();
    }

    #[tokio::test]
    async fn memory_clients_use_isolated_namespaces() {
        let first = Repository::new(init_memory().await.unwrap());
        let created = first
            .create_note(graphrag_core::Note::new("isolated memory fixture"))
            .await
            .unwrap();
        let second = Repository::new(init_memory().await.unwrap());
        assert!(second
            .get_note(&graphrag_core::record_id_to_string(
                created.id.as_ref().unwrap(),
            ))
            .await
            .unwrap()
            .is_none());
    }

    #[cfg(feature = "rocksdb")]
    #[tokio::test]
    async fn persistent_database_keeps_schema_version_after_reopen() {
        let temporary_directory = tempfile::tempdir().unwrap();

        let db = init_persistent(temporary_directory.path()).await.unwrap();
        assert_eq!(
            migrations::current_version(&db).await.unwrap(),
            migrations::LATEST_SCHEMA_VERSION
        );
        // The embedded engine owns the RocksDB lock through this client. Drop
        // it before attempting to reopen the same directory.
        drop(db);

        // SurrealDB releases the embedded-engine lock asynchronously after the
        // final client session is dropped.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let reopened = init_persistent(temporary_directory.path()).await.unwrap();
        assert_eq!(
            migrations::current_version(&reopened).await.unwrap(),
            migrations::LATEST_SCHEMA_VERSION
        );
    }
}
