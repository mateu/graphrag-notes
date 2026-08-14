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
    parse_record_id, InferenceCacheEntry, Repository, SourceDeleteSummary, SourceImportAction,
    SourceImportPlan,
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
    setup_database(&db).await?;
    Ok(db)
}

/// Setup database namespace, database, and schema
async fn setup_database(db: &DbConnection) -> Result<()> {
    db.use_ns("graphrag").use_db("notes").await?;
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
