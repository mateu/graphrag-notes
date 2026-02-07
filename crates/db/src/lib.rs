//! Database layer for GraphRAG Notes
//!
//! Provides SurrealDB integration with schema management and queries.

pub mod error;
pub mod repository;
pub mod schema;

pub use error::{DbError, Result};
pub use repository::Repository;

use std::path::Path;
use surrealdb::engine::local::{Db, Mem, RocksDb};
use surrealdb::Surreal;

/// Database connection type
pub type DbConnection = Surreal<Db>;

/// Initialize database with RocksDB (persistent)
pub async fn init_persistent(path: impl AsRef<Path>) -> Result<DbConnection> {
    let db = Surreal::new::<RocksDb>(path.as_ref()).await?;
    setup_database(&db).await?;
    Ok(db)
}

/// Initialize database in-memory (for testing)
pub async fn init_memory() -> Result<DbConnection> {
    let db = Surreal::new::<Mem>(()).await?;
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
}
