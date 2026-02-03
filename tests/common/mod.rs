//! Common test utilities

use graphrag_db::{init_memory, Repository};

/// Create a test repository with in-memory database
pub async fn create_test_repo() -> Repository {
    let db = init_memory().await.expect("Failed to create test database");
    Repository::new(db)
}
