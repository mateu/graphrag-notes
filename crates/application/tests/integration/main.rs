//! All of this crate's integration tests, compiled into one test binary.
//!
//! Each module was a separate `tests/*.rs` binary. Every test binary links
//! SurrealDB and RocksDB, so one binary per crate saves disk and link time.
//! Run one module with: `cargo test -p graphrag-application --test integration <module>::`.

mod capture_retrieval_ranking;
mod endpoint_proposals;
mod entity_identity_lifecycle;
mod operations;
mod remote_jobs;
mod remote_mutations;
mod remote_operations;
mod source_policy_migration;
mod vector_source_enrichment;
