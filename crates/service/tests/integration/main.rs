//! All of this crate's integration tests, compiled into one test binary.
//!
//! Each module was a separate `tests/*.rs` binary. Every test binary links
//! SurrealDB and RocksDB, so one binary per crate saves disk and link time.
//! Run one module with: `cargo test -p graphrag-service --test integration <module>::`.

mod alchemist;
mod endpoint_proposals;
mod http_service;
mod importer_adoption_export;
mod remote_mutations;
mod source_enrichment;
mod upload_jobs;
