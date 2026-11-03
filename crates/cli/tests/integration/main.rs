//! All of this crate's integration tests, compiled into one test binary.
//!
//! Each module was a separate `tests/*.rs` binary. Every test binary links
//! SurrealDB and RocksDB, so one binary per crate saves disk and link time.
//! Run one module with: `cargo test -p graphrag-cli --test integration <module>::`.

mod archive_only_commands;
mod capture_commands;
mod connection_review_commands;
mod folder_sync_commands;
mod init_commands;
mod keyword_commands;
mod navigation_commands;
mod remote_candidate_scan_http;
mod remote_capture_http;
mod remote_commands;
mod remote_doctor_http;
mod remote_mutation_http;
mod workspace_commands;

/// Libtest name of `test` in `module` (pass `module_path!()`), for tests that
/// re-run one of their helper tests in a child process with `--exact`.
/// Libtest names omit the crate, so `integration::a::b` runs as `a::b`.
fn exact_test_name(module: &str, test: &str) -> String {
    let module = module.split_once("::").map_or(module, |(_, rest)| rest);
    format!("{module}::{test}")
}
