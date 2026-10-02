# First-run validation record

This record belongs to [issue #57](https://github.com/mateu/graphrag-notes/issues/57).
The sample and commands are documented in [getting started](getting-started.md).
Use isolated temporary configuration and database paths for walkthroughs; do
not substitute an existing personal notes database.

## Walkthrough procedure

1. Record the operating system/architecture, CLI version or commit, install
   method, provider version, and model names.
2. From a checkout, run `./setup.sh`; after a binary release is published, also
   exercise `bash scripts/install.sh --version VERSION` with its matching asset.
3. Run `graphrag --config /tmp/graphrag-first-run/config.toml --db-path
   /tmp/graphrag-first-run/data init --backend ollama`. Confirm the preview lists
   those paths and creates neither file nor database.
4. Repeat the same command, including `--config` and `--db-path`, with
   `--write`. Confirm only the configuration is created and contains the
   selected database path; repeat `--write` and confirm existing configuration
   is rejected without changing it.
5. Start Ollama, pull the selected models, then run `graphrag --config
   /tmp/graphrag-first-run/config.toml init --check`. Verify the providers and the
   1024-dimension embedding probe.
6. Run `graphrag --config /tmp/graphrag-first-run/config.toml doctor`. A database
   warning is expected before import.
7. Run `graphrag --config /tmp/graphrag-first-run/config.toml import
   samples/first-notes.md`, then `graphrag --config
   /tmp/graphrag-first-run/config.toml search "What is the Atlas project launch
   plan?" --limit 3`. Confirm a result states the three launch stages.
8. Repeat import and confirm it reports unchanged; run `doctor` again and check
   schema, provider, and embedding compatibility.
9. Run `init` without a preset against the existing configuration. Confirm that
   the file and database remain unchanged; include an environment override and
   verify it is reflected in the preview rather than persisted automatically.
10. Record failure and recovery with a stopped Ollama service and a missing model
    name. Confirm the reported serve/pull commands match the selected provider.

Run steps using a fresh temporary directory of your own if
`/tmp/graphrag-first-run` already exists. Apply the same `--config` flag to every
command. The explicit `--db-path` in step 4 is saved in that configuration, so
later commands keep using the same database through `--config`. Confirm this
with an existing-config `init` preview before continuing. A `GRAPHRAG_DB_PATH`
environment override still takes precedence over the saved path; clear it for
this isolated walkthrough, or pass the same explicit `--db-path` on subsequent
commands. The offline tests use temporary paths and deterministic inference
doubles; they do not establish live-model retrieval quality or a walkthrough
on another platform.

## Evidence

Validation date: October 2, 2026.

| Platform / scenario | Status | Evidence |
| --- | --- | --- |
| Full Rust workspace tests | Passed | On macOS ARM64 with Rust 1.97.1, `cargo test --workspace --all-features --locked` passed 409 tests with 4 ignored, including all 4 init unit tests and all 14 init black-box integration tests. |
| Rust quality checks | Passed | Workspace Clippy with all targets/features and `-D warnings`, plus `cargo fmt --all -- --check`. |
| macOS debug CLI setup walkthrough | Passed | macOS 27.0.1 ARM64, Rust 1.97.1, isolated temporary config/database paths: preview exited 0 and created neither target; `--write` exited 0 and created only config. |
| macOS live missing-model diagnostic | Passed | With local Ollama running, `init --check` exited 2, reported both `bge-m3:latest` and `phi4-mini:latest` absent, printed the corresponding `ollama pull` commands, and left the database absent. Installed local models did not include either required model; no downloads were performed. |
| Deterministic first-result fixture | Passed | A deterministic HTTP provider fixture executes the real CLI: init preview/write/check, config validation, sample import/search/show/reimport, and doctor. It verifies the command/data workflow, config and database safety, and actionable failure states; it does not measure live-model retrieval quality. |
| Documentation and script-help checks | Passed | Relative Markdown links resolve; source/installer script help and debug CLI `init`, `doctor`, `import`, and `search` help match the documented commands; `git diff --check` passes. |
| Offline installer/source setup | Passed | Deterministic platform/download fixtures validate macOS ARM/Intel and Linux x86_64 selection, checksum verification/refusal, explicit overwrite, sample preservation, unsupported/missing releases, archive safety, and optional sccache/missing prerequisites. No network download or native compilation occurs in these tests. |
| Native release workflow builds | Pending | Requires deliberately running the workflow/tag on its macOS and Ubuntu runners; fixture tests do not prove the packaged binary builds. |
| macOS source prerequisite recovery | Passed | Actual `./setup.sh` on macOS 27.0.1 ARM64 stopped because CMake and pkg-config were missing and printed the Homebrew recovery command. No packages or models were installed. |
| macOS source release build and live Ollama import/search | Pending | The prerequisite and debug CLI checks above passed; a completed source release build and import/search with `bge-m3:latest` and `phi4-mini:latest` remain pending. |
| Linux source build and live Ollama import/search | Pending | Requires a Linux host; a release build or mocked provider test alone is not a live walkthrough. |
| Published binary install with checksum verification | Pending | No release assets were published when checked; requires published assets matching the target platform. |

Keep pending items explicit until their walkthrough has been performed. This
record should distinguish live provider behavior from offline tests and build
verification.
