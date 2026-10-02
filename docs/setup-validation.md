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
| Native Apple Silicon release build | Passed | `./setup.sh` built `0.1.0-rc.1` with Rust 1.97.1, `OPENSSL_STATIC=1`, and `MACOSX_DEPLOYMENT_TARGET=15.0`. The ARM64 executable links only Apple/system libraries and declares macOS 15.0 as its minimum. The candidate tag is `b25df8d`; its application sources match the live-tested `d22386b` build. |
| Full release workflow matrix | Deferred | Stable tags build macOS ARM/Intel and Linux x86_64; candidate tags require explicit dispatch after the workflow reaches the default branch. This candidate was built and packaged locally. Intel and Linux native builds have not been validated. |
| macOS source prerequisite recovery | Passed | Initial setup reported missing CMake/pkg-config. Installed only CMake 4.4.3 and pkgconf 3.0.7 with Homebrew updates/upgrades disabled, then built successfully. Existing LLVM/OpenSSL installations and Ollama models were retained. |
| macOS source release and live Ollama walkthrough | Passed | macOS 27.0.1 ARM64, Ollama 0.35.0: all 19 isolated steps passed with downloaded `bge-m3:latest` (1024 dimensions) and `phi4-mini:latest`. Imported four sample chunks with zero failures; the launch plan ranked first among four results; reimport was unchanged; existing-config inspection/refusal, unavailable/missing-model diagnostics, and healthy recovery preserved config bytes and database file sizes/mtimes. |
| Live entity extraction | Passed | `extract-entities --note-id` exercised local phi4-mini generation with both source-built and downloaded binaries. Entity inspection and final doctor completed successfully; the source-built run linked 12 entities. Model-dependent entity counts are not a fixed acceptance assertion. |
| Linux native build and live Ollama import/search | Deferred | Deferred by request while completing macOS first. Linux offline CI is separate evidence and does not establish a live-model walkthrough or published Linux binary. |
| Published Apple Silicon asset installation | Passed | Published [v0.1.0-rc.1](https://github.com/mateu/graphrag-notes/releases/tag/v0.1.0-rc.1) as a prerelease, excluded from latest-stable lookup. Downloaded the tag-pinned installer and installed with explicit `--version 0.1.0-rc.1` into a fresh temporary HOME using only `/usr/bin:/bin:/usr/sbin:/sbin` in PATH. The installer verified the archive checksum; installed binary and sample hashes matched the published `BUILDINFO.json`. Existing-binary refusal and explicit reinstallation preserving edited sample notes passed. |
| Downloaded binary live walkthrough | Passed | Repeated all 19 steps against the installed executable and installed sample: four notes, four results, launch plan at rank 1, unchanged reimport, config/database safety, diagnostics and recovery. Explicit local phi4-mini extraction and final healthy doctor also passed. |

Keep pending items explicit until their walkthrough has been performed. This
record should distinguish live provider behavior from offline tests and build
verification.

## Candidate provenance and CI corrections

The published archive contains the executable and starter Markdown; configuration
is still created explicitly. [BUILDINFO.json](https://github.com/mateu/graphrag-notes/releases/download/v0.1.0-rc.1/BUILDINFO.json)
records the source commit, toolchain, deployment target, and hashes. The executable
SHA-256 is `4c2e8de9366d35890ffce0f2aff5bbf4b9eeb119ecea523e3b676ddd7c169917`.
The macOS 15 minimum comes from the binary's load commands; this walkthrough
ran on macOS 27.0.1, not on macOS 15 or Intel hardware.

The first Linux onboarding CI run aborted during native RocksDB teardown after
a successful import. Normal commands and doctor now return their status through
Tokio runtime shutdown instead of immediately exiting the process. The existing
typed-status test covers doctor exit codes 0/1/2, and [the corrected Linux CI
run](https://github.com/mateu/graphrag-notes/actions/runs/37041287733) passed all
applicable quality, workspace, retrieval, and persistence jobs.

A subsequent macOS offline fixture run exposed incomplete request reads on
accepted sockets inheriting nonblocking mode. The fixture now explicitly uses
blocking I/O and consumes complete HTTP request bodies. All 14 onboarding
integration tests passed after that correction, including four focused
round-trip runs; the final CLI suite passed 96 tests and workspace Clippy
with warnings denied passed. These fixture corrections do not change the
application or published executable.
