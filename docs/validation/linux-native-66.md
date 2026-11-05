# Linux x86_64 source build and offline validation (#66, partial)

Aggregate results only. No note text, credentials or private paths are recorded.
This record covers the **source-build and offline** part of the Linux acceptance
that [#66](https://github.com/mateu/graphrag-notes/issues/66) defers. It does not
close #66. The live-provider and release-packaging steps remain open and are listed
under [Not covered](#not-covered).

## Environment

| Item | Value |
| --- | --- |
| Source | `98fb72f5bc1290edd2f20bfb8290477b774ba697` (main, 2026-11-04) |
| OS | Ubuntu 24.04.5 LTS, kernel 6.18.44, x86_64, glibc 2.39 |
| Resources | 4 CPUs, 15 GiB memory |
| Toolchain | Rust 1.97.1 (pinned by `rust-toolchain.toml`), CMake 3.28.3, Clang 18.1.3, pkg-config 1.8.1 |
| Profile | Debug (`cargo build --workspace --locked`), no sccache |
| Network | Unrestricted outbound HTTPS through a proxy |

`RUSTC_WRAPPER=""` was set for every Cargo command, as CI does. See
[Observations](#observations).

## Results

| Step | Command | Result |
| --- | --- | --- |
| Dependency fetch | `cargo fetch --locked` | Passed |
| Workspace build | `cargo build --workspace --locked` | Passed, 15 min 0 s from a cold target directory |
| Daily workflow, offline | `python3 scripts/validate-daily-workflow.py --binary target/debug/graphrag --offline` | Passed: 37 CLI subprocesses, 6 scripted workspace inputs, 12.9 s |
| Workspace tests | `cargo nextest run --workspace --locked --profile ci` | 1,088 run, 1,088 passed, 25 skipped (ignored), 0 failed, 236 s |
| Doctests | `cargo test --workspace --locked --doc` | Passed (no doctests defined in the six library crates) |

The nextest and doctest commands are the ones `rust-ci.yml` runs on
`ubuntu-latest`. The nextest binary was built with `cargo install cargo-nextest --locked`
because the prebuilt download endpoint returned HTTP 502 during this run.

The skipped tests are the `#[ignore]` tests; they were not run here.

## Not covered

This record is a source-build and offline result. It satisfies none of the #66
acceptance criteria on its own, and a build or offline test on this host must not
be counted as a live walkthrough. Still open from #66, on Linux x86_64:

- **Native release build and packaging.** A release-profile build, `setup.sh`,
  archive packaging, SHA-256 checksums and build provenance.
- **Runtime library inspection.** Linked-library and OS/library compatibility
  evidence for a release binary. No `ldd` or equivalent inspection was recorded.
- **Documented baseline.** Ubuntu 22.04 / glibc 2.35+. This host is Ubuntu 24.04.5
  with glibc 2.39, so it does not show a release binary runs on the baseline.
- **Published-asset installation.** The tag-pinned installer into a fresh HOME with
  Rust absent from `PATH`; archive and installed binary/sample hash checks;
  existing-binary refusal, explicit replacement and edited-sample preservation.
- **Live provider walkthrough.** Real Ollama with `bge-m3:latest` and
  `phi4-mini:latest` and isolated configuration/database paths: init preview, write
  and check, sample import and search (expected Atlas launch stages), unchanged
  reimport, entity extraction, and a healthy final `doctor`.
- **Failure and recovery.** Unavailable-provider and missing-model behavior and
  recovery, with each diagnostic giving the applicable next command.
- **Existing-config safety.** Preview of an existing configuration and refusal to
  overwrite it, plus configuration and database safety checks.
- **Intel macOS.** All of the above on Intel macOS 15+.
- **Other CI lanes.** The `#[ignore]` tests, the exact-test lanes in
  `rust-ci.yml`, and the migration gate.

The offline daily-workflow run above exercises the CLI against a deterministic
provider fixture. It is not evidence for the live-provider items.

## Observations

- A raw `cargo` command fails on a host without `sccache`, because
  `.cargo/config.toml` sets `rustc-wrapper = "sccache"`. The error is
  `could not execute process sccache ... (never executed)`. `setup.sh` and
  CI already override it; `docs/getting-started.md` could say that plain
  `cargo build` needs `RUSTC_WRAPPER=""` when `sccache` is absent.
- The cold debug build dominates: most of the 15 minutes is RocksDB's C++ build
  script, which runs on all four cores.
