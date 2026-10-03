# CI quality gates

The Rust workflow is deliberately split so a failing PR identifies the command,
fixture, and invariant that need attention without requiring a live inference
service. Normal pull requests run only deterministic offline tests.

| Job | Exact local-equivalent command | What it protects |
| --- | --- | --- |
| `format` | `bash scripts/test-install.sh`, `python3 scripts/test-release.py` and `python3 -m unittest discover -s scripts/tests -p 'test_*.py'`, then `cargo fmt --all -- --check` | Offline installer/setup and packaging behavior, compiled-input identity, and formatting drift |
| `clippy` | `cargo clippy --workspace --all-targets --all-features --locked -- -D warnings` | Warnings and lint regressions |
| `msrv` | `python3 scripts/check-workspace-rust-version.py`, then `cargo +1.97.1 check --workspace --locked` | Exact workspace declaration and Rust 1.97.1 MSRV |
| `offline-integration` | `cargo test --workspace --locked`, then `python3 scripts/validate-daily-workflow.py --binary target/debug/graphrag --offline` | Unit tests, offline integration, and the combined daily workflow with deterministic doubles |
| `persistent-round-trip` | commands shown in `.github/workflows/rust-ci.yml` | Fresh/upgrade migrations, source idempotency, resilient processing, and portable round trips |
| `retrieval-regression` | `cargo test -p graphrag-cli eval::tests::committed_retrieval_fixture_matches_versioned_baseline --bin graphrag -- --exact --nocapture` | Committed retrieval fixture baseline |
| `dependency-audit` | `scripts/check-audit-exemptions.sh && cargo audit` | Scheduled RustSec scan plus reachability checks for scoped exemptions |

Each build job uses the same `quality-gates-v1` Cargo cache key. Compilation is
not merged into one opaque job: a focused failure is more useful to an agent
than a few saved incremental build minutes. The workflow cancels superseded PR
commits, while preserving every main-branch run. The scheduled/manual audit is
the only job allowed to install or query an advisory tool over the network.

## Audit exemption policy

`.cargo/audit.toml` contains two narrowly scoped advisory exemptions that are
present in `Cargo.lock` but unreachable from the supported native dependency
graph:

- `RUSTSEC-2026-0235`: `rkyv` is an optional `rust_decimal` dependency, and the
  archive feature is not enabled.
- `RUSTSEC-2023-0071`: `rsa` is selected by SurrealDB's WebAssembly
  `jsonwebtoken` path. Native builds select `aws-lc-rs`, and the advisory has no
  fixed `rsa` release.

Before `cargo audit`, CI runs `scripts/check-audit-exemptions.sh` against the
native Linux target. The job fails if either exempted crate becomes reachable,
forcing the exemption to be removed or reassessed instead of silently masking
a production dependency. Revisit these exemptions whenever `surrealdb`,
`surrealdb-types`, `rust_decimal`, or `jsonwebtoken` changes.

## Retrieval baseline policy

`tests/baselines/retrieval-v1.json` is an ordinary, versioned JSON file. The
offline test seeds `retrieval-regression-cases-v1.jsonl` into an in-memory
database, uses deterministic embeddings, then runs the real `SearchAgent`
fusion/filtering and context-packing path. It cannot download a model or
depend on a local database. The test prints every baseline/current metric delta
and uses these strict v1 thresholds:

| Metric | Maximum allowed drop |
| --- | ---: |
| Recall@k | 0.00 |
| Precision@k | 0.00 |
| MRR | 0.00 |
| nDCG@k | 0.00 |
| Provenance accuracy | 0.00 |

To propose a replacement, first generate a report from the deterministic
fixture harness (it must carry `provider: fixture` and
`model: deterministic-stack-v1`), then review it:

```bash
make retrieval-fixture-report OUT=/path/to/eval-report.json
make update-baseline CANDIDATE=/path/to/eval-report.json
# After reviewing the printed diff in an interactive terminal:
make update-baseline CANDIDATE=/path/to/eval-report.json APPLY=1
```

The fixture-report command refuses to overwrite an existing file. The baseline
command never changes a baseline by default; `APPLY=1` requires typing
`UPDATE`. It also rejects reports not produced by that fixture. CI never
blesses or updates baselines.

## Test concurrency and live services

The old global `--test-threads=1` restriction is gone. The migration module
uses its own scoped lock because it deliberately initializes the same in-memory
schema from concurrent test tasks. Other tests must use isolated fixtures or a
similarly narrow lock with a documented shared resource; do not reintroduce a
workspace-wide serial test flag. TEI, TGI, and Ollama smoke testing remains a
manual or scheduled concern, never a pull-request gate.

## Combined workflow and release acceptance

The [daily workflow procedure](daily-workflow-validation.md) exercises public
commands against a fresh fictional corpus. Use an already built binary for
the deterministic run:

```sh
python3 scripts/validate-daily-workflow.py \
  --binary /absolute/path/to/graphrag --offline \
  --report /absolute/path/to/workflow-metrics.json
```

The harness uses private configuration/database/draft paths and a loopback
provider double. It records actual command results, recovery outcomes,
automated canonical-ID transfers, and local elapsed time. CLI subprocesses and
entered workspace commands are counted separately. Before/after totals
are not comparable when supported task sets differ; report per-task coverage
and keep unobserved human manual-copy count/time null. Supplying a live endpoint
does not turn a deterministic pass into live-provider evidence.

Live smoke runs use explicit `--live --ollama-url URL` with existing Ollama
models; they are opt-in and separate from ordinary offline gates. A native
release build, local package validation, live source walkthrough, and install
from a published asset are separate checks. Record each result with its binary
and source provenance in [daily workflow validation](daily-workflow-validation.md).
The [rc.2 release guide](releases/0.1.0-rc.2.md) defines packaging, checksums,
compiled-input identity, and published version-pinned installation.

The candidate acceptance platform is macOS Apple Silicon (macOS 15+ deployment
target). Native Linux and Intel macOS asset/live acceptance remains in
[#66](https://github.com/mateu/graphrag-notes/issues/66); an Ubuntu offline CI pass
does not satisfy those checks. `v0.1.0-rc.1` remains historical onboarding
evidence. Existing feature validation records are retained with their original
versions, dates, and limitations.
