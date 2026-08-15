# CI quality gates

The Rust workflow is deliberately split so a failing PR identifies the command,
fixture, and invariant that need attention without requiring a live inference
service. Normal pull requests run only deterministic offline tests.

| Job | Exact local-equivalent command | What it protects |
| --- | --- | --- |
| `format` | `cargo fmt --all -- --check` | Formatting drift |
| `clippy` | `cargo clippy --workspace --all-targets --all-features --locked -- -D warnings` | Warnings and lint regressions |
| `msrv` | `cargo +1.97.1 check --workspace --locked` | Declared Rust 1.97 MSRV |
| `offline-integration` | `cargo test --workspace --locked` | Unit tests and offline integration with deterministic doubles |
| `persistent-round-trip` | commands shown in `.github/workflows/rust-ci.yml` | Fresh/upgrade migrations, source idempotency, resilient processing, and portable round trips |
| `retrieval-regression` | `cargo test -p graphrag-cli eval::tests::committed_retrieval_fixture_matches_versioned_baseline --bin graphrag -- --exact --nocapture` | Committed retrieval fixture baseline |

Each build job uses the same `quality-gates-v1` Cargo cache key. Compilation is
not merged into one opaque job: a focused failure is more useful to an agent
than a few saved incremental build minutes. The workflow cancels superseded PR
commits, while preserving every main-branch run. The scheduled/manual audit is
the only job allowed to install or query an advisory tool over the network.

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
