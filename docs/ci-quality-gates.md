# CI quality gates

The Rust workflow separates focused pre-merge checks from full nightly coverage.
Pull requests and pushes to `main` run format, Clippy, MSRV, persistent round-trip,
and retrieval-regression checks. All five are required on `main`, with strict
up-to-date branch protection and GitHub Actions App binding.

The full `unit and offline integration` job runs nightly at **03:23 UTC** on
`main`, or by manual dispatch. It is no longer a required PR check. This knowingly
moves full workspace unit/offline and combined daily-workflow regression detection
after merge; the five focused checks do not provide equivalent coverage.

| Job | Exact local-equivalent command | What it protects |
| --- | --- | --- |
| `format` | `bash scripts/test-install.sh`, `python3 scripts/test-release.py` and `python3 -m unittest discover -s scripts/tests -p 'test_*.py'`, then `cargo fmt --all -- --check` | Offline installer/setup and packaging behavior, compiled-input identity, and formatting drift |
| `clippy` | `cargo clippy --workspace --all-targets --all-features --locked -- -D warnings` | Warnings and lint regressions |
| `msrv` | `python3 scripts/check-workspace-rust-version.py`, then `cargo +1.97.1 check --workspace --locked` | Exact workspace declaration and Rust 1.97.1 MSRV |
| `offline-integration` (nightly/manual) | `GRAPHRAG_MIGRATION_CONFLICT_PROBE=1 cargo test --workspace --locked`, then `python3 scripts/validate-daily-workflow.py --binary target/debug/graphrag --offline` | Full unit tests, offline integration, graph replay/scale assertions, and the combined daily workflow with deterministic doubles |
| `persistent-round-trip` | commands shown in `.github/workflows/rust-ci.yml` | Fresh/upgrade migrations, source idempotency, resilient processing, and portable round trips |
| `retrieval-regression` | `python3 scripts/run-exact-rust-test.py --package graphrag-cli --bin graphrag --test-name eval::tests::committed_retrieval_fixture_matches_versioned_baseline --nocapture` | Committed retrieval fixture baseline |
| `dependency-audit` (weekly/manual) | `scripts/check-audit-exemptions.sh && cargo audit` | Weekly RustSec scan plus reachability checks for scoped exemptions |

Each build job uses the same `quality-gates-v1` Cargo cache key. Compilation is
not merged into one opaque job: focused failures remain independently visible.
The workflow cancels superseded PR commits while preserving main-branch runs.
Scheduled runs execute only the job belonging to their schedule: full coverage
daily at 03:23 UTC, dependency audit Monday at 04:23 UTC. Manual dispatch runs all
lanes. The scheduled/manual audit is the only job allowed to install or query an
advisory tool over the network.

To run the complete workflow at a selected revision:

```sh
gh workflow run rust-ci.yml --repo mateu/graphrag-notes --ref <branch-or-tag>
```

GitHub scheduled workflows use the default branch, not a PR revision; their
actual start can be delayed. A skipped unit/offline PR job is not a substitute
required gate. The five focused contexts above must remain required.

The repository maintainer owns nightly failures: retain the failed run and tested
SHA, reproduce/localize the defect, then fix it or revert the responsible change.
A passing retry alone does not prove repair, especially for transaction conflicts.

All six focused Rust test invocations use `scripts/run-exact-rust-test.py`.
It first lists the exact target and requires one matching test, then requires
the actual run to report exactly one passed test and no ignored/failed tests.
Cargo's successful zero-test exit is a gate failure. In particular, the embedding
resume fixture uses the full `ingestion::librarian::tests::` module path; the
shorter historical filter selected zero tests. The other focused gates cover
fresh and upgraded migrations, source idempotency, portable backup round trips,
and the committed retrieval baseline. Script fixtures exercise both listing and
execution failures without invoking Cargo.

## Runtime attribution and scheduling decision (#143)

The five historical successful PR jobs in [#143](https://github.com/mateu/graphrag-notes/issues/143)
had a whole-job median of 25m18s. Different revisions and cache states make this
an observational baseline, not a controlled before/after benchmark.

The [20m29s job](https://github.com/mateu/graphrag-notes/actions/runs/37655689771/job/112911624150)
tested merge SHA `6e1058ac0c5529d9e0f930383094d79df6ac6e65` on Ubuntu 24.04.5,
image `20260927.320.1`, Rust 1.97.1, Linux x64. Runner CPU/memory were not recorded
in its log. Its 19m29s Cargo step breaks down as follows:

| Component | Elapsed |
| --- | ---: |
| Compilation including linking | 3m45s |
| Graph latency fixture | 349.50s |
| DB unit-test suite | 257.07s |
| Agents unit-test suite | 107.99s |
| Other suites and residual Cargo overhead | approximately 229.44s |

Checkout/toolchain/cache setup took approximately 28s from runner initialization
to Cargo start; daily-workflow validation took approximately 12s; post-job work
took approximately 18s. Queue time is excluded. Cache restore was a fallback
from `v0-rust-quality-gates-v1-Linux-x64-7f7ca0a9-da41afd2`; the requested key ended
in `7b830d83`. The save lost a reservation race to another job.
The [exact-hit job](https://github.com/mateu/graphrag-notes/actions/runs/37639647591/job/112854887971)
used key `v0-rust-quality-gates-v1-Linux-x64-d0ea4c79-da41afd2` and still reported
4m47s compilation, 462.91s graph execution, and 354.62s DB-suite execution.

Ranked follow-up candidates:

1. Measure graph population/configuration, end-to-end searches, independent
   probes, report generation, and teardown separately. Defaults execute 144
   end-to-end searches plus three rounds of eight probes for each of twelve
   case/population combinations. Every warmed search checks ranking/evidence
   replay equality; probes also assert eligibility and accepted-edge behavior.
   Reducing rounds or removing probes would change correctness coverage.
2. Profile database qualification and fixture setup separately from assertions.
   Preserve both runtime flavors, exact payload/distance comparisons, records
   beyond the 200-ID retained page, and multi-page portable restoration.
3. Investigate cache contents and save ownership across jobs. Shared keys do not
   prove equivalent build artifacts; [rust-cache](https://github.com/Swatinem/rust-cache#cache-details)
   normally retains dependencies, not workspace artifacts, and disables incremental
   compilation. Cache changes need controlled cold/restored measurements.

Nightly routing removes this job from PR feedback, not from execution cost.
No fixture, replay count, timing threshold, compiler cache, or concurrency behavior
changes in this delivery. No execution speedup or total runner-minute reduction
is claimed. Safe fixture/cache optimization remains measurement-gated follow-up
work; the historical baseline does not justify a new runtime budget.

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
