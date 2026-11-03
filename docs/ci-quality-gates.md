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

Each build lane has its own Cargo cache key (`check-v1` shared by clippy and
MSRV, then `nextest-v1`, `persistent-round-trip-v1`, `retrieval-regression-v1`,
`migration-regression-v1` and `audit-v1`), saved only from `main`; see
[cache keys and dependency optimization](#cache-keys-and-dependency-optimization).
Compilation is not merged into one opaque job: focused failures remain
independently visible.
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

## Cache keys and dependency optimization

Measured on nightly/manual run
[41564843224](https://github.com/mateu/graphrag-notes/actions/runs/41564843224)
(attempt 2, `84d7d72`): the unit/offline job spent 3m26s compiling 407 crates
after a "full match" restore, then 810s in nextest with all four slots busy for
the whole run (3,237 test-seconds). The top nine tests were 41% of that time;
`graphrag-db` alone was 1,576 test-seconds.

**Cache ownership (follow-up 3 above).** A GitHub cache key cannot be overwritten.
With one `quality-gates-v1` key, the first job to finish on a new key saved it.
In the main push run that created the restored entry
([41559406617](https://github.com/mateu/graphrag-notes/actions/runs/41559406617)),
that was `msrv`, so the cache held `cargo check` metadata only. Clippy and MSRV
reused it, but every lane that runs tests rebuilt every dependency with codegen:
the unit job, plus roughly 2.5–3 minutes in each focused test lane. Each lane
now has its own key; clippy and MSRV share `check-v1` because they need the same
check-only dependency artifacts. Focused `-p` lanes do not share with
`--workspace` because feature unification differs. Only `main` saves
(`save-if`), so PR caches cannot evict main's within the 10 GB repository limit,
and PRs still restore main's entries. The audit lane caches only the registry
and installs the prebuilt `cargo-audit` release via `taiki-e/install-action`
instead of compiling it for about 2.5 minutes.

**Dependency opt-level.** `[profile.dev.package."*"] opt-level = 1` optimizes
third-party crates only (SurrealDB, RocksDB's C++ through `cc`, vector math);
workspace crates stay at opt-level 0. Local measurement (Apple Silicon, 18 cores,
`cargo test -p graphrag-db --lib`, `RUSTC_WRAPPER=""`):

| Measurement | deps opt 0 | deps opt 1 | deps opt 2 |
| --- | ---: | ---: | ---: |
| `native_float_all_scopes_preserve_exact_payload_keys_and_distance_bits` | 65.3s | 11.2s | 12.7s |
| `lexical_migration_backfills_all_rows_and_portable_restore_rebuilds_exact_indexes` | 40.5s | 8.1s | 7.2s |
| `graph_alias_fast_path_preserves_high_degree_pages_tiers_and_alias_caps` | 23.0s | 3.2s | 2.8s |
| Whole `graphrag-db` lib suite, `--test-threads=4` | 116s | 24s | — |
| Cold `--no-run` build (wall / CPU) | 88s / 10 min | 160s / 23 min | 197s / 34 min |

Opt-level 2 was no faster than 1 and costs about 50% more dependency compile
CPU, so 1 is used. No fixture, replay count, assertion or concurrency setting
changed.

CI measurement, cold manual run on the PR branch
([41582026970](https://github.com/mateu/graphrag-notes/actions/runs/41582026970),
no cache for the new keys): nextest dropped from **810s to 214s** (3,237 to about
830 test-seconds; the slowest test went from 233s to 47s). Compilation took
19m01s cold. About 11 minutes of that is the RocksDB C++ build script
(`surrealdb-librocksdb-sys`), whose output the shared `cargo check` cache used
to provide. It is not caused by the opt-level change: locally with `-j 4` the
build script took 111s with that crate at opt-level 0 and 119s at opt-level 1.
rust-cache falls back to the lane's previous entry when only `Cargo.lock`
changes, so the cold cost recurs only after a toolchain change, a key bump, a
dependency change to RocksDB itself, or eviction (seven days without access or
the 10 GB repository limit). Until
main has saved the new keys, PR runs on these lanes are cold.

## Audit exemption policy

`.cargo/audit.toml` contains three narrowly scoped advisory exemptions that are
present in `Cargo.lock` but unreachable from the supported native dependency
graph:

- `RUSTSEC-2026-0235`: `rkyv` is an optional `rust_decimal` dependency, and the
  archive feature is not enabled.
- `RUSTSEC-2023-0071`: `rsa` is selected by SurrealDB's WebAssembly
  `jsonwebtoken` path. Native builds select `aws-lc-rs`, and the advisory has no
  fixed `rsa` release.
- `RUSTSEC-2026-0387`: `crypto-bigint` 0.5.5 is pulled in by
  `elliptic-curve` 0.13 through `p256`/`p384` and `jsonwebtoken`'s
  WebAssembly-only `rust_crypto` backend. Native builds use `aws-lc-rs`.
  The fixed `crypto-bigint` 0.7.6 is not compatible with that dependency
  requirement. As checked on 2026-10-26, SurrealDB core 3.2.6 and 3.3.4
  still require `jsonwebtoken` 10; even `jsonwebtoken` 11.1.0 still uses
  `p256`/`p384` 0.13. A forced major-version override is not a valid fix.
  This exemption does not qualify WebAssembly builds as secure.

Before `cargo audit`, CI runs `scripts/check-audit-exemptions.sh` against the
native Linux target. The job fails if either exempted crate becomes reachable,
forcing the exemption to be removed or reassessed instead of silently masking
a production dependency. Revisit these exemptions whenever `surrealdb`,
`surrealdb-types`, `rust_decimal`, `jsonwebtoken`, or its RustCrypto backend
changes. Remove the crypto-bigint exemption once upstream adopts a fixed
compatible stack; do not enable that backend on native builds without
reassessing it.

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

## Hot retrieval plans and conditional migrations (#179)

The `retrieval regression baseline` PR/main lane additionally runs the named,
non-ignored plan fixtures listed in [the hot-statement inventory](hot-retrieval-plans.md).
These are provider-free generated databases, not copies of a live database.
Plan checks guard access paths, not latency or relevance. Existing baseline
assertions remain unchanged. This is a focused extension of #143, not a return
of full unit/offline coverage to every PR.

To add a hot statement, identify the SQL actually issued by the search service,
add it to the inventory with its scan population/index, then add an EXPLAIN FULL
assertion using the existing operator walker and wire its exact test name into
`rust-ci.yml` with `scripts/run-exact-rust-test.py`. Explain each nested/correlated
statement separately with representative bindings: an indexed outer plan does
not establish that an inner lookup is indexed. Record any engine visibility
limitation and the concrete alternative guard; never infer an index hidden by
DynamicScan. Keep the de-indexed negative controls proving the guards reject
historical scan shapes. Query-behavior fixes belong in separate issues.

`conditional migration and schema fixtures` starts on every PR and main push.
It selects changed paths between the PR base/head (or the push before/head),
using a full checkout history. The covered paths are:

- `crates/db/src/migrations/**`, including the registry-derived historical fixture
  helper in `crates/db/src/migrations/mod.rs` from #182;
- `crates/db/src/schema.rs`;
- `.github/workflows/rust-ci.yml` and `scripts/run-migration-gate.py`, so edits to
  the gate itself are qualified too.

A matching change runs every discovered `migrations::tests::*` fixture, the
`lexical_migration_backfills_all_rows_and_portable_restore_rebuilds_exact_indexes`
fixture, and these CLI backup schema fixtures:

- `schema_fifteen_v1_backup_remains_restorable_without_capture_table`;
- `empty_backup_round_trip_is_valid_and_restores_a_fresh_schema`;
- `schema23_reviewed_enrichment_backup_reopens_and_recovers_exact_leased_checkpoint`;
- `typed_relation_evidence_survives_archives_replay_and_undo` (schema 28 typed
  evidence through native and JSONL archives, receipt replay and undo).

Discovery and execution fail closed on missing, failed, ignored or zero-test
fixtures. The ignored schema23 subprocess worker is launched only by its parent
qualification. A non-matching change produces a successful job without toolchain
setup, so making this context required cannot leave a path-filtered check pending.
Manual dispatch and initial pushes without a usable previous SHA select all.
Nightly routing is unchanged.

Owner proposal (not applied by this PR): retain the five existing required
contexts and add `conditional migration and schema fixtures`, bound to GitHub
Actions app ID `15368`, preserving `strict: true`. Plan checks stay in the
already-required `retrieval regression baseline` context. Repository settings
remain an owner action; workflow publication does not change branch protection.

### Current qualification limit

The #179 inventory explicitly distinguishes indexed-path gates from existing
scan gaps and incomplete paths. Exact-vector and provenance statements already
full-scan at the unchanged base. Therefore the current subset guards do not
satisfy universal AC-1; this delivery is **not delivery-ready** until that scope
conflict is resolved. The ignored scan diagnostic is evidence, not a passing
plan-regression test. See the inventory for each status.

## Exact-vector plan exception (#191)

The retrieval lane also runs the named non-ignored
`repository::exact_vector_plan_policy_tests::exact_vector_scan_exception_preserves_plans_and_filtered_top_k`
guard on every PR/main push. [The explicit policy](exact-vector-plan-policy.md)
allows only the three exact-vector table scans; it pins their known
KnnTopK/TableScan shapes and independently verifies filtered exact results.
It must not be reported as indexed/no-scan coverage. A changed access path
requires exactness qualification and policy review. Other hot paths retain
the indexed/no-scan requirement.

## Exact-head Review gate (#225)

`Review gate` is a separate GitHub Actions check on the PR head. Drafts and new
heads are pending. Trusted `chatgpt-codex-connector[bot]` completed state or clean
receipt comments must identify that head; old reviews and forged-author comments
never satisfy it. Completed unresolved findings and unresolved review threads on
that head fail the gate. Conflicting completed evidence fails closed.

With no completed Codex verdict, an independent local review can qualify after
20 minutes from the latest commit committer date or latest ready-for-review event,
whichever is later. Repository variable `REVIEW_GATE_FALLBACK_MINUTES` overrides
20; `REVIEW_GATE_ATTESTERS` is a comma-separated author allowlist (default repository
owner `mateu`). Only trusted attesters may post the following, after obtaining an
independent review (the implementer's own review is not sufficient):

```text
<!-- local-review:v1 {"commit_sha":"<full head SHA>","verdict":"clean","reviewer":"<independent reviewer id>"} -->
```

Helper (uses your authenticated `gh` identity):

```sh
python3 scripts/attest-local-review.py <PR> --commit-sha <reviewed-full-SHA> --reviewer <session-or-reviewer-id> --verdict clean
```

A non-clean attestation fails; stale or untrusted attestations are ignored. A
completed Codex verdict takes precedence over local fallback. New commits require
new review evidence. The schedule polls every five minutes (GitHub may delay it).
To recover from delayed events or API failures, inspect the evaluator workflow
logs and re-dispatch on **main**:

```sh
gh workflow run review-gate.yml --repo mateu/graphrag-notes --ref main
```

Workflow code is always checked out from the default branch, never a PR ref,
including for fork PRs. The token grants only checks write, PR read, and issues
read. No secrets or automatic review requests are used. Review threads and
comments are paginated; excessively large individual threads fail closed.
The evaluator is intentionally unavailable before its first default-branch merge;
PR-version execution would violate this trust boundary. Recorded PR #188 fixtures
qualify policy transitions offline but are not a live GitHub check proof.

**Post-merge maintainer step (Hunter/Forge):** retain all existing contexts and
add `Review gate`, GitHub Actions app ID **15368**, to main's required status
checks, preserving strict up-to-date checks. Verify both live paths (Codex clean
and independent fallback) and the required-context binding before closing #225.
Publishing this workflow does not itself change branch protection.

### Accepted trust boundary and soft guard

Required checks bind only to the GitHub Actions app and check name, not the
publishing workflow. A workflow merged or pushed into a non-fork PR could publish
a fake `Review gate` success, including on the test-merge commit (whose checks
can take precedence over head checks). Hunter explicitly accepted this residual
limit: this gate guards accidental/premature merges, such as auto-merge before
review; it does **not** protect against a malicious insider.

Fork PR workflows cannot write checks with their `GITHUB_TOKEN`; fork workflow
approval is configured for all external contributors. The trusted target evaluator
can publish checks but never runs fork code.

The required `format` job runs `python3 scripts/check-review-gate-name.py` and
rejects the literal `Review gate` anywhere in other `.yml`/`.yaml` workflow files
(including job names, check payloads, status contexts and comments). The evaluator
adds a visible warning to its check summary when the PR changes or renames its
workflow or `scripts/review-gate.py`. Such changes require explicit independent
inspection, not a title/label exception. Tests cover both guards. These are soft
guards: dynamic names, indirect publishers, or changes to the guard can evade
them. They do not establish workflow provenance or prevent deliberate bypass.

Hard-guarantee upgrades remain future work: publish through a dedicated GitHub
App with its own required `integration_id`, or move this user-owned repository
into an organization and use a ruleset `workflows` rule. That rule is org-only
and unavailable here (the maintainer verified a 422 response).

### Narrow dependency-bot exemption (#234)

Before requiring review evidence, the evaluator checks GitHub REST PR, commits,
and files data for the **current head**. Exempt success reads
`Exempt: bot dependency update (author, commits, paths, non-major verified)`.
Drafts stay pending; exact-head unresolved Codex findings/threads still fail.
Otherwise the normal review path runs with the first `Not exempt` reason.

`scripts/review-gate-bots.json` is the reviewed configuration: `bots` maps exact
GitHub logins to accepted commit emails, and `paths` lists manifest/lockfile
shell globs (case-sensitive; `*` may cross directory separators). Extend it only
through independent review, with identity/path regression fixtures. Both author
and committer must have allowlisted GitHub-attributed logins AND matching emails;
missing attribution, mixed human commits, incomplete API lists and stale heads
fail closed. In particular GitHub's `web-flow` committer is **not** allowlisted.
There is no workflow-input change or new repository-variable wiring in this PR.

Hidden paths, `scripts` and `src` components are always excluded; Dockerfiles,
Renovate config and all other non-allowlisted files are excluded. Renames check
both names. Numeric `from X.Y.Z to A.B.C` title evidence must show the same major;
explicit major signals reject. Unknown/grouped/unparseable update types receive
normal review. Available diff version-major multisets must also agree; missing
patches and ambiguous additions/removals receive normal review. This deliberately
conservative rule may decline harmless lockfile churn. Labels/body cannot grant
exemption. Security updates use exactly the same rules, including major rejection.

Recorded REST fixture `dependabot224.json` captures real PR #224: Dependabot
attributed author, **web-flow committer**, therefore non-exempt. No Renovate PR
was returned by the repository's app-author history query. Positive Dependabot
and Renovate fixtures are explicitly synthetic mutations, not live bot proof.
Pre-merge tests exercise clean manifest/lockfile updates, human/committer identity,
workflow/rename exclusions, major/unknown versions, spoofing, new heads, config,
and unresolved findings. No bot credentials are available to create a genuinely
bot-authored/committed throwaway PR, and default-branch evaluator code cannot run
this exemption live before merge. After merge, inspect the check on a real bot-only
non-major PR, then verify a human-commit/path-change variant returns to review
and exact-head unresolved findings still fail. Close/delete only proof PRs and
branches created for that verification. Requiring `Review gate` remains a
**maintainer follow-up after merge and real-bot verification**, not this change.

Residual risk: a compromised allowed bot or malicious dependency version can
still qualify; other CI, audits and CodeQL—not this exemption—must catch it.
The evaluator continues to run default-branch code, never PR-head code.
