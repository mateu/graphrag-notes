# Daily workflow validation

This guide defines the combined acceptance procedure for
[#64](https://github.com/mateu/graphrag-notes/issues/64). It covers the
[daily workflow](daily-workflow.md) and published Apple Silicon `0.1.0-rc.2`
candidate. It does not replace the historical setup, navigation, keyword,
folder-sync, capture, review, or workspace validation records.

## Run against isolated fictional data

From the candidate checkout, use an already built executable:

```sh
python3 scripts/validate-daily-workflow.py --help
python3 scripts/validate-daily-workflow.py \
  --binary /absolute/path/to/graphrag --offline \
  --report /absolute/path/to/offline-metrics.json
```

Offline is the default. The harness uses a local deterministic HTTP provider
double and public CLI commands. Each run creates a fresh private HOME,
configuration, persistent database, notes, drafts, editor/opener doubles, and
working directory. It never opens personal notes or reads the user's provider
environment overrides. Both successful and failed runs retain their printed
evidence directories, including command output, recovery information, and a
per-binary `metrics.json`. The comparison report goes to `--report`, or to
`comparison.json` in the candidate evidence directory when that flag is omitted.
Remove these synthetic directories and reports manually when finished.

The private TOML uses the documented Ollama settings with `bge-m3:latest` and
`phi4-mini:latest`, `retry_attempts = 1`, `cache_enabled = false`,
`processing_concurrency = 1`, and logging at `info`. Entity extraction remains
enabled. The fixture sets `[gardener] similarity_threshold = 0.0` to ensure the
fictional pair can exercise review, with `auto_apply = false`; this is a test
setting rather than a recommended policy for personal notes. Decisions still
use explicit confirmation. The opener receives a literal executable argument
array. Offline mode points providers at the loopback double; live mode uses
the selected Ollama URL.

Each CLI execution has a 180-second timeout, with a 1800-second deadline per
binary workflow. Adjust these explicitly with `--command-timeout SECONDS`
and `--deadline-seconds SECONDS` for slower local models.

The scenarios connect capture/edit, folder import and refresh, keyword find,
inspection and literal source opening, raw context reuse, governed proposal
accept/undo, and interrupted sync checkpoint/resume. An empty result, stale
selection, unavailable provider, or declined review must remain a truthful
outcome rather than a silent mode change or unconfirmed mutation.

## Compare the published baseline with the candidate

Install `v0.1.0-rc.1` into a separate temporary binary directory using the
[versioned onboarding instructions](getting-started.md#published-onboarding-baseline).
Supply that executable explicitly:

```sh
python3 scripts/validate-daily-workflow.py \
  --binary /absolute/path/to/rc2/graphrag \
  --before-binary /absolute/path/to/rc1/graphrag --offline \
  --report /absolute/path/to/comparison-metrics.json
```

Both sides receive separate fresh databases. The harness probes the older
binary's supported commands and exercises only its available public workflow.
`rc.1` is the published onboarding baseline, not a reconstruction of the
project before U1. Daily-use commands absent from it are reported as unavailable.
Use native release binaries for final before/after release measurements and
record their exact versions, source commits, artifact hashes, and build provenance.

Interpret metrics per task and capability:

| Measurement | Interpretation |
| --- | --- |
| CLI subprocess count | `cli_subprocess_count` counts actual CLI processes, with their success/failure evidence. A total is not comparable when the binaries support different scenario sets. |
| Workspace input count | `workspace_input_count` counts entered workspace commands separately: six lines can run within one CLI process. |
| Elapsed time | Observed local automated wall time with the recorded provider/backend and build profile. It is not a user's task time or a live-model quality metric. |
| Explicit canonical-ID transfers | Places where automation forwards an ID between standalone commands. Workspace selection avoids some such transfers; this is not observed human copying. |
| Recovery outcome | Whether the reported next command recovered the expected state without losing durable content. |
| Human manual-copy count/time | Unmeasured; keep these values null unless a separate human study is performed. |
| Installation execution | Unmeasured by this harness, which receives an already built binary. Validate source setup and published-asset installation separately. |

Do not turn missing baseline capabilities into zero-command successes, compare
unequal overall totals as a speedup, or replace unavailable measurements with
estimates. Deterministic fixture scores establish workflow behavior; live
retrieval ranks and entity counts can vary with model output.

## Live Ollama smoke path

Use an existing local Ollama instance with `bge-m3:latest` (1024 dimensions)
and `phi4-mini:latest`. Follow [getting started](getting-started.md#3-start-ollama-and-download-models)
to install/start providers deliberately before the smoke run. The harness
neither downloads models nor starts services:

```sh
python3 scripts/validate-daily-workflow.py \
  --binary /absolute/path/to/rc2/graphrag --live \
  --ollama-url http://127.0.0.1:11434 \
  --report /absolute/path/to/live-metrics.json
```

Add `--before-binary /absolute/path/to/rc1/graphrag` for an isolated live
baseline comparison. Use only the documented provider URL, runtime settings,
and public CLI commands; no direct SQL or personal corpus is needed. A
deterministic offline pass does not establish live-model success. A live source
run also does not establish installation from a published release asset.

## Historical preparation evidence

The following automated walkthroughs passed on 2026-10-03 UTC (2026-10-02
America/Denver), on macOS 27.0.1 Apple Silicon. Both executables used native
release profiles. These were preparation runs before publication, when the
candidate was available locally. Their original measurements are preserved
below; the subsequent published-asset checks have a separate record.

- Candidate: `graphrag 0.1.0-rc.2`, built from `4028e129829cf7cf225406b787fec03820de07c7`.
  Binary SHA-256: `5c5045d0f6106949bcd367e3de59a594897b0dfb09bc53aee9572937f21390d4`.
  Rust 1.97.1, static OpenSSL, ARM64, declared minimum macOS 15.0,
  only Apple/system shared libraries. Final docs/scripts commits preserve the
  sealed compiled-input identity; final package provenance is in `BUILDINFO.json`.
- Baseline: the checksum-verified published `v0.1.0-rc.1` Apple Silicon asset,
  source tag commit `b25df8d6c907c3f7eb53fb02f81bc7cbb27fde5c`.
  Binary SHA-256: `4c2e8de9366d35890ffce0f2aff5bbf4b9eeb119ecea523e3b676ddd7c169917`.
- Live backend: existing local Ollama 0.35.1, `bge-m3:latest` (1024 dimensions,
  digest `7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab`),
  `phi4-mini:latest` (digest `78fad5d182a7c33065e153a5f8ba210754207ba9d91973f57dffa7f487363753`).
  No model downloads or service startup were performed by the walkthrough.

Run commands were the offline/live commands above with both `--binary` and
`--before-binary`, and fresh explicit `--report` destinations. Local comparison
reports are `/tmp/graphrag-64-offline-native-comparison.json` and
`/tmp/graphrag-64-live-native-comparison.json`; their printed per-run directories
retain command stdout/stderr, drafts and metrics. They contain fictional data.
Metrics are opt-in and are not uploaded by the application.

| Run | Success | Local elapsed seconds | CLI subprocesses | Workspace input lines | Automated canonical-ID transfers |
| --- | --- | ---: | ---: | ---: | ---: |
| Offline rc.2 | Passed | 1.536 | 37 | 6 | 16 |
| Offline rc.1 | Passed | 0.800 | 21 | 0 | 7 |
| Live rc.2 | Passed | 12.930 | 37 | 6 | 16 |
| Live rc.1 | Passed | 6.267 | 21 | 0 | 7 |

These totals cover different capabilities and fixture sizes. They are not a
speedup comparison. Candidate capture runs before baseline capture, and live
model loading/caching can affect elapsed time. Human task time and manual ID
copies remain unmeasured/null. Automated ID forwarding is an explicit
implementation step, not evidence that a person copied an ID.

The before/after walkthrough uses these public operations:

| Task | Published rc.1 | Candidate rc.2 | Offline/live outcome |
| --- | --- | --- | --- |
| Install | Actual HTTPS asset install, one installer invocation, 1.702 seconds excluding script download | Native package and unchanged installer with local transport fixture, one fresh install invocation, 0.288 seconds; published download had not yet run | Both checksum/version passed; transports differ, so elapsed times are not a network speedup comparison |
| Capture/edit | `add`, then `notes list` to obtain the ID; `notes edit/show` | `capture --stdin --format json`, guarded edit/show; recoverable provider-failure draft | Both saved exact edited text; rc.2 replayed its printed Recover command, retained private input, and preserved previous content |
| Folder refresh | Import one file, repeat import, `sources reimport` after change | Register three-file folder, initial/unchanged/changed `sync` | Both refreshed; rc.2 preserved IDs on unchanged sync, made no offline provider requests, refreshed exactly one source, and refused stale inspection |
| Find a source | Hybrid `search`, copy canonical ID to `notes show`; source-open absent | Provider-free keyword search, revision-bound `inspect`, literal-argv `open` | Both located the expected source; rc.2 opened the exact file without shell interpolation |
| Reuse context | Human-formatted `augment` | Cited `augment --raw`, plus workspace `copy` | Both returned cited context; rc.2 raw stdout omitted presentation diagnostics |
| Review a link | Proposal list/show, accept, inspect graph edge, undo | Inbox cards, inspect both endpoints, confirmed accept with reason, audited undo | Both applied and removed the edge; rc.2 retained the original reviewed timestamp and retired the proposal |
| Interrupted work | No folder-sync/resume equivalent | SIGINT during an actual import, inspect durable job, run reported `sync --resume`, inspect completion | rc.2 exited 5 with pending files and a next-action hint, completed the same job, and subsequent sync was unchanged |
| Terminal selection | No U7 selection workspace | Six lines: keyword search, select 1, inspect, open, copy, quit | rc.2 used one CLI process, six supplied workspace lines, and zero canonical-ID transfers |

The local install measurement above records the clean preflight commit. The
final package is regenerated from the final clean commit, installed again, and
sealed with passed source/offline/live/local-install evidence. Its archive hash
and source commit are kept in external `BUILDINFO.json` to avoid a circular
relationship between a tracked hash and the commit containing it. Final and
preflight output directories are never overwritten.

Legacy equivalents have narrower coverage: the baseline folder task imports one
file, whereas the candidate registers three and checks additional guards. The
baseline lacks provider-free keyword/source opening, recoverable capture drafts,
and folder-job interruption/resume; these are reported unavailable, not passed.

| Acceptance item | Status | Evidence |
| --- | --- | --- |
| Combined deterministic workflow and comparison | Passed | Nine candidate scenarios and available baseline equivalents; exact reports and command evidence above. |
| Existing lifecycle, safe CRUD, backup/restore, schema/model compatibility and retrieval checks | Passed | `cargo test --workspace --all-features --locked --offline`: 576 passed, zero failures, five existing ignored tests; 22 test targets. Retrieval fixture/baseline unchanged. |
| Source quality and installer regressions | Passed | Clippy with all targets/features and `-D warnings`, formatting, exact MSRV declaration, and nine existing offline installer checks. Sixteen release checks and nine harness contract tests validate evidence, isolation, bounded failure and no overwrite. |
| Native Apple Silicon candidate build | Passed | Locked release build, version/help, architecture, deployment and system-library checks; sealed build record. Build host 27.0.1 does not establish a live macOS 15 walkthrough. |
| Native package and local asset installation | Passed locally | Preflight package at `75d072a13b4b42f1c59adbe28b16339e9099a990` installed the sealed binary in 0.288 seconds with local transport; overwrite refusal, force reinstall, and edited sample preservation passed. Final source/archive identity is recorded separately in `BUILDINFO.json`. |
| Live Ollama daily workflow | Passed | Nine candidate scenarios and available rc.1 equivalents, including actual provider failure/recovery and SIGINT checkpoint/resume. |
| Published candidate asset install and repeat smoke | Not run during preparation | The completed publication checks are recorded separately below. |
| Linux and Intel macOS native/live acceptance | Deferred to [#66](https://github.com/mateu/graphrag-notes/issues/66) | Offline Linux CI is separate evidence. |
| Network/MCP integration | Deferred to [#56](https://github.com/mateu/graphrag-notes/issues/56) | Local prompt-context reuse does not establish remote access. |

## Published-asset verification

[v0.1.0-rc.2](https://github.com/mateu/graphrag-notes/releases/tag/v0.1.0-rc.2)
was published on **2026-10-03 at 03:57:30 UTC** as an Apple Silicon prerelease.
The tag points to reviewed source commit
`7a8deb859037e396e3314976aa57ea4b58de6e39`, tree
`64b01927f10096e4723eef636a3d19c40b9dcce0`. The binary remains the original
native build from `4028e129829cf7cf225406b787fec03820de07c7`.

Actual HTTPS downloads of the archive, `BUILDINFO.json`, and `SHA256SUMS`
matched the sealed local assets byte for byte. The version-pinned installer
also matched the reviewed tagged source. A fresh install passed checksum,
payload, exact version/help, existing-binary refusal, explicit force reinstall,
and edited-sample preservation checks. Its fresh installer invocation took
**1.313188 seconds**, excluding the earlier installer-script download and
verification setup. The installer ran three times in total: fresh install,
refusal, and force reinstall. This is observed automated installation time,
not a human task time or a network performance comparison.

| Published payload | SHA-256 |
| --- | --- |
| Installed executable | `5c5045d0f6106949bcd367e3de59a594897b0dfb09bc53aee9572937f21390d4` |
| Apple Silicon archive | `e88a426c0f140aa364ff1b56bb5dbc9a30fd77021f6e232894e8255e596064d6` |
| `BUILDINFO.json` | `e51e5cee2fa36c2a9ee304347c32d5f8f0cf662b97d5516dd88723f25784f4c0` |
| `SHA256SUMS` | `304287d87a849f294d5d9790204484c66ba371501e84a99a814578c546e9c91f` |
| Tagged installer | `d10f4f16ff86ab10aaf5ef4b9c86d5ec2db8bf2a9a2983d6f1dd1c133f5f2d43` |

The installed, downloaded executable then passed all nine candidate scenarios
in both offline and live Ollama modes on macOS 27.0.1 Apple Silicon. Each mode
also exercised six available or legacy rc.1 scenarios successfully; four
unsupported capabilities remain explicitly unavailable.

| Downloaded-binary run | Success | Local elapsed seconds | CLI subprocesses | Workspace input lines | Automated canonical-ID transfers |
| --- | --- | ---: | ---: | ---: | ---: |
| Offline rc.2 | Passed | 1.934 | 37 | 6 | 16 |
| Offline rc.1 | Passed | 0.840 | 21 | 0 | 7 |
| Live rc.2 | Passed | 8.851 | 37 | 6 | 16 |
| Live rc.1 | Passed | 5.362 | 21 | 0 | 7 |

These repetitions used fresh fictional corpora and the same documented live
models described above. They confirm the downloaded artifact's workflow and
recovery behavior. Different task coverage and model/cache state still prevent
interpreting total elapsed time as a usability speedup. Human copying and
human elapsed time remain null; six supplied workspace lines are counted
separately from the one workspace process.

Retained operator evidence is under `dist/releases/0.1.0-rc.2-publication/`
in the validation operator's checkout:

- `published-https/published-install-report.json` records the actual HTTPS installation.
- `evidence/published-assets-equality.json` records all three asset comparisons.
- `evidence/published-offline-workflow.json` and `evidence/published-live-workflow.json`
  retain the downloaded-binary comparison reports; their adjacent logs and
  printed private directories retain command output and recovery evidence.

The published tag and assets remain unchanged. Released `BUILDINFO.json`
keeps `published_asset_install.status = "not_run"`, accurately reflecting
its preparation-time state. This later verification is recorded here rather
than replacing sealed metadata or creating a circular hash/commit dependency.
Published installation and both repeat workflows are now **passed**. Native
Linux/Intel acceptance remains in #66, and network/MCP access remains in #56.

Before upgrading an existing corpus, create and verify a portable backup.
The current source applies additive schema migrations when it opens the
database. `rc.1` supports schema 14 and refuses a database upgraded to schema
15. Returning to an older binary requires a compatible pre-upgrade archive
restored into a fresh database; never lower schema metadata with SQL.
