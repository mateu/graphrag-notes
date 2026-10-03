# Uploaded source and durable job validation

Validated on Apple Silicon macOS on 2026-10-03 using isolated in-memory or disposable persistent databases, synthetic Markdown, deterministic providers, private credential files, and loopback HTTP. The earlier integrated code checkpoint `a1bcef1` includes foundation `7de4da` and remote mutations `05f87d1`, plus both upload/job review rounds. The historical review-fix checkpoint `d4f541e` includes foundation `cb847712` and remote mutations `1a9b049`. The preceding combined checkpoint used `c862de6`/`29ed6ee`; the earlier standalone #77 gate used foundation `5675096`. Published rc.2 does not contain these commands.

## Automated acceptance

The Codex-reviewed Rust checkpoint `d146db2` passed **705 tests**, with zero failures and five existing provider tests ignored across 26 nonempty targets (33 including empty and documentation targets). Workspace/all-target/all-feature Clippy with warnings denied, formatting, whitespace checks and a fresh all-feature CLI build passed. Its binary SHA-256 is `896e081973977236150240aa9de21ff8e4648ebd07647772a9ef8a974ee86d6d`. The #78 harness checkpoint `6217f07` has identical Rust sources/manifests and passed all 44 Python checks on macOS, with one Linux-only check skipped; that native target also passed all 26 checks on clawd Linux.

This checkpoint includes final-content draft pinning for capture/edit/upload, a real held-admission HTTP regression that preserves later draft edits, portable path-like uploaded-title preservation across archive/JSONL round trips, durable malformed-input quarantine, and the standalone foundation's bounded body-ingress/provenance credential checks. These are code changes after the 697-test checkpoint; the historical binary and counts below do not identify the current candidate.

The earlier integrated workspace/all-feature run passed **697 tests**, with zero failures and five existing ignored tests across 25 nonempty targets (32 including empty and documentation targets). All-target/all-feature Clippy with warnings denied, formatting, whitespace checks, all 18 Python helper tests, and a fresh all-feature CLI build passed. The executable SHA-256 is `ff64eeddbfaa47ba6c5181876a5e661bcb9d5d2689817352fc62a2cf24590362`.

The subsequent stack rebase onto foundation `44b1d4` / mutations `67794d1` moves the same capture-recovery helper into the foundation PR. At that rebase checkpoint, all crate sources, dependency manifests/lockfile, and scripts were byte-identical to the recorded candidate. The differences are a separately passing standalone capture HTTP regression and foundation documentation. The 697-test count continues to identify its recorded checkpoint.

The historical `d4f541e` gate passed 681 Rust tests and 15 Python tests. The preceding combined gate passed 667 Rust tests; the earlier standalone gate passed 638 across 22 nonempty targets. All these Rust runs had the same five existing ignored tests. These historical counts describe their own checkpoints, rather than later additions to the suite.

The installed OpenClaw 2026.9.7 and Hermes 0.21.2 runtimes also exercised capture, revision-checked edit/delete, upload admission, cancellation/resume, owner isolation, cross-client reads, and replay after service restart against this combined candidate. That smoke uses one Mac and a synthetic corpus; its reproducible harness and sanitized evidence are tracked in #78.

Nineteen dedicated database regressions cover principal-scoped admission retries/conflicts, source identity, lease/CAS ownership, cancellation outside the transition gate, restart reconciliation, bounded inputs, hidden staging, active versus pending provenance, unchanged title/configuration policy, extraction checkpoints, promotion/reconciliation retries, accepted-edge undo, portable input preservation, generic local-job bypass refusal, deletion/recreation of the same source identity, older unprepared admissions after a newer generation, and cancellation while terminalization waits for the lifecycle gate. Malformed queued input is durably quarantined as failed/validation before acquiring a lease. A bounded FIFO scan continues to healthy following jobs; repaired input requires explicit resume in the same process. Storage failures remain visible rather than being classified as invalid input; post-claim payload corruption settles through minimal authority without allowing an obsolete worker to finish a new owner's job.

Eight application regressions cover provider failure then resume, exact request replay, independent instance ownership, source refresh and unchanged runs, cancellation of blocked preparation, stale worker epochs, extraction failure after a committed checkpoint, incompatible resume, and the 200-chunk admission limit before providers. A CRLF-to-LF unchanged refresh returns the latest exact Markdown and provenance with a new inspection revision while preserving the backing chunk text, byte spans, note IDs, and generation.

Real HTTP service fixtures verify that jobs outlive the admitting HTTP action, caller scope and capability checks reject spoofed or host-path input, multibyte byte bounds hold, cancellation remains available during blocked upload and capture preparation, and HTTP resume completes the saved job. A separate transport fixture aborts a genuinely pending upload waiter and confirms accepted admission drains during shutdown. A blocked-worker fixture shuts down the actual worker pool, observes a durable interrupted checkpoint, starts a new HTTP service, and resumes that saved input to completion.

At ordinary HTTP capacity limits one and two, every ordinary slot is occupied by blocked capture preparation; a fresh authenticated MCP client still initializes, discovers tools, and durably cancels an owned upload. Invalid credentials remain rejected. Another actual HTTP fixture disconnects a blocked admission waiter and confirms its detached task retains the admission slot, rejects 20 excess admissions, and leaves ordinary reads and bounded cancellation reachable.

The CLI fixtures retain a private upload draft and replay command on failure, preserve option-like metadata/request IDs and the resolved absolute draft directory, and reject local-only upload before opening a corpus. Admission response validation prevents incompatible or mismatched acknowledgements from deleting the draft. A real CLI/HTTP recovery regression executes the copied admission command from a different working directory using `--recover-draft`, removes its retained recovery file, preserves the ordinary supplied input, and creates no draft directory in the retry's working directory. The same fixture also covers actual lost acknowledgements and copied recovery commands for capture and edit.

A service worker regression injects actual stored authority-decode and terminal-write failures, verifies fenced recovery remains tracked, repairs the fault, and resumes the job in the same process. A separate worker-pool regression leaves terminal storage faulty, confirms graceful shutdown returns without dropping a write in progress, repairs the fault, reconciles the durable lease in a new pool, and resumes the original job to completion.

The real portable archive regression completes an upload, refreshes it and prunes its original chunks, verifies/restores the archive, and replays the exact admission and stored outcome. A second real-application regression preserves distinct active and pending processing snapshots through vector-inclusive and vectorless archives and JSONL exports, including their embedding configuration objects. Caller file-URI provenance and ordinary metadata words such as `token` and `embedding` remain opaque user content; ordinary host paths and secrets are stripped. Six rewritten archives containing malformed or wrong-table historical item/checkpoint IDs are rejected by both verify and restore before a target database is created.

## Reproduction

```sh
CARGO_BUILD_JOBS=4 RUSTC_WRAPPER='' cargo test --workspace --all-features --locked --offline
CARGO_BUILD_JOBS=4 RUSTC_WRAPPER='' cargo clippy --workspace --all-targets --all-features --locked --offline -- -D warnings
cargo fmt --all -- --check
git diff --check
```

Retained local evidence:

- `/tmp/graphrag-56-codex-combined-tests.log` — 705-test Codex-reviewed combined Rust checkpoint.
- `/tmp/graphrag-56-codex-combined-clippy.log` — combined workspace Clippy.
- `/tmp/graphrag-56-codex-combined-fmt.log` — formatting.
- `/tmp/graphrag-56-codex-combined-build.log` — fresh all-feature build.
- `/tmp/graphrag-56-codex-combined-python.log` — 44 Python checks (one Linux-only skipped on macOS).

- `/tmp/graphrag-81-final-integrated-tests.log` — 697-test final integrated suite at `a1bcef1`.
- `/tmp/graphrag-81-final-integrated-clippy.log` — final workspace/all-target/all-feature Clippy.
- `/tmp/graphrag-81-final-integrated-fmt.log` — final formatting check.
- `/tmp/graphrag-81-final-integrated-python.log` — 18 helper tests before the separate native harness is added.
- `/tmp/graphrag-81-final-integrated-build.log` — final all-feature CLI build.
- `/tmp/graphrag-81-absolute-upload-tests.log` — four real CLI/HTTP recovery checks, including copied upload recovery from another directory.
- `/tmp/graphrag-81-second-db-tests.log` — 18 dedicated job database checks after the second review.
- `/tmp/graphrag-81-second-service-tests.log` — 58 application/service checks including admission capacity, reserved cancellation, and faulted shutdown.
- `/tmp/graphrag-81-review-workspace-tests.log` — 681-test review-fix full suite.
- `/tmp/graphrag-81-review-workspace-clippy.log` — review-fix workspace Clippy.
- `/tmp/graphrag-81-review-runtime-tests.log` — focused admission fencing/cancellation checks.
- `/tmp/graphrag-81-review-service-tests.log` — storage-fault recovery worker checks.
- `/tmp/graphrag-81-review-backup-tests.log` — active/pending snapshot backup checks.
- `/tmp/graphrag-81-recovery-http-tests.log` — four real CLI/HTTP recovery checks.
- `/tmp/graphrag-77-integrated-tests.log` — preceding 667-test combined full suite.
- `/tmp/graphrag-77-integrated-clippy.log` — preceding combined workspace Clippy.
- `/tmp/graphrag-77-workspace-tests.log` — earlier standalone full suite.
- `/tmp/graphrag-77-clippy-final.log` — final all-target/all-feature Clippy.
- `/tmp/graphrag-77-db-tests-final.log` — 145 database library tests, including the 14 upload regressions.
- `/tmp/graphrag-77-application-service-tests-final.log` — seven application job regressions and the preceding service run.
- `/tmp/graphrag-77-transport-cli-tests-final.log` — strengthened lost-ack/shutdown/restart HTTP fixtures plus CLI tests; the final full suite additionally includes the rewritten malformed archives.

The two-computer OpenClaw/Hermes deployment walkthrough is tracked by #78. These local fixtures exercise the native server/client boundaries without attributing them to external client sessions or a native Linux deployment.
