# Uploaded source and durable job validation

Validated on Apple Silicon macOS on 2026-10-03 using isolated in-memory or disposable persistent databases, synthetic Markdown, deterministic providers, private credential files, and loopback HTTP. The review-fix checkpoint `d4f541e` includes foundation `cb847712` and remote mutations `1a9b049`. The preceding combined checkpoint used `c862de6`/`29ed6ee`; the earlier standalone #77 gate used foundation `5675096`. Published rc.2 does not contain these commands.

## Automated acceptance

The review-fix workspace/all-feature run passed **681 tests**, with zero failures and five existing ignored tests across 25 nonempty targets. All-target/all-feature Clippy with warnings denied, formatting, whitespace checks, and all 15 Python tests (including six credential-helper tests) passed. The preceding combined gate passed 667 tests; the earlier standalone gate passed 638 tests across 22 nonempty targets with the same five ignored tests.

The installed OpenClaw 2026.9.7 and Hermes 0.21.2 runtimes also exercised capture, revision-checked edit/delete, upload admission, cancellation/resume, owner isolation, cross-client reads, and replay after service restart against this combined candidate. That smoke uses one Mac and a synthetic corpus; its reproducible harness and sanitized evidence are tracked in #78.

Sixteen dedicated database regressions cover principal-scoped admission retries/conflicts, source identity, lease/CAS ownership, cancellation outside the transition gate, restart reconciliation, bounded inputs, hidden staging, active versus pending provenance, unchanged title/configuration policy, extraction checkpoints, promotion/reconciliation retries, accepted-edge undo, portable input preservation, generic local-job bypass refusal, deletion/recreation of the same source identity, older unprepared admissions after a newer generation, and cancellation while terminalization waits for the lifecycle gate.

Seven application regressions cover provider failure then resume, exact request replay, independent instance ownership, source refresh and unchanged runs, cancellation of blocked preparation, stale worker epochs, extraction failure after a committed checkpoint, incompatible resume, and the 200-chunk admission limit before providers.

Real HTTP service fixtures verify that jobs outlive the admitting HTTP action, caller scope and capability checks reject spoofed or host-path input, multibyte byte bounds hold, cancellation remains available during blocked upload and capture preparation, and HTTP resume completes the saved job. A separate transport fixture aborts a genuinely pending upload waiter and confirms accepted admission drains during shutdown. A blocked-worker fixture shuts down the actual worker pool, observes a durable interrupted checkpoint, starts a new HTTP service, and resumes that saved input to completion.

The CLI fixtures retain a private upload draft and replay command on connection failure, preserve option-like metadata/request IDs and the explicit draft directory, and reject local-only upload before opening a corpus. Admission response validation prevents incompatible or mismatched acknowledgements from deleting the draft. A real CLI/HTTP recovery regression retries the exact admission using `--recover-draft`, removes its retained recovery file, and preserves the ordinary supplied input; the same fixture also covers actual lost acknowledgements for capture and edit.

A service worker regression injects actual stored status-decode and terminal-write failures, verifies fenced recovery remains tracked, repairs the fault, and resumes the job in the same process.

The real portable archive regression completes an upload, refreshes it and prunes its original chunks, verifies/restores the archive, and replays the exact admission and stored outcome. A second real-application regression preserves distinct active and pending processing snapshots through vector-inclusive and vectorless archives and JSONL exports, including their embedding configuration objects. Caller file-URI provenance and ordinary metadata words such as `token` and `embedding` remain opaque user content; ordinary host paths and secrets are stripped. Six rewritten archives containing malformed or wrong-table historical item/checkpoint IDs are rejected by both verify and restore before a target database is created.

## Reproduction

```sh
CARGO_BUILD_JOBS=4 RUSTC_WRAPPER='' cargo test --workspace --all-features --locked --offline
CARGO_BUILD_JOBS=4 RUSTC_WRAPPER='' cargo clippy --workspace --all-targets --all-features --locked --offline -- -D warnings
cargo fmt --all -- --check
git diff --check
```

Retained local evidence:

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
