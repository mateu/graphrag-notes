# Uploaded source and durable job validation

Validated on Apple Silicon macOS on 2026-10-03 using isolated in-memory or disposable persistent databases, synthetic Markdown, deterministic providers, private credential files, and loopback HTTP. The standalone #77 working tree is based on foundation `5675096`; the owner repeats integration checks after stacking the remote-mutation changes. Published rc.2 does not contain these commands.

## Automated acceptance

The complete standalone workspace/all-feature run passed **638 tests**, with zero failures and five existing ignored tests across 22 nonempty targets. All-target/all-feature Clippy with warnings denied, formatting, and whitespace checks passed.

Fourteen dedicated database regressions cover principal-scoped admission retries/conflicts, source identity, lease/CAS ownership, cancellation outside the transition gate, restart reconciliation, bounded inputs, hidden staging, active versus pending provenance, unchanged title/configuration policy, extraction checkpoints, promotion/reconciliation retries, accepted-edge undo, portable input preservation, generic local-job bypass refusal, and deletion/recreation of the same source identity.

Seven application regressions cover provider failure then resume, exact request replay, independent instance ownership, source refresh and unchanged runs, cancellation of blocked preparation, stale worker epochs, extraction failure after a committed checkpoint, incompatible resume, and the 200-chunk admission limit before providers.

Real HTTP service fixtures verify that jobs outlive the admitting HTTP action, caller scope and capability checks reject spoofed or host-path input, multibyte byte bounds hold, cancellation remains available during blocked upload and capture preparation, and HTTP resume completes the saved job. A separate transport fixture aborts a genuinely pending upload waiter and confirms accepted admission drains during shutdown. A blocked-worker fixture shuts down the actual worker pool, observes a durable interrupted checkpoint, starts a new HTTP service, and resumes that saved input to completion.

The CLI fixtures retain a private upload draft and replay command on connection failure, preserve option-like metadata/request IDs and the explicit draft directory, and reject local-only upload before opening a corpus. Admission response validation prevents incompatible or mismatched acknowledgements from deleting the draft.

The real portable archive regression completes an upload, refreshes it and prunes its original chunks, verifies/restores the archive, and replays the exact admission and stored outcome. Caller file-URI provenance and ordinary metadata words such as `token` and `embedding` remain opaque user content; ordinary host paths and secrets are stripped. Six rewritten archives containing malformed or wrong-table historical item/checkpoint IDs are rejected by both verify and restore before a target database is created.

## Reproduction

```sh
CARGO_BUILD_JOBS=4 RUSTC_WRAPPER='' cargo test --workspace --all-features --locked --offline
CARGO_BUILD_JOBS=4 RUSTC_WRAPPER='' cargo clippy --workspace --all-targets --all-features --locked --offline -- -D warnings
cargo fmt --all -- --check
git diff --check
```

Retained local evidence:

- `/tmp/graphrag-77-workspace-tests.log` — final standalone full suite.
- `/tmp/graphrag-77-clippy-final.log` — final all-target/all-feature Clippy.
- `/tmp/graphrag-77-db-tests-final.log` — 145 database library tests, including the 14 upload regressions.
- `/tmp/graphrag-77-application-service-tests-final.log` — seven application job regressions and the preceding service run.
- `/tmp/graphrag-77-transport-cli-tests-final.log` — strengthened lost-ack/shutdown/restart HTTP fixtures plus CLI tests; the final full suite additionally includes the rewritten malformed archives.

The two-computer OpenClaw/Hermes deployment walkthrough is tracked by #78. These local fixtures exercise the native server/client boundaries without attributing them to external client sessions or a native Linux deployment.
