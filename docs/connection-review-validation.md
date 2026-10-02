# Connection review validation

Issue: [#62](https://github.com/mateu/graphrag-notes/issues/62), under
the [day-to-day usability epic](https://github.com/mateu/graphrag-notes/issues/55).

All acceptance fixtures use disposable databases and isolated HOME/configuration.
The CLI fixture seeds or snapshots its persistent database in a child process
that exits before the next CLI command, releasing embedded database locks. An
ephemeral loopback listener records whether any review/decision contacts an
inference provider; every fixture confirms that no connection occurred. The
seed helper returns without writes during normal test discovery and is executed
explicitly by each real fixture.

| Check | Evidence |
| --- | --- |
| Both endpoint titles/excerpts, source heading/generation, confidence, reason, generator, state; JSON/JSONL remain read-only | `inbox_shows_both_notes_provenance_and_replayable_inspection_offline` passed, including exact revision-guarded inspection commands from another directory |
| Skip, cancelled confirmation, EOF during reason entry, and quit preserve all proposal timestamps/reviewer/reason/edge fields | `skip_cancel_and_eof_leave_proposals_and_audit_unchanged` passed with complete before/after snapshots |
| Accept/reject use existing reviewer/reason/manual audit; human history shows update/review times and the manual flag; accepted→undo retains original audit and retires the edge; superseded cannot be reaccepted | `decisions_and_undo_keep_the_existing_audit_and_terminal_state` passed |
| A legacy missing endpoint remains visible as an unavailable card and cannot be accepted | `missing_endpoints_block_acceptance_without_mutating_the_proposal` passed |
| Batch still requires confirmation and finite threshold; proposals below the selected threshold and logical/manual proposals remain pending; human history distinguishes manual CLI batch acceptance from automatic policy acceptance | `batch_threshold_confirmation_and_generator_boundaries_are_preserved` passed |
| Interactive JSON/JSONL, invalid IDs/limits, and missing focused proposal have clear failure outcomes and no prompts on machine stdout | `invalid_review_options_are_clear_and_machine_stdout_has_no_prompts` passed |
| Actual default backup/restore redacts source URI while preserving imported source identity/type/generation in the inbox | `restored_imported_notes_keep_source_identity_when_local_uris_are_redacted` passed |
| JSON/JSONL excerpts preserve Unicode and stay within 500 characters including the ellipsis at input lengths 499, 500, 501, and 1,500 | `machine_excerpts_keep_unicode_and_ellipsis_within_the_character_limit` failed against the original 501-character output, then passed after the fix |
| Changing displayed note content, rescanning proposal metadata, or deleting an endpoint cancels an obsolete decision without a new review audit | Three `commands::connections::tests` regressions passed |

Focused commands:

```sh
RUSTC_WRAPPER='' CARGO_BUILD_JOBS=2 cargo test --workspace \
  --test connection_review_commands --locked --offline -- --test-threads=2
RUSTC_WRAPPER='' CARGO_BUILD_JOBS=2 cargo test -p graphrag-cli \
  commands::connections::tests --locked --offline -- --test-threads=2
```

On macOS ARM64, the initial full workspace/all-feature suite passed **509 tests**, with
**5 existing ignored** diagnostics/fixtures (the keyword seeder is explicitly
run by its fixtures). This run preceded the additional Unicode excerpt
regression. It includes the pre-existing proposal acceptance,
rejection, lifecycle locking, source refresh, idempotence, and undo recovery
regressions. Follow-up targeted validation passed all **nine CLI tests** (eight
acceptance scenarios and one seed-helper discovery check) and **three snapshot
safety tests**, including the human audit display and excerpt-limit fixes.
Workspace/all-target/all-feature Clippy passed with `-D warnings`; formatting
and `git diff --check` also passed.

## macOS source-binary walkthrough

An additional walkthrough exercised the built executable from this worktree
with a fresh HOME/current directory, a temporary config/database, and both
inference endpoints set to `http://127.0.0.1:1`. A fictional corpus of Markdown
and manual notes, with pending Gardener and logical/manual proposals, was seeded
using the same isolated database helper as the tests. No personal corpus,
inference service, editor, or downloaded model was used.

**16 checks passed**, covering offline human/JSON/JSONL cards, both printed
inspection commands replayed from a different directory with matching
revisions, read-only skip and cancelled confirmation, accept with manual audit,
the printed undo command replayed from another directory, unchanged original
audit after undo, terminal refusal of acceptance, rejection, required batch
threshold, incompatible machine/interactive options, and missing endpoint
acceptance blocking. The script saved a local JSON report for the review
handoff. It used a config path containing an apostrophe and the source filename
contained an apostrophe, spaces, Unicode, and Japanese characters.

Failure/recovery coverage deliberately preserves pending proposals after skip
and cancellation. A superseded proposal cannot be resurrected by the inbox;
its full card remains available with `--id` or `--all-statuses`. A changed
snapshot is refused before mutation and the interactive session refreshes the
card. Missing proposals exit 3; invalid input or machine/interactive conflicts
exit 2. Existing durable `accepting` claims remain recoverable through the
explicit acceptance command, documented in the guide.

The revision comparison is a CLI pre-action guard, not a new atomic compare-and-
swap repository operation. The repository's existing lifecycle lock and visible
endpoint checks remain the authoritative write safeguards. Proposal scoring,
confidence thresholds, and graph governance are unchanged.

Published `v0.1.0-rc.1` assets predate this feature. Linux source behavior is
covered by portable CLI fixtures in CI; no native Linux walkthrough or new
binary release is claimed by this change. Review inbox cards are bounded to
200 proposals per invocation; use a focused stable proposal ID to revisit a
specific historical item.
