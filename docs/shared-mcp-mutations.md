# Remote note changes and connection review

The shared service supports manual note edits/deletion and explicitly confirmed
proposal acceptance, rejection and undo. Each operation uses the opening revision
and a durable request ID scoped to the authenticated instance. A second computer
can therefore review the same corpus without opening its RocksDB database.

Credentials retain their existing permissions. Grant `edit`, `delete`, `accept`,
`reject` or `undo` explicitly in the private credential policy for an instance
that needs that action. `read` and `capture` do not grant any of these permissions.
The tool catalog reflects the credential, and each call is authorized separately;
client tool filters do not substitute for server permissions. The trusted audit
actor is `mcp:<instance_id>` from authentication, never a client-supplied reviewer.
For a new private credential directory, the provisioning helper accepts explicit grants:

```bash
python3 scripts/provision-mcp-credentials.py "$HOME/.graphrag/mcp-private" \
  --client openclaw-a --client hermes \
  --grant openclaw-a=edit,delete --grant hermes=accept,reject,undo
```

Use an atomic mode-0600 policy replacement when changing existing grants;
provisioning never overwrites an existing credential directory.

## Inspect, then edit

```bash
graphrag --server http://127.0.0.1:3300/mcp notes show note:ID --format json
# Copy the returned revision after reviewing the note.
graphrag --server http://127.0.0.1:3300/mcp \
  --expected-revision REVISION --request-id edit-001 \
  notes edit note:ID --title 'Updated title' --tags project

graphrag --server http://127.0.0.1:3300/mcp \
  --expected-revision REVISION --request-id edit-002 \
  notes edit note:ID --editor --draft-dir "$HOME/.graphrag/private-drafts"
```

`--content-file` and `--stdin` upload replacement text from the client. An editor
starts from a server snapshot bound to the selected revision. Its private recovery
command preserves the same request ID, revision, title/tags and draft directory.
A conflict or lost response retains the draft. After a conflict, inspect the
current note and reconcile the draft before submitting a *new* request ID with
that current revision. After an uncertain outcome, retry the exact original
request ID and payload to recover its authoritative result.

Only manual notes are editable/deletable remotely. Imported file chunks, chat
messages and generated chat notes remain source-owned; change their original
source. Remote `--detach` is rejected. Metadata-only edits need no inference
provider; changed content uses the host's configured embedding/entity workflow.
The MCP `edit_note` patch also supports `clear_title: true` and an empty `tags`
array. Content is limited to 65536 UTF-8 bytes, titles to 512 characters and tags
to 32 nonempty values of 64 characters each.

Deletion requires both the reviewed revision and explicit confirmation:

```bash
graphrag --server http://127.0.0.1:3300/mcp notes delete note:ID --dry-run --format json
graphrag --server http://127.0.0.1:3300/mcp \
  --expected-revision REVISION --request-id delete-001 \
  notes delete note:ID --yes --format json
```

The remote dry run reads the note snapshot and sends no delete request. The
successful deletion result records its original cascade. Note removal, links,
mentions, affected proposal retirements and the receipt commit together.

## Review a connection

```bash
graphrag --server http://127.0.0.1:3300/mcp garden review --format json
graphrag --server http://127.0.0.1:3300/mcp \
  garden review --id proposed_edge:ID --format json

graphrag --server http://127.0.0.1:3300/mcp \
  --expected-revision REVISION --request-id accept-001 \
  garden proposals accept proposed_edge:ID --yes --reason 'Reviewed both notes'
```

Replace `accept` with `reject` for a pending proposal or `undo` for an accepted
one, using a fresh snapshot/revision and request ID for each new decision. Undo
removes the accepted edge and preserves the proposal audit as superseded. The
service rejects unavailable lifecycle transitions, changed endpoints, stale
cards and missing confirmation. Remote batch acceptance and interactive inbox
mutation are unsupported; select one reviewed proposal per decision.

The MCP tools are `get_note`, `edit_note`, `delete_note`, `list_proposals`,
`get_proposal` and `decide_proposal`. Decisions include `confirmed: true`, an
`action` of `accept`, `reject` or `undo`, and an optional reason (2048 characters
maximum). Read responses expose current revisions and availability; they contain
no vectors or credentials.

## Durable outcomes

The note/proposal mutation and its bounded original receipt commit in one database
transaction. Concurrent writes against one revision have one winner. An accepted
write finishes after an HTTP disconnect and during graceful service shutdown.
Reusing the same instance/request ID with the identical payload returns that
original outcome, even after later edits, deletion, restart or portable restore.
Changing the payload/action under that identity produces `revision_conflict`.

A receipt contains the previous revision and outcome; it does not claim to be a
current snapshot. Refresh `notes show` or `garden review` after success. The CLI
validates request identity, target, previous revision, operation and outcome
before removing recovery, so an incompatible acknowledgement cannot discard it.
Portable backups preserve the immutable receipt payload/result, while credentials
remain outside the corpus. Schema 17 requires a current binary; earlier portable
archives remain importable into a fresh current database.
