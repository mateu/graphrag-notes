# GraphRAG CLI module map

The CLI keeps its public command line contract stable while separating its
implementation by responsibility:

- `main.rs` owns only process entry and documented exit-code conversion.
- `cli.rs` owns Clap parser definitions, aliases, flags, defaults, and
  CLI-facing value conversions.
- `app.rs` resolves configuration, performs lazy bootstrap and provider health
  checks, and constructs the small `AppContext` passed to dispatch.
- `dispatch.rs` maps parsed commands to command handlers and contains the
  cohesive non-notes command families plus their renderers.
- `commands/notes.rs` owns the safe note CRUD handlers.
- `output.rs` owns the shared human/JSON/JSONL envelope contract.
- `interactive.rs` owns only the interactive REPL.

Bootstrap intentionally returns before opening the database for configuration,
doctor, verification, import/restore, reset, and embedding-dimension command
paths where the prior behavior required it. Provider requirements are computed
from the parsed command in `app.rs`, so read-only and metadata-only operations
do not contact unnecessary inference providers.
