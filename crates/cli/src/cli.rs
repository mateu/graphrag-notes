//! Clap parser definitions and command-line value types.
//!
//! This module intentionally owns only parsing and CLI-facing conversion helpers.
//! Bootstrap, dispatch, and command behavior live outside it.

use crate::commands::notes::NotesCommand;
use crate::eval::EvalOutputFormat;
use crate::output;
use clap::{Parser, Subcommand, ValueEnum};
use graphrag_agents::{ChatImportMode, GraphMode};
use graphrag_core::ProposedEdgeStatus;
use std::path::PathBuf;

/// GraphRAG Notes - An evolving knowledge graph for your notes
#[derive(Parser)]
#[command(name = "graphrag")]
#[command(author, version, about, long_about = None)]
pub(crate) struct Cli {
    /// Configuration file. Defaults to GRAPHRAG_CONFIG, then
    /// the platform config directory's graphrag/config.toml when it exists.
    #[arg(long, global = true, value_name = "PATH")]
    pub(crate) config: Option<PathBuf>,

    /// Shared MCP endpoint (or GRAPHRAG_SERVER). Remote commands never open a local database.
    #[arg(long, global = true, value_name = "URL")]
    pub(crate) server: Option<String>,

    /// Environment variable containing the remote bearer credential.
    #[arg(
        long,
        global = true,
        default_value = "GRAPHRAG_TOKEN",
        value_name = "NAME"
    )]
    pub(crate) credential_env: String,

    /// Stable identity for a remote write; reuse it with the identical payload after uncertainty.
    #[arg(long, global = true, value_name = "ID")]
    pub(crate) request_id: Option<String>,

    /// Adopt a private --content-file recovery draft and remove it after verified remote success.
    #[arg(long, global = true, requires = "request_id")]
    pub(crate) recover_draft: bool,

    /// Revision from remote notes show or garden review; required for remote mutations.
    #[arg(long, global = true, value_name = "REVISION")]
    pub(crate) expected_revision: Option<String>,

    /// Database path (overrides the resolved configuration)
    #[arg(short, long, global = true)]
    pub(crate) db_path: Option<PathBuf>,

    /// Use in-memory database (for testing)
    #[arg(long)]
    pub(crate) memory: bool,

    /// Verbose output (DEBUG logs)
    #[arg(short, long)]
    pub(crate) verbose: bool,

    /// Maximum in-flight requests per inference provider/operation.
    #[arg(long, global = true, value_name = "N")]
    pub(crate) concurrency: Option<usize>,

    /// Maximum inference attempts including the initial request.
    #[arg(long, global = true, value_name = "N")]
    pub(crate) retry_attempts: Option<usize>,

    /// Bypass the durable local inference cache for this invocation.
    #[arg(long, global = true)]
    pub(crate) no_cache: bool,

    /// Include versioned retrieval and context-selection evidence where the
    /// selected command supports explainability.
    #[arg(long, global = true)]
    pub(crate) explain: bool,

    #[command(subcommand)]
    pub(crate) command: Commands,
}

#[derive(Subcommand)]
pub(crate) enum Commands {
    /// Own this host's database and expose authenticated Streamable HTTP MCP.
    Serve {
        #[arg(long, default_value = "127.0.0.1:3000", value_name = "ADDRESS")]
        listen: std::net::SocketAddr,
        /// Private JSON credential policy containing token hashes, never raw credentials.
        #[arg(long, value_name = "PATH")]
        credentials_file: PathBuf,
        /// Confirm an encrypted proxy or private tunnel protects non-loopback access.
        #[arg(long)]
        external_encryption: bool,
        /// Explicit allowed HTTP Host headers (repeat for each hostname).
        #[arg(long = "allowed-host", value_name = "HOST")]
        allowed_hosts: Vec<String>,
        #[arg(long, default_value_t = 131_072)]
        max_request_body_bytes: usize,
        #[arg(long, default_value_t = 8)]
        max_concurrent_requests: usize,
    },
    /// Preview first-run settings; explicitly create a new config or check providers
    Init {
        /// Preset for a NEW config; environment overrides still apply at runtime.
        #[arg(long, value_enum)]
        backend: Option<SetupBackend>,
        /// Create the selected config file; never overwrite an existing file.
        #[arg(long)]
        write: bool,
        /// Check inference providers without opening or creating a database.
        #[arg(long)]
        check: bool,
        #[arg(long, value_enum, default_value_t = DoctorFormat::Human)]
        format: DoctorFormat,
    },
    /// Create, verify, or restore a portable logical backup
    Backup {
        #[command(subcommand)]
        command: BackupCommand,
    },

    /// Export portable logical records as JSONL plus a checksum manifest sidecar
    Export {
        path: PathBuf,
        #[arg(long, value_enum, default_value_t = PortableDataFormat::Jsonl)]
        format: PortableDataFormat,
        #[arg(long, value_enum, default_value_t = BackupOutputFormat::Human)]
        output: BackupOutputFormat,
    },

    /// Import a verified portable JSONL export into a fresh database path
    ImportData {
        path: PathBuf,
        /// Validate all safety preconditions without writing the target
        #[arg(long)]
        dry_run: bool,
        #[arg(long, value_enum, default_value_t = BackupOutputFormat::Human)]
        format: BackupOutputFormat,
    },

    /// Diagnose configuration, database compatibility, and local providers without changing data
    Doctor {
        /// Output format for the diagnostic report
        #[arg(long, value_enum, default_value_t = DoctorFormat::Human)]
        format: DoctorFormat,
    },

    /// Inspect or validate the resolved runtime configuration
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },

    /// Register host-local Markdown folders or preview missing-file pruning
    Folders {
        #[command(subcommand)]
        command: FolderCommand,
    },

    /// Explicit one-shot Markdown folder sync; never starts a watcher
    #[command(group(clap::ArgGroup::new("target").required(true).args(["name", "all", "resume"])))]
    Sync {
        name: Option<String>,
        #[arg(long)]
        all: bool,
        #[arg(long)]
        dry_run: bool,
        /// Resume only the pinned roots and file identities of this job
        #[arg(long, value_name = "JOB_ID")]
        resume: Option<String>,
        #[arg(long, value_enum, default_value_t = output::OutputFormat::Human)]
        format: output::OutputFormat,
    },

    /// Safely list, show, edit, or delete notes
    Notes {
        #[command(subcommand)]
        command: NotesCommand,
    },

    /// Capture multiline text or a blocking editor draft with recoverable failures.
    Capture {
        #[command(flatten)]
        args: crate::commands::capture::CaptureArgs,
    },

    /// Add a new note
    Add {
        /// Note content (reads from stdin if not provided)
        content: Option<String>,

        /// Note title
        #[arg(short, long)]
        title: Option<String>,

        /// Tags (comma-separated)
        #[arg(short = 'T', long)]
        tags: Option<String>,
    },

    /// Import from a file
    Import {
        /// Path to file
        path: PathBuf,

        /// Rebuild the source even when its normalized content hash is unchanged
        #[arg(long)]
        force: bool,
    },

    /// Inspect, remove, or reimport file-backed sources
    Sources {
        #[command(subcommand)]
        command: SourcesCommand,
    },

    /// Import chat conversations from Claude Desktop or other chat exports
    ImportChats {
        /// Path to JSON file containing chat export
        path: PathBuf,

        /// Import mode: qa (default), message, or hybrid
        #[arg(long, value_enum, default_value_t = ImportModeArg::Qa)]
        mode: ImportModeArg,

        /// Skip entity extraction (faster for testing)
        #[arg(long)]
        skip_extraction: bool,
    },

    /// Migrate chat exports into conversation/message tables (and optionally regenerate notes)
    MigrateChats {
        /// Path to JSON file containing chat export
        path: PathBuf,

        /// Preview counts without writing to the database
        #[arg(long)]
        dry_run: bool,

        /// Also regenerate derived notes (idempotent: skips conversations already linked to notes)
        #[arg(long)]
        with_notes: bool,

        /// Note import mode when using --with-notes
        #[arg(long, value_enum, default_value_t = ImportModeArg::Hybrid)]
        mode: ImportModeArg,

        /// Skip entity extraction when using --with-notes
        #[arg(long)]
        skip_extraction: bool,
    },

    /// Search notes, messages, or conversation summaries
    Search {
        /// Search query
        query: String,

        /// Retrieval mode: hybrid uses vectors and full text; keyword is offline full text only.
        #[arg(long, value_enum, default_value_t = SearchModeArg::Hybrid)]
        mode: SearchModeArg,

        /// Maximum results
        #[arg(short, long)]
        limit: Option<usize>,

        /// Search scope: notes, messages, or all
        #[arg(long, value_enum, default_value_t = SearchScopeArg::Notes)]
        scope: SearchScopeArg,

        /// Restrict results to records updated/created in the last N days
        #[arg(long)]
        since_days: Option<u32>,

        /// Restrict results to a specific source URI
        #[arg(long)]
        source_uri: Option<String>,

        /// Include graph context
        #[arg(short, long)]
        context: bool,

        /// Accepted-edge graph retrieval policy
        #[arg(long, value_enum, default_value_t = GraphModeArg::Auto)]
        graph: GraphModeArg,

        /// Render results for people, JSON consumers, or JSONL pipelines.
        #[arg(long, value_enum, default_value_t = output::OutputFormat::Human)]
        format: output::OutputFormat,
    },

    /// Inspect a note, message, or conversation by its full search-result ID
    Inspect {
        /// Canonical note:ID, message:ID, or conversation:ID (never a result number)
        id: String,
        /// Adjacent messages on each side of a chat hit (maximum 20)
        #[arg(long, default_value_t = 2)]
        neighbors: usize,
        /// Refuse a record that changed since the displayed search result
        #[arg(long)]
        revision: Option<String>,
        #[arg(long, value_enum, default_value_t = output::OutputFormat::Human)]
        format: output::OutputFormat,
    },

    /// Explicitly open a result's original local source file
    Open {
        /// Canonical note:ID, message:ID, or conversation:ID
        id: String,
        /// Refuse a record that changed since the displayed search result
        #[arg(long)]
        revision: Option<String>,
        /// Opener executable path; configured arguments are overridden
        #[arg(long, value_name = "PATH")]
        opener: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = output::OutputFormat::Human)]
        format: output::OutputFormat,
    },

    /// Build prompt-ready augmentation context with citations
    Augment {
        /// Retrieval query
        query: String,

        /// Maximum context chunks to include
        #[arg(short, long)]
        limit: Option<usize>,

        /// Search scope: notes, messages, or all
        #[arg(long, value_enum, default_value_t = SearchScopeArg::All)]
        scope: SearchScopeArg,

        /// Restrict results to records updated/created in the last N days
        #[arg(long)]
        since_days: Option<u32>,

        /// Restrict results to a specific source URI
        #[arg(long)]
        source_uri: Option<String>,

        /// Filter note chunks to notes linked to matching entities
        #[arg(long)]
        entity: Option<String>,

        /// Approximate max tokens across all chunks
        #[arg(long)]
        max_tokens: Option<usize>,

        /// Approximate max tokens per chunk
        #[arg(long)]
        max_chunk_tokens: Option<usize>,

        /// Accepted-edge graph retrieval policy
        #[arg(long, value_enum, default_value_t = GraphModeArg::Auto)]
        graph: GraphModeArg,

        /// Print only packed prompt context and citations for redirection.
        #[arg(long, conflicts_with = "format")]
        raw: bool,

        /// Render results for people, JSON consumers, or JSONL pipelines.
        #[arg(long, value_enum, default_value_t = output::OutputFormat::Human)]
        format: output::OutputFormat,
    },

    /// Evaluate augmentation retrieval quality from a JSON/JSONL test set
    EvalAugment {
        /// Path to eval cases (.json array or .jsonl object-per-line)
        path: PathBuf,

        /// Default max chunks when case limit is not set
        #[arg(short, long)]
        limit: Option<usize>,

        /// Default scope when case scope is not set
        #[arg(long, value_enum, default_value_t = SearchScopeArg::All)]
        scope: SearchScopeArg,

        /// Default recency filter when case since_days is not set
        #[arg(long)]
        since_days: Option<u32>,

        /// Default source filter when case source_uri is not set
        #[arg(long)]
        source_uri: Option<String>,

        /// Default global token budget when case max_tokens is not set
        #[arg(long)]
        max_tokens: Option<usize>,

        /// Default per-chunk token budget when case max_chunk_tokens is not set
        #[arg(long)]
        max_chunk_tokens: Option<usize>,

        /// Exit with status 1 when any expected case misses
        #[arg(long)]
        fail_on_miss: bool,

        /// Output format for the evaluation report
        #[arg(long, value_enum, default_value_t = EvalOutputFormat::Human)]
        format: EvalOutputFormat,

        /// Compare aggregate metrics against a prior JSON report
        #[arg(long)]
        baseline: Option<PathBuf>,

        /// Allowed metric decrease, such as `recall_at_k=0.02` (repeatable)
        #[arg(
            long = "max-regression",
            value_name = "METRIC=MAX_DROP",
            requires = "baseline"
        )]
        max_regression: Vec<String>,
    },

    /// List recent notes
    List {
        /// Maximum results
        #[arg(short, long, default_value = "20")]
        limit: usize,
    },

    /// Scan and review persistent Gardener edge proposals
    Garden {
        #[command(subcommand)]
        command: GardenCommand,
    },

    /// Inspect and control durable local inference jobs
    Jobs {
        #[command(subcommand)]
        command: JobsCommand,
    },

    /// Rebuild embeddings through a durable, all-or-nothing cutover job
    Reindex {
        /// Reindex visible notes
        #[arg(long)]
        notes: bool,
        /// Reindex chat messages
        #[arg(long)]
        messages: bool,
        /// Reindex conversation summaries
        #[arg(long)]
        summaries: bool,
        /// Reindex notes, messages, and summaries
        #[arg(long)]
        all: bool,
        /// Show the immutable job scope and provider preflight without writing
        #[arg(long)]
        dry_run: bool,
        /// Resume a failed or cancelled reindex processing job
        #[arg(long, value_name = "JOB_ID")]
        resume: Option<String>,
        #[arg(long, value_enum, default_value_t = JobOutputFormat::Human)]
        format: JobOutputFormat,
    },

    /// Show database statistics
    Stats,

    /// Show the database's current and this binary's latest schema version
    SchemaVersion,

    /// Persistent terminal workspace for browsing, capture, and review
    #[command(name = "workspace", visible_alias = "interactive")]
    Interactive,

    /// Show the embedding dimension from the active embeddings provider
    EmbeddingDim {
        /// Optional text to embed (defaults to "dimension probe")
        text: Option<String>,
    },

    /// Extract entities for notes that are missing entity links
    ExtractEntities {
        /// Maximum notes to process (or page size when using --all)
        #[arg(short, long, default_value = "100")]
        limit: usize,

        /// Process all notes (not just those missing mentions)
        #[arg(long)]
        all: bool,

        /// Process only specific note id(s); repeat this flag per note
        #[arg(long = "note-id")]
        note_ids: Vec<String>,

        /// Clear existing mentions before re-extracting (use with --all or --note-id)
        #[arg(long)]
        force: bool,
    },

    /// Show entities linked to a note
    ShowEntities {
        /// Note ID (e.g., note:xxxxxxxx)
        note_id: String,
    },

    /// Show a note by ID
    ShowNote {
        /// Note ID (e.g., note:xxxxxxxx)
        note_id: String,
    },

    /// List note-to-note edges
    ListEdges {
        /// Maximum edges per edge type
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },

    /// Show note-to-note edges for a specific note
    ShowNoteEdges {
        /// Note ID (e.g., note:xxxxxxxx)
        note_id: String,
    },

    /// Safely inspect or remove accepted note-to-note edges
    Edges {
        #[command(subcommand)]
        command: EdgesCommand,
    },

    /// Delete the local database (fresh start)
    ResetDb {
        /// Database path (defaults to ~/.graphrag/data-v3)
        #[arg(short, long)]
        db_path: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
pub(crate) enum FolderCommand {
    /// Append a new folder declaration to the selected existing TOML config
    Add {
        name: String,
        path: PathBuf,
        #[arg(long)]
        no_recursive: bool,
        /// Include glob relative to the root; repeat to replace default includes
        #[arg(long)]
        include: Vec<String>,
        /// Additional exclude glob relative to the root; repeat as needed
        #[arg(long)]
        exclude: Vec<String>,
        #[arg(long, value_enum, default_value_t = output::OutputFormat::Human)]
        format: output::OutputFormat,
    },
    List {
        #[arg(long, value_enum, default_value_t = output::OutputFormat::Human)]
        format: output::OutputFormat,
    },
    /// Preview missing generated sources; confirmation requires the preview token
    Prune {
        name: String,
        #[arg(long, requires = "revision")]
        yes: bool,
        #[arg(long, requires = "yes")]
        revision: Option<String>,
        #[arg(long, value_enum, default_value_t = output::OutputFormat::Human)]
        format: output::OutputFormat,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub(crate) enum SetupBackend {
    Ollama,
    TeiTgi,
}

#[derive(Subcommand)]
pub(crate) enum ConfigCommand {
    /// Print the fully resolved configuration (with secrets redacted)
    Show,
    /// Validate configuration and exit without opening the database
    Validate,
}

#[derive(Subcommand)]
pub(crate) enum BackupCommand {
    /// Stream a versioned portable backup into a new directory
    Create {
        path: PathBuf,
        /// Include vectors only when persisted model identity is available
        #[arg(long)]
        include_embeddings: bool,
        #[arg(long, value_enum, default_value_t = BackupOutputFormat::Human)]
        format: BackupOutputFormat,
    },
    /// Validate manifest, checksum, record counts, dimensions, and references
    Verify {
        path: PathBuf,
        #[arg(long, value_enum, default_value_t = BackupOutputFormat::Human)]
        format: BackupOutputFormat,
    },
    /// Restore a verified archive into a fresh, nonexistent database path
    Restore {
        path: PathBuf,
        /// Validate all safety preconditions without writing the target
        #[arg(long)]
        dry_run: bool,
        #[arg(long, value_enum, default_value_t = BackupOutputFormat::Human)]
        format: BackupOutputFormat,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub(crate) enum BackupOutputFormat {
    Human,
    Json,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub(crate) enum PortableDataFormat {
    Jsonl,
}

#[derive(Subcommand)]
pub(crate) enum SourcesCommand {
    /// List source lifecycle state
    List {
        #[arg(long, value_enum, default_value_t = SourceOutputFormat::Human)]
        format: SourceOutputFormat,
    },
    /// Show one source by record id or normalized URI
    Show {
        id_or_uri: String,
        #[arg(long, value_enum, default_value_t = SourceOutputFormat::Human)]
        format: SourceOutputFormat,
    },
    /// Preview or permanently delete one source and its generated records
    Delete {
        id_or_uri: String,
        /// Print exact mutation counts without changing the database
        #[arg(long)]
        dry_run: bool,
        /// Confirm permanent deletion (required unless --dry-run is set)
        #[arg(long)]
        yes: bool,
        #[arg(long, value_enum, default_value_t = SourceOutputFormat::Human)]
        format: SourceOutputFormat,
    },
    /// Read the original local file and run the normal staged import flow
    Reimport {
        id_or_uri: String,
        #[arg(long)]
        force: bool,
        #[arg(long, value_enum, default_value_t = SourceOutputFormat::Human)]
        format: SourceOutputFormat,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub(crate) enum SourceOutputFormat {
    Human,
    Json,
}

#[derive(Subcommand)]
pub(crate) enum GardenCommand {
    /// Review both notes for persisted proposals without inference providers
    Review {
        /// Focus a single proposal using its full proposed_edge:ID
        #[arg(long, conflicts_with_all = ["status", "all_statuses"])]
        id: Option<String>,
        /// Filter the inbox; defaults to pending unless --all-statuses is set
        #[arg(long, value_enum, conflicts_with = "all_statuses")]
        status: Option<ProposalStatusArg>,
        /// Include reviewed and superseded proposals in the inbox
        #[arg(long)]
        all_statuses: bool,
        /// Maximum proposals to review (1–200)
        #[arg(short, long, default_value_t = 20)]
        limit: usize,
        /// Walk the inbox with confirmed accept/reject/undo and read-only skip
        #[arg(long)]
        interactive: bool,
        #[arg(long, value_enum, default_value_t = crate::output::OutputFormat::Human)]
        format: crate::output::OutputFormat,
    },
    /// Generate reviewable proposals; accepted edge tables are never mutated
    Scan {
        /// Preview candidate count without persisting proposals
        #[arg(long)]
        dry_run: bool,
    },
    /// Scan and apply the explicitly configured auto-apply policy
    Apply {
        /// Confirm mutation of accepted edges under the configured policy
        #[arg(long, required = true)]
        yes: bool,
    },
    /// Inspect or act on persisted proposals
    Proposals {
        #[command(subcommand)]
        command: ProposalCommand,
    },
}

#[derive(Subcommand)]
pub(crate) enum JobsCommand {
    List {
        #[arg(short, long, default_value_t = 50)]
        limit: usize,
        #[arg(long, value_enum, default_value_t = JobOutputFormat::Human)]
        format: JobOutputFormat,
    },
    Show {
        id: String,
        #[arg(long, value_enum, default_value_t = JobOutputFormat::Human)]
        format: JobOutputFormat,
    },
    /// Resume a failed or cancelled job from its durable checkpoint.
    Resume { id: String },
    /// Request cancellation between atomic item mutations.
    Cancel { id: String },
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub(crate) enum JobOutputFormat {
    Human,
    Json,
}

#[derive(Subcommand)]
pub(crate) enum ProposalCommand {
    List {
        #[arg(long, value_enum)]
        status: Option<ProposalStatusArg>,
        #[arg(short, long, default_value_t = 50)]
        limit: usize,
    },
    Show {
        id: String,
    },
    Accept {
        /// Proposal ID. Omit only with --all.
        id: Option<String>,
        /// Accept every pending Gardener proposal meeting --min-confidence.
        #[arg(long)]
        all: bool,
        /// Required with --all; values are in [0, 1].
        #[arg(long)]
        min_confidence: Option<f32>,
        /// Confirm a mutating acceptance.
        #[arg(long)]
        yes: bool,
        #[arg(long)]
        reason: Option<String>,
    },
    Reject {
        id: String,
        #[arg(long)]
        reason: Option<String>,
        /// Confirm a mutating rejection.
        #[arg(long)]
        yes: bool,
    },
    /// Undo an accepted proposal and preserve its review audit.
    Undo {
        id: String,
        #[arg(long)]
        reason: Option<String>,
        #[arg(long)]
        yes: bool,
    },
}

#[derive(Subcommand)]
pub(crate) enum EdgesCommand {
    Delete {
        id: String,
        /// Show whether the edge exists without deleting it.
        #[arg(long)]
        dry_run: bool,
        /// Confirm deletion.
        #[arg(long)]
        yes: bool,
    },
    /// Alias for `edges delete`; preserves proposal audit state.
    Undo {
        id: String,
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        yes: bool,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub(crate) enum ProposalStatusArg {
    Pending,
    Accepting,
    Accepted,
    Rejected,
    Superseded,
}

impl From<ProposalStatusArg> for ProposedEdgeStatus {
    fn from(value: ProposalStatusArg) -> Self {
        match value {
            ProposalStatusArg::Pending => Self::Pending,
            ProposalStatusArg::Accepting => Self::Accepting,
            ProposalStatusArg::Accepted => Self::Accepted,
            ProposalStatusArg::Rejected => Self::Rejected,
            ProposalStatusArg::Superseded => Self::Superseded,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub(crate) enum ImportModeArg {
    Qa,
    Message,
    Hybrid,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub(crate) enum SearchModeArg {
    Hybrid,
    Keyword,
}

impl SearchModeArg {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Hybrid => "hybrid",
            Self::Keyword => "keyword",
        }
    }

    /// Base retrieval channels. Accepted-edge graph contributions retain
    /// their separate evidence and summary in hybrid mode.
    pub(crate) fn channels(self) -> &'static [&'static str] {
        match self {
            Self::Hybrid => &["vector", "full_text"],
            Self::Keyword => &["full_text"],
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub(crate) enum SearchScopeArg {
    Notes,
    Messages,
    All,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub(crate) enum GraphModeArg {
    Off,
    Auto,
    On,
}

impl From<GraphModeArg> for GraphMode {
    fn from(value: GraphModeArg) -> Self {
        match value {
            GraphModeArg::Off => Self::Off,
            GraphModeArg::Auto => Self::Auto,
            GraphModeArg::On => Self::On,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub(crate) enum DoctorFormat {
    Human,
    Json,
}

pub(crate) fn to_import_mode(mode: ImportModeArg) -> ChatImportMode {
    match mode {
        ImportModeArg::Qa => ChatImportMode::Qa,
        ImportModeArg::Message => ChatImportMode::Message,
        ImportModeArg::Hybrid => ChatImportMode::Hybrid,
    }
}

/// Metadata-only note edits are local repository updates. Content replacement
/// and source detachment each need the embedding and extraction providers.
pub(crate) fn notes_edit_requires_inference(command: &NotesCommand) -> bool {
    matches!(
        command,
        NotesCommand::Edit {
            content_file: Some(_),
            ..
        } | NotesCommand::Edit { stdin: true, .. }
            | NotesCommand::Edit { detach: true, .. }
    ) || matches!(command, NotesCommand::Edit { editor, .. } if editor.editor)
}
