use super::editor::{self, Draft, EditorOptions, EditorOutcome};
use super::navigation::shell_quote;
use crate::output::{self, OutputFormat};
use anyhow::{bail, Result};
use clap::Subcommand;
use graphrag_agents::LibrarianAgent;
use graphrag_core::{record_id_to_string, Note};
use graphrag_db::{repository::SearchResult, Repository, SourceDeleteSummary};
use serde::Serialize;
use std::fmt;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

/// Explicit validation failure for local edit input. This is deliberately
/// distinct from provider/database I/O so the top-level CLI can honor the
/// documented validation exit code for a missing or unreadable content file.
#[derive(Debug)]
pub enum NotesEditValidationError {
    UnreadableContentFile { path: PathBuf, source: io::Error },
}

impl fmt::Display for NotesEditValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnreadableContentFile { path, .. } => {
                write!(formatter, "failed to read content file: {}", path.display())
            }
        }
    }
}

impl std::error::Error for NotesEditValidationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::UnreadableContentFile { source, .. } => Some(source),
        }
    }
}

#[derive(Subcommand)]
pub enum NotesCommand {
    /// List visible notes, optionally constrained by every requested tag and source URI.
    List {
        #[arg(short, long, default_value_t = 20)]
        limit: usize,
        /// Require this tag (repeat or use comma-separated values for multiple tags).
        #[arg(long, value_delimiter = ',')]
        tag: Vec<String>,
        #[arg(long)]
        source_uri: Option<String>,
        #[arg(long, value_enum, default_value_t = OutputFormat::Human)]
        format: OutputFormat,
    },
    /// Show one visible note by record id.
    Show {
        id: String,
        #[arg(long, value_enum, default_value_t = OutputFormat::Human)]
        format: OutputFormat,
    },
    /// Edit a manual note. Source-generated notes require --detach, which creates a manual copy.
    Edit {
        id: String,
        #[arg(long)]
        title: Option<String>,
        #[arg(long, value_name = "PATH", conflicts_with = "stdin")]
        content_file: Option<PathBuf>,
        #[arg(long, conflicts_with = "content_file")]
        stdin: bool,
        #[command(flatten)]
        editor: EditorOptions,
        #[arg(long, value_delimiter = ',')]
        tags: Option<Vec<String>>,
        /// Create a new manual note instead of changing a source-generated chunk.
        #[arg(long)]
        detach: bool,
        #[arg(long, value_enum, default_value_t = OutputFormat::Human)]
        format: OutputFormat,
    },
    /// Preview or permanently delete one note and only its dependent records.
    Delete {
        id: String,
        /// Show the exact cascade without changing data.
        #[arg(long)]
        dry_run: bool,
        /// Confirm the permanent delete. Without this flag the command previews safely.
        #[arg(long)]
        yes: bool,
        #[arg(long, value_enum, default_value_t = OutputFormat::Human)]
        format: OutputFormat,
    },
}

#[derive(Serialize)]
struct NoteListOutput<'a> {
    notes: &'a [SearchResult],
}

#[derive(Serialize)]
struct NoteMutationOutput<'a> {
    note: &'a Note,
    detached: bool,
}

#[derive(Serialize)]
struct NoteDeleteOutput<'a> {
    id: &'a str,
    dry_run: bool,
    cascade: &'a SourceDeleteSummary,
}

/// Content consumed before provider health checks. Reusing it in execution
/// prevents `--stdin` from being read twice.
#[derive(Debug)]
pub struct PreparedEdit {
    content: Option<String>,
    draft: Option<Draft>,
    outcome: Option<EditorOutcome>,
    expected: Option<Note>,
}

impl PreparedEdit {
    pub fn needs_inference(&self, command: &NotesCommand) -> bool {
        self.content.is_some() || matches!(command, NotesCommand::Edit { detach: true, .. })
    }
}

/// Validate an edit's local inputs before any embedding/extraction preflight.
/// This makes missing-note, source-ownership, unreadable-file, and empty-body
/// errors deterministic even when providers are offline.
pub async fn prepare_edit(
    repo: &Repository,
    command: &NotesCommand,
    config_path: Option<&Path>,
    database: &Path,
) -> Result<Option<PreparedEdit>> {
    let NotesCommand::Edit {
        id,
        title,
        content_file,
        stdin,
        editor: options,
        tags,
        detach,
        format,
        ..
    } = command
    else {
        return Ok(None);
    };
    let existing = get_visible_note(repo, id).await?;
    if !detach && repo.note_requires_detach(&existing).await? {
        let base = editor::base_command(config_path, database)?;
        let source_action = if existing.source_generation.is_some() {
            format!("open its source with {base} open {}", shell_quote(id))
        } else {
            format!(
                "inspect its chat provenance with {base} inspect {}",
                shell_quote(id)
            )
        };
        bail!("refusing to edit source-generated note {id} in place; {source_action} or explicitly create a manual copy with {base} notes edit {} --detach --editor", shell_quote(id));
    }
    let mut draft = None;
    let mut outcome = None;
    let content = if options.editor || content_file.is_some() || *stdin {
        let base = editor::base_command(config_path, database)?;
        let directory = editor::directory(options, database)?;
        let bytes = if options.editor {
            existing.content.as_bytes().to_vec()
        } else if let Some(path) = content_file {
            std::fs::read(path).map_err(|source| {
                NotesEditValidationError::UnreadableContentFile {
                    path: path.clone(),
                    source,
                }
            })?
        } else {
            let mut bytes = Vec::new();
            io::stdin().read_to_end(&mut bytes)?;
            bytes
        };
        let saved = Draft::save(&bytes, &directory, |path| {
            let mut command = format!(
                "{base} notes edit {} --content-file {}",
                shell_quote(id),
                shell_quote(&path.to_string_lossy())
            );
            if *detach {
                command.push_str(" --detach");
            }
            if let Some(title) = title {
                command.push_str(&format!(" --title={}", shell_quote(title)));
            }
            if let Some(tags) = tags {
                command.push_str(&format!(" --tags={}", shell_quote(&tags.join(","))));
            }
            command.push_str(&format!(" --format {}", editor::format_flag(*format)));
            if options.draft_dir.is_some() {
                command.push_str(&format!(
                    " --draft-dir={}",
                    shell_quote(&directory.to_string_lossy())
                ));
            }
            command
        })?;
        let original = saved.read()?;
        let (editor_outcome, content) = if options.editor {
            saved.edit(options, &original)?
        } else {
            (EditorOutcome::Changed, original)
        };
        draft = Some(saved);
        if options.editor && editor_outcome == EditorOutcome::Cancelled {
            outcome = Some(editor_outcome);
            None
        } else if options.editor && editor_outcome == EditorOutcome::Unchanged {
            // Opening and leaving an editor unchanged is always a no-op,
            // including explicit metadata and detach flags on this invocation.
            outcome = Some(editor_outcome);
            None
        } else {
            Some(content)
        }
    } else {
        None
    };
    if outcome.is_some() {
        return Ok(Some(PreparedEdit {
            content,
            draft,
            outcome,
            expected: options.editor.then_some(existing),
        }));
    }
    validate_edit_request(
        &existing,
        id,
        title.as_deref(),
        tags.as_deref(),
        content.as_deref(),
        *detach,
    )?;
    Ok(Some(PreparedEdit {
        content,
        draft,
        outcome,
        expected: options.editor.then_some(existing),
    }))
}

pub fn print_editor_noop(prepared: &mut PreparedEdit, command: &NotesCommand) -> Result<bool> {
    let Some(outcome) = prepared.outcome else {
        return Ok(false);
    };
    let NotesCommand::Edit { id, format, .. } = command else {
        return Ok(false);
    };
    let cancelled = outcome == EditorOutcome::Cancelled;
    let draft = prepared.draft.as_mut().expect("editor session has a draft");
    let retained = cancelled || !draft.discard(Some(id));
    output::print(
        *format,
        "notes.edit",
        serde_json::json!({
            "status": if cancelled { "cancelled" } else { "unchanged" }, "id": id,
            "detached": false, "draft_path": retained.then_some(&draft.path),
            "recovery_command": if cancelled { draft.recoverable_command() } else { None },
        }),
        |writer| {
            writeln!(
                writer,
                "Editor {}; note {id} retained.",
                if cancelled { "cancelled" } else { "unchanged" }
            )
        },
    )?;
    Ok(true)
}

pub async fn run(
    repo: Repository,
    librarian: LibrarianAgent,
    command: NotesCommand,
    prepared_edit: Option<PreparedEdit>,
) -> Result<()> {
    match command {
        NotesCommand::List {
            limit,
            tag,
            source_uri,
            format,
        } => list(repo, limit, tag, source_uri, format).await,
        NotesCommand::Show { id, format } => show(repo, id, format).await,
        NotesCommand::Edit {
            id,
            title,
            content_file,
            stdin,
            editor: _,
            tags,
            detach,
            format,
        } => {
            edit(
                repo,
                librarian,
                id,
                title,
                content_file,
                stdin,
                tags,
                detach,
                format,
                prepared_edit,
            )
            .await
        }
        NotesCommand::Delete {
            id,
            dry_run,
            yes,
            format,
        } => delete(repo, id, dry_run || !yes, format).await,
    }
}

pub async fn list(
    repo: Repository,
    limit: usize,
    tags: Vec<String>,
    source_uri: Option<String>,
    format: OutputFormat,
) -> Result<()> {
    let notes = repo
        .list_notes_filtered(limit, &tags, source_uri.as_deref())
        .await?;
    if format == OutputFormat::Jsonl {
        return output::print_jsonl("notes.list", notes);
    }
    output::print(
        format,
        "notes.list",
        NoteListOutput { notes: &notes },
        |writer| print_note_list(writer, &notes),
    )
}

pub async fn show(repo: Repository, id: String, format: OutputFormat) -> Result<()> {
    let note = get_visible_note(&repo, &id).await?;
    output::print(format, "notes.show", &note, |writer| {
        print_note(writer, &note)
    })
}

#[allow(clippy::too_many_arguments)]
async fn edit(
    repo: Repository,
    librarian: LibrarianAgent,
    id: String,
    title: Option<String>,
    content_file: Option<PathBuf>,
    stdin: bool,
    tags: Option<Vec<String>>,
    detach: bool,
    format: OutputFormat,
    mut prepared_edit: Option<PreparedEdit>,
) -> Result<()> {
    let guarded = prepared_edit
        .as_ref()
        .is_some_and(|prepared| prepared.expected.is_some());
    let existing = if let Some(expected) = prepared_edit
        .as_ref()
        .and_then(|prepared| prepared.expected.as_ref())
    {
        expected.clone()
    } else {
        get_visible_note(&repo, &id).await?
    };
    if !detach && repo.note_requires_detach(&existing).await? {
        bail!("refusing to edit source-generated note {id} in place; use --detach to create a manual note that retains source provenance");
    }
    let content = select_edit_content(prepared_edit.as_mut(), content_file, stdin)?;
    validate_edit_request(
        &existing,
        &id,
        title.as_deref(),
        tags.as_deref(),
        content.as_deref(),
        detach,
    )?;

    let updated = if detach {
        let content = content.unwrap_or_else(|| existing.content.clone());
        if guarded {
            librarian
                .detach_note_to_manual_guarded(&existing, content, title, tags)
                .await?
        } else {
            librarian
                .detach_note_to_manual(&existing, content, title, tags)
                .await?
        }
    } else if let Some(content) = content {
        if guarded {
            librarian
                .update_manual_note_content_guarded(&existing, content, title, tags)
                .await?
        } else {
            librarian
                .update_manual_note_content(&existing, content, title, tags)
                .await?
        }
    } else {
        let mut replacement = existing;
        if let Some(title) = title {
            replacement.title = Some(title);
        }
        if let Some(tags) = tags {
            replacement.tags = tags;
        }
        replacement.updated_at = chrono::Utc::now();
        repo.update_note(&id, replacement).await?
    };

    if let Some(draft) = prepared_edit
        .as_mut()
        .and_then(|prepared| prepared.draft.as_mut())
    {
        draft.discard(updated.id.as_ref().map(record_id_to_string).as_deref());
    }

    output::print(
        format,
        "notes.edit",
        NoteMutationOutput {
            note: &updated,
            detached: detach,
        },
        |writer| {
            writeln!(
                writer,
                "{} note {}",
                if detach {
                    "Created detached"
                } else {
                    "Updated"
                },
                record_id_to_string(updated.id.as_ref().expect("persisted note has id"))
            )
        },
    )
}

fn validate_edit_request(
    existing: &Note,
    id: &str,
    title: Option<&str>,
    tags: Option<&[String]>,
    content: Option<&str>,
    detach: bool,
) -> Result<()> {
    if !has_edit_action(title, tags, content, detach) {
        bail!("notes edit requires --title, --tags, --content-file, --stdin, or --detach");
    }
    if content.is_some_and(|content| content.trim().is_empty()) {
        bail!("note content cannot be empty");
    }
    if existing.source_generation.is_some() && !detach {
        bail!(
            "refusing to edit source-generated note {id} in place; use --detach to create a manual note that retains source provenance"
        );
    }
    Ok(())
}

fn has_edit_action(
    title: Option<&str>,
    tags: Option<&[String]>,
    content: Option<&str>,
    detach: bool,
) -> bool {
    detach || title.is_some() || tags.is_some() || content.is_some()
}

async fn delete(repo: Repository, id: String, dry_run: bool, format: OutputFormat) -> Result<()> {
    let cascade = if dry_run {
        repo.preview_note_delete(&id).await?
    } else {
        repo.delete_note_with_summary(&id).await?
    };
    output::print(
        format,
        "notes.delete",
        NoteDeleteOutput {
            id: &id,
            dry_run,
            cascade: &cascade,
        },
        |writer| print_delete_summary(writer, &id, dry_run, &cascade),
    )
}

async fn get_visible_note(repo: &Repository, id: &str) -> Result<Note> {
    repo.get_visible_note(id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("note not found: {id}"))
}

fn read_edit_content(content_file: Option<PathBuf>, stdin: bool) -> Result<Option<String>> {
    match (content_file, stdin) {
        (Some(path), false) => Ok(Some(std::fs::read_to_string(&path).map_err(|source| {
            anyhow::Error::new(NotesEditValidationError::UnreadableContentFile {
                path: path.clone(),
                source,
            })
        })?)),
        (None, true) => {
            let mut content = String::new();
            io::stdin().read_to_string(&mut content)?;
            Ok(Some(content))
        }
        (None, false) => Ok(None),
        (Some(_), true) => unreachable!("clap rejects conflicting content inputs"),
    }
}

fn select_edit_content(
    prepared_edit: Option<&mut PreparedEdit>,
    content_file: Option<PathBuf>,
    stdin: bool,
) -> Result<Option<String>> {
    match prepared_edit {
        Some(prepared) => Ok(prepared.content.take()),
        None => read_edit_content(content_file, stdin),
    }
}

fn print_note_list(writer: &mut dyn Write, notes: &[SearchResult]) -> io::Result<()> {
    for note in notes {
        writeln!(
            writer,
            "{}\t{}\t{}",
            record_id_to_string(&note.id),
            note.title.as_deref().unwrap_or("(untitled)"),
            note.tags.join(",")
        )?;
    }
    Ok(())
}

fn print_note(writer: &mut dyn Write, note: &Note) -> io::Result<()> {
    writeln!(
        writer,
        "id: {}",
        record_id_to_string(note.id.as_ref().expect("persisted note"))
    )?;
    writeln!(
        writer,
        "title: {}",
        note.title.as_deref().unwrap_or("(untitled)")
    )?;
    writeln!(writer, "tags: {}", note.tags.join(","))?;
    writeln!(writer, "content:\n{}", note.content)
}

fn print_delete_summary(
    writer: &mut dyn Write,
    id: &str,
    dry_run: bool,
    cascade: &SourceDeleteSummary,
) -> io::Result<()> {
    writeln!(
        writer,
        "{} note {id}: notes={} mentions={} edges={} proposals={} conversation_provenance={} message_provenance={}",
        if dry_run { "Would delete" } else { "Deleted" },
        cascade.notes,
        cascade.mentions,
        cascade.note_edges,
        cascade.proposals,
        cascade.note_conversation_provenance,
        cascade.note_message_provenance,
    )
}

#[cfg(test)]
mod tests {
    use super::{edit, has_edit_action, select_edit_content, validate_edit_request, PreparedEdit};
    use crate::commands::editor::Draft;
    use crate::output::{ExitCode, OutputFormat};
    use graphrag_agents::{DeterministicEmbedder, FixtureEntityExtractor, LibrarianAgent};
    use graphrag_core::{record_id_to_string, Note};
    use graphrag_db::{init_memory, Repository};
    use std::sync::Arc;

    #[tokio::test]
    async fn stale_editor_snapshot_preserves_concurrent_change_and_recovery_draft() {
        let repo = Repository::new(init_memory().await.unwrap());
        let opening = repo.create_note(Note::new("opening body")).await.unwrap();
        let id = record_id_to_string(opening.id.as_ref().unwrap());
        let directory = tempfile::tempdir().unwrap();
        let draft = Draft::save(b"editor replacement", directory.path(), |_| {
            "recover draft".into()
        })
        .unwrap();
        let draft_path = draft.path.clone();
        let prepared = PreparedEdit {
            content: Some("editor replacement".into()),
            draft: Some(draft),
            outcome: None,
            expected: Some(opening.clone()),
        };
        let mut concurrent = opening.clone();
        concurrent.content = "concurrently updated body".into();
        concurrent.title = Some("another client".into());
        // A content change must be detected even if an external writer reused
        // the original timestamp rather than advancing it.
        repo.update_note(&id, concurrent.clone()).await.unwrap();
        let librarian = LibrarianAgent::new(
            repo.clone(),
            Arc::new(DeterministicEmbedder::default()),
            Arc::new(FixtureEntityExtractor::default()),
        );
        let error = edit(
            repo.clone(),
            librarian,
            id.clone(),
            None,
            None,
            false,
            None,
            false,
            OutputFormat::Json,
            Some(prepared),
        )
        .await
        .unwrap_err();
        assert_eq!(crate::app::exit_code_for(&error), ExitCode::Validation);
        assert!(error.to_string().contains("changed"));
        let stored = repo.get_visible_note(&id).await.unwrap().unwrap();
        assert_eq!(stored.content, concurrent.content);
        assert_eq!(stored.title, concurrent.title);
        assert_eq!(stored.updated_at, opening.updated_at);
        assert_eq!(
            std::fs::read_to_string(draft_path).unwrap(),
            "editor replacement"
        );
        assert_eq!(repo.list_notes(10).await.unwrap().len(), 1);
    }

    #[test]
    fn detach_alone_is_an_edit_action() {
        assert!(has_edit_action(None, None, None, true));
        assert!(!has_edit_action(None, None, None, false));
    }

    #[test]
    fn edit_validation_rejects_empty_content_and_source_owned_in_place_before_inference() {
        let manual = Note::new("manual");
        assert!(
            validate_edit_request(&manual, "note:one", None, None, Some("  "), false)
                .unwrap_err()
                .to_string()
                .contains("content cannot be empty")
        );

        let mut source_generated = Note::new("source chunk");
        source_generated.source_generation = Some(1);
        assert!(validate_edit_request(
            &source_generated,
            "note:source",
            Some("replacement"),
            None,
            None,
            false,
        )
        .unwrap_err()
        .to_string()
        .contains("use --detach"));
    }

    #[test]
    fn prepared_edit_content_is_reused_without_reading_the_original_input_again() {
        let mut prepared = PreparedEdit {
            content: Some("already read".into()),
            draft: None,
            outcome: None,
            expected: None,
        };
        let content = select_edit_content(
            Some(&mut prepared),
            Some(std::path::PathBuf::from(
                "definitely-missing-content-file.md",
            )),
            false,
        )
        .unwrap();
        assert_eq!(content.as_deref(), Some("already read"));
    }
}
