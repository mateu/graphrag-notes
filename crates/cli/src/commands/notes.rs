use crate::output::{self, OutputFormat};
use anyhow::{bail, Context, Result};
use clap::Subcommand;
use graphrag_agents::LibrarianAgent;
use graphrag_core::{record_id_to_string, Note};
use graphrag_db::{repository::SearchResult, Repository, SourceDeleteSummary};
use serde::Serialize;
use std::io::{self, Read, Write};
use std::path::PathBuf;

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

pub async fn run(repo: Repository, librarian: LibrarianAgent, command: NotesCommand) -> Result<()> {
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
) -> Result<()> {
    let existing = get_visible_note(&repo, &id).await?;
    let content = read_edit_content(content_file, stdin)?;
    if !has_edit_action(
        title.as_deref(),
        tags.as_deref(),
        content.as_deref(),
        detach,
    ) {
        bail!("notes edit requires --title, --tags, --content-file, --stdin, or --detach");
    }
    if content
        .as_deref()
        .is_some_and(|content| content.trim().is_empty())
    {
        bail!("note content cannot be empty");
    }

    let source_generated = existing.source_generation.is_some();
    if source_generated && !detach {
        bail!(
            "refusing to edit source-generated note {id} in place; use --detach to create a manual note that retains source provenance"
        );
    }

    let updated = if detach {
        librarian
            .detach_note_to_manual(
                &existing,
                content.unwrap_or_else(|| existing.content.clone()),
                title,
                tags,
            )
            .await?
    } else if let Some(content) = content {
        librarian
            .update_manual_note_content(&existing, content, title, tags)
            .await?
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
        (Some(path), false) => {
            Ok(Some(std::fs::read_to_string(&path).with_context(|| {
                format!("failed to read content file: {}", path.display())
            })?))
        }
        (None, true) => {
            let mut content = String::new();
            io::stdin().read_to_string(&mut content)?;
            Ok(Some(content))
        }
        (None, false) => Ok(None),
        (Some(_), true) => unreachable!("clap rejects conflicting content inputs"),
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
    use super::has_edit_action;

    #[test]
    fn detach_alone_is_an_edit_action() {
        assert!(has_edit_action(None, None, None, true));
        assert!(!has_edit_action(None, None, None, false));
    }
}
