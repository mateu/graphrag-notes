//! Provider-first manual capture with private drafts retained until commit.
use super::editor::{self, Draft, EditorOptions, EditorOutcome};
use super::navigation::shell_quote;
use crate::output::{self, OutputFormat};
use anyhow::Result;
use clap::Args;
use graphrag_agents::LibrarianAgent;
use graphrag_core::record_id_to_string;
use serde::Serialize;
use std::io::Read;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub struct CaptureValidationError(pub String);
impl std::fmt::Display for CaptureValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for CaptureValidationError {}

#[derive(Args)]
pub struct CaptureArgs {
    /// Note body; otherwise read multiline stdin, or start a blank editor draft.
    #[arg(conflicts_with = "content_file")]
    pub content: Option<String>,
    #[arg(long, value_name = "PATH")]
    pub content_file: Option<PathBuf>,
    /// Read multiline stdin explicitly (also the default without editor/content/file).
    #[arg(long, conflicts_with_all = ["content", "content_file"])]
    pub stdin: bool,
    #[arg(short, long)]
    pub title: Option<String>,
    #[arg(short = 'T', long, value_delimiter = ',')]
    pub tags: Vec<String>,
    #[command(flatten)]
    pub editor: EditorOptions,
    #[arg(long, value_enum, default_value_t = OutputFormat::Human)]
    pub format: OutputFormat,
}

pub struct PreparedCapture {
    pub content: String,
    pub draft: Draft,
    pub outcome: EditorOutcome,
    pub base_command: String,
}

pub fn prepare(
    args: &CaptureArgs,
    config: Option<&Path>,
    database: &Path,
) -> Result<PreparedCapture> {
    let base = editor::base_command(config, database)?;
    let directory = editor::directory(&args.editor, database)?;
    let bytes = if let Some(content) = &args.content {
        content.as_bytes().to_vec()
    } else if let Some(path) = &args.content_file {
        std::fs::read(path).map_err(|error| {
            CaptureValidationError(format!(
                "failed to read content file {}: {error}",
                path.display()
            ))
        })?
    } else if args.editor.editor {
        Vec::new()
    } else {
        let mut bytes = Vec::new();
        std::io::stdin().read_to_end(&mut bytes)?;
        bytes
    };
    let draft = Draft::save(&bytes, &directory, |path| {
        let mut command = format!(
            "{base} capture --content-file {}",
            shell_quote(&path.to_string_lossy())
        );
        if let Some(title) = &args.title {
            command.push_str(&format!(" --title={}", shell_quote(title)));
        }
        for tag in &args.tags {
            command.push_str(&format!(" --tags={}", shell_quote(tag)));
        }
        command.push_str(&format!(" --format {}", editor::format_flag(args.format)));
        if args.editor.draft_dir.is_some() {
            command.push_str(&format!(
                " --draft-dir={}",
                shell_quote(&directory.to_string_lossy())
            ));
        }
        command
    })?;
    let original = draft.read()?;
    let (outcome, content) = if args.editor.editor {
        draft.edit(&args.editor, &original)?
    } else {
        (EditorOutcome::Changed, original)
    };
    if outcome == EditorOutcome::Changed && content.trim().is_empty() {
        return Err(CaptureValidationError("note content cannot be empty".into()).into());
    }
    Ok(PreparedCapture {
        content,
        draft,
        outcome,
        base_command: base,
    })
}

#[derive(Serialize)]
struct CaptureOutput<'a> {
    status: &'a str,
    id: Option<String>,
    draft_path: Option<&'a Path>,
    recovery_command: Option<&'a str>,
}

pub fn print_noop(prepared: &mut PreparedCapture, format: OutputFormat) -> Result<()> {
    let cancelled = prepared.outcome == EditorOutcome::Cancelled;
    let retained = cancelled || !prepared.draft.discard(None);
    let data = CaptureOutput {
        status: if cancelled { "cancelled" } else { "unchanged" },
        id: None,
        draft_path: retained.then_some(prepared.draft.path.as_path()),
        recovery_command: if cancelled {
            prepared.draft.recoverable_command()
        } else {
            None
        },
    };
    output::print(format, "capture", data, |writer| {
        writeln!(
            writer,
            "Capture {}; no note saved.",
            if cancelled { "cancelled" } else { "unchanged" }
        )
    })?;
    Ok(())
}

pub async fn run(
    librarian: &LibrarianAgent,
    args: CaptureArgs,
    mut prepared: PreparedCapture,
) -> Result<()> {
    let note = librarian
        .capture_manual_note(prepared.content.clone(), args.title, args.tags)
        .await?;
    let id = record_id_to_string(note.id.as_ref().expect("persisted note has id"));
    let removed = prepared.draft.discard(Some(&id));
    output::print(
        args.format,
        "capture",
        CaptureOutput {
            status: "created",
            id: Some(id.clone()),
            draft_path: (!removed).then_some(prepared.draft.path.as_path()),
            recovery_command: None,
        },
        |writer| {
            writeln!(writer, "Captured note {id}")?;
            writeln!(
                writer,
                "Inspect: {} inspect {}",
                prepared.base_command,
                shell_quote(&id)
            )
        },
    )
}
