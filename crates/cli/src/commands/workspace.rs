//! Local interaction adapters around shared application operations.

use super::{capture as capture_command, editor, navigation};
use crate::output::OutputFormat;
use anyhow::Result;
use graphrag_application::{
    ActionCancellation, ApplicationError, ApplicationOperations, CaptureRequest, RecordRef,
};
use graphrag_config::RuntimeConfig;
use graphrag_db::Repository;
use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::sync::Arc;

pub(crate) struct WorkspaceContext {
    pub(crate) application: Arc<dyn ApplicationOperations>,
    pub(crate) repo: Repository,
    pub(crate) config: RuntimeConfig,
    pub(crate) config_path: Option<PathBuf>,
    pub(crate) memory: bool,
}

pub(crate) async fn capture(
    context: &WorkspaceContext,
    input: &str,
    cancellation: ActionCancellation,
) -> Result<()> {
    if context.memory {
        anyhow::bail!("recoverable capture requires persistent storage; restart without --memory");
    }
    if input.trim().is_empty() {
        anyhow::bail!("Use capture <text> or capture --editor for a multiline private draft.");
    }
    let editor = input.trim() == "--editor";
    let args = capture_command::CaptureArgs {
        content: (!editor).then(|| input.to_string()),
        content_file: None,
        stdin: false,
        title: None,
        tags: Vec::new(),
        editor: editor::EditorOptions {
            editor,
            ..Default::default()
        },
        format: OutputFormat::Human,
    };
    let config = context.config_path.clone();
    let database = context.config.database.path.clone();
    // A blocking editor must not prevent signal handling or hold the async
    // runtime thread while the user is working on their retained draft.
    let (args, mut prepared) = tokio::task::spawn_blocking(move || {
        let prepared = capture_command::prepare(&args, config.as_deref(), &database)?;
        Ok::<_, anyhow::Error>((args, prepared))
    })
    .await??;
    if prepared.outcome != editor::EditorOutcome::Changed {
        return capture_command::print_noop(&mut prepared, args.format);
    }
    if cancellation.is_cancelled() {
        return Err(ApplicationError::Cancelled.into());
    }
    let note = context
        .application
        .capture(
            CaptureRequest {
                content: prepared.content.clone(),
                title: args.title,
                tags: args.tags,
            },
            cancellation,
        )
        .await?;
    capture_command::finish(args.format, prepared, note)
}

pub(crate) async fn open(context: &WorkspaceContext, reference: RecordRef) -> Result<()> {
    navigation::open(
        &context.repo,
        &reference.id,
        reference.revision.as_deref(),
        None,
        &context.config.navigation.opener,
        OutputFormat::Human,
    )
    .await
}

pub(crate) async fn copy(context: &WorkspaceContext, reference: RecordRef) -> Result<()> {
    let record = context.application.inspect(reference, 0).await?;
    // Portable terminal reuse: selected content on stdout; citation on stderr.
    // Clipboard programs are deliberately optional client integrations.
    let mut stdout = std::io::stdout().lock();
    let content = if stdout.is_terminal() {
        crate::output::safe_text(&record.content, true)
    } else {
        record.content.clone()
    };
    stdout.write_all(content.as_bytes())?;
    if !record.content.ends_with('\n') {
        writeln!(stdout)?;
    }
    stdout.flush()?;
    eprintln!(
        "Copied context: id={} revision={}",
        serde_json::to_string(&record.id)?,
        serde_json::to_string(&record.revision)?
    );
    Ok(())
}
