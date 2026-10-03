//! Remote writes preserve the reviewed revision and original request identity.
use crate::cli::{Cli, Commands, GardenCommand, ProposalCommand};
use crate::commands::{
    editor::{Draft, EditorOutcome},
    navigation::shell_quote,
    notes::NotesCommand,
};
use crate::output::{self, OutputFormat};
use crate::remote::{call_tool, Invocation};
use anyhow::{Context, Result};
use graphrag_application::{ApplicationError, RemoteMutationResponse, RemoteNoteSnapshot};
use serde_json::{json, Value};
use std::io::Read;

fn revision(cli: &Cli) -> Result<&str> {
    let revision = cli
        .expected_revision
        .as_deref()
        .context("remote writes require --expected-revision from notes show or garden review")?;
    if revision.len() != 64
        || !revision
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        anyhow::bail!(
            "--expected-revision must be the 64-character revision returned by the service"
        );
    }
    Ok(revision)
}
fn request_id(cli: &Cli) -> String {
    let id = cli
        .request_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    eprintln!("Mutation request ID: {}", output::safe_text(&id, false));
    id
}
fn read(tool: &'static str, arguments: Value, format: OutputFormat) -> Invocation {
    Invocation {
        tool,
        arguments,
        format,
        raw: false,
        draft: None,
    }
}
fn status(value: crate::cli::ProposalStatusArg) -> &'static str {
    use crate::cli::ProposalStatusArg::*;
    match value {
        Pending => "pending",
        Accepting => "accepting",
        Accepted => "accepted",
        Rejected => "rejected",
        Superseded => "superseded",
    }
}

pub(super) async fn prepare(cli: &Cli, server: &str) -> Result<Option<Invocation>> {
    let invocation = match &cli.command {
        Commands::Notes {
            command: NotesCommand::Show { id, format },
        } => read("get_note", json!({"id":id,"revision":null}), *format),
        Commands::Notes {
            command:
                NotesCommand::Delete {
                    id,
                    dry_run,
                    yes,
                    format,
                },
        } => {
            if *dry_run {
                read("get_note", json!({"id":id,"revision":null}), *format)
            } else {
                if !yes {
                    anyhow::bail!("remote deletion requires --yes after reviewing notes show");
                }
                let revision = revision(cli)?;
                read(
                    "delete_note",
                    json!({"request_id":request_id(cli),"id":id,"revision":revision,"confirmed":true}),
                    *format,
                )
            }
        }
        Commands::Notes {
            command: command @ NotesCommand::Edit { .. },
        } => return Ok(Some(prepare_edit(cli, server, command).await?)),
        Commands::Garden {
            command:
                GardenCommand::Review {
                    id,
                    status: selected,
                    all_statuses,
                    limit,
                    interactive,
                    format,
                },
        } => {
            if *interactive {
                anyhow::bail!("remote review uses garden review followed by an explicit proposal accept/reject/undo with --expected-revision and --yes");
            }
            if let Some(id) = id {
                read("get_proposal", json!({"id":id}), *format)
            } else {
                read(
                    "list_proposals",
                    json!({"status":if *all_statuses { None } else { Some(selected.map(status).unwrap_or("pending")) },"limit":limit}),
                    *format,
                )
            }
        }
        Commands::Garden {
            command: GardenCommand::Proposals { command },
        } => match command {
            ProposalCommand::List {
                status: selected,
                limit,
            } => read(
                "list_proposals",
                json!({"status":selected.map(status),"limit":limit}),
                OutputFormat::Human,
            ),
            ProposalCommand::Show { id } => {
                read("get_proposal", json!({"id":id}), OutputFormat::Human)
            }
            ProposalCommand::Accept {
                id,
                all,
                min_confidence,
                yes,
                reason,
            } => {
                if *all || min_confidence.is_some() {
                    anyhow::bail!("remote acceptance selects one inspected proposal; batch policy changes remain host-side");
                }
                decision(
                    cli,
                    id.as_deref()
                        .context("provide a canonical proposed_edge:ID")?,
                    "accept",
                    reason,
                    *yes,
                )?
            }
            ProposalCommand::Reject { id, reason, yes } => {
                decision(cli, id, "reject", reason, *yes)?
            }
            ProposalCommand::Undo { id, reason, yes } => decision(cli, id, "undo", reason, *yes)?,
        },
        _ => return Ok(None),
    };
    if cli.recover_draft && invocation.tool != "edit_note" {
        anyhow::bail!("--recover-draft requires a remote edit with --content-file");
    }
    if !matches!(
        invocation.tool,
        "edit_note" | "delete_note" | "decide_proposal"
    ) && (cli.request_id.is_some() || cli.expected_revision.is_some())
    {
        anyhow::bail!("--request-id/--expected-revision are only valid for remote writes");
    }
    Ok(Some(invocation))
}
fn decision(
    cli: &Cli,
    id: &str,
    action: &str,
    reason: &Option<String>,
    yes: bool,
) -> Result<Invocation> {
    if !yes {
        anyhow::bail!("remote proposal decisions require --yes after reviewing the proposal card");
    }
    let revision = revision(cli)?;
    Ok(read(
        "decide_proposal",
        json!({"request_id":request_id(cli),"id":id,"revision":revision,"action":action,"reason":reason,"confirmed":true}),
        OutputFormat::Human,
    ))
}
async fn prepare_edit(cli: &Cli, server: &str, command: &NotesCommand) -> Result<Invocation> {
    let NotesCommand::Edit {
        id,
        title,
        content_file,
        stdin,
        editor,
        tags,
        detach,
        format,
    } = command
    else {
        unreachable!()
    };
    if *detach {
        anyhow::bail!(
            "remote edits support manual notes; source-owned notes must be changed at their source"
        );
    }
    let recovery_path = if cli.recover_draft {
        let path = content_file
            .as_deref()
            .context("--recover-draft requires --content-file")?;
        Draft::validate_recovery_path(path)?;
        Some(path)
    } else {
        None
    };
    let revision = revision(cli)?;
    let request_id = request_id(cli);
    let mut invocation = read(
        "edit_note",
        json!({"request_id":request_id,"id":id,"revision":revision,"patch":{"content":null,"title":title,"clear_title":false,"tags":tags}}),
        *format,
    );
    if !editor.editor && !stdin && content_file.is_none() {
        if title.is_none() && tags.is_none() {
            anyhow::bail!("provide --title, --tags, --content-file, --stdin or --editor");
        }
        return Ok(invocation);
    }
    let directory = editor.draft_dir.clone().unwrap_or_else(|| {
        std::env::var_os("HOME")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(std::env::temp_dir)
            .join(".graphrag/remote-drafts")
    });
    let directory = crate::commands::editor::recovery_directory(&directory)?;
    let original = if editor.editor {
        let response = call_tool(
            cli,
            server,
            "get_note",
            json!({"id":id,"revision":revision}),
        )
        .await?;
        let snapshot: RemoteNoteSnapshot = serde_json::from_value(response["data"].clone())
            .map_err(|_| {
                ApplicationError::Compatibility(
                    "incompatible note snapshot; no editor was opened".into(),
                )
            })?;
        if snapshot.id != *id || snapshot.revision != revision || !snapshot.editable {
            anyhow::bail!(
                "remote note is stale or source-owned; refresh notes show before editing"
            );
        }
        snapshot.content.into_bytes()
    } else {
        let mut bytes = Vec::new();
        if let Some(path) = content_file {
            if recovery_path.is_some() {
                bytes = Draft::read_recovery_input(path, 65_537)?;
            } else {
                std::fs::File::open(path)
                    .context("could not read client edit file")?
                    .take(65_537)
                    .read_to_end(&mut bytes)?;
            }
        } else {
            std::io::stdin().take(65_537).read_to_end(&mut bytes)?;
        }
        bytes
    };
    if original.len() > 65_536 {
        anyhow::bail!("remote edit content must be at most 65536 UTF-8 bytes");
    }
    let mut draft = Draft::save_or_recover(&original, &directory, recovery_path, |path| {
        let mut retry = format!("graphrag --server {} --credential-env {} --request-id={} --expected-revision {} --recover-draft notes edit {} --content-file {} --format {}", shell_quote(server), shell_quote(&cli.credential_env), shell_quote(&request_id), shell_quote(revision), shell_quote(id), shell_quote(&path.to_string_lossy()), crate::commands::editor::format_flag(*format));
        if editor.draft_dir.is_some() {
            retry.push_str(&format!(
                " --draft-dir {}",
                shell_quote(&directory.to_string_lossy())
            ));
        }
        if let Some(title) = title {
            retry.push_str(&format!(" --title={}", shell_quote(title)));
        }
        if let Some(tags) = tags {
            for tag in tags {
                retry.push_str(&format!(" --tags={}", shell_quote(tag)));
            }
        }
        retry
    })?;
    let original = draft.read()?;
    let content = if editor.editor {
        let (outcome, content) = draft.edit(editor, &original)?;
        match outcome {
            EditorOutcome::Cancelled => {
                anyhow::bail!("edit editor cancelled; draft retained and no remote edit was sent")
            }
            EditorOutcome::Unchanged if title.is_none() && tags.is_none() => {
                draft.discard(None);
                anyhow::bail!("edit editor unchanged; no remote edit was sent");
            }
            EditorOutcome::Unchanged => content,
            EditorOutcome::Changed => content,
        }
    } else {
        original
    };
    if content.trim().is_empty() || content.len() > 65_536 {
        anyhow::bail!("remote edit content must contain text and be at most 65536 UTF-8 bytes");
    }
    draft.pin_remote_submission(content.as_bytes())?;
    invocation.arguments["patch"]["content"] = json!(content);
    invocation.draft = Some(draft);
    Ok(invocation)
}

pub(super) fn validate_result(invocation: &Invocation, value: &Value) -> Result<()> {
    let (operation, status) = match invocation.tool {
        "edit_note" => ("edit", "edited"),
        "delete_note" => ("delete", "deleted"),
        "decide_proposal" => match invocation.arguments["action"].as_str() {
            Some("accept") => ("accept", "accepted"),
            Some("reject") => ("reject", "rejected"),
            Some("undo") => ("undo", "superseded"),
            _ => anyhow::bail!("unknown proposal action"),
        },
        _ => return Ok(()),
    };
    let response: RemoteMutationResponse = serde_json::from_value(value["data"].clone()).map_err(|_| ApplicationError::Compatibility("incompatible mutation result; retain original request ID, revision and recovery draft".into()))?;
    if Some(response.request_id.as_str()) != invocation.arguments["request_id"].as_str()
        || Some(response.outcome.id.as_str()) != invocation.arguments["id"].as_str()
        || Some(response.outcome.previous_revision.as_str())
            != invocation.arguments["revision"].as_str()
        || response.outcome.operation != operation
        || response.outcome.status != status
        || !response.outcome.actor.starts_with("mcp:")
        || response.outcome.actor.len() <= 4
    {
        return Err(ApplicationError::Compatibility("mutation response does not match its request; retain original request ID, revision and recovery draft".into()).into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mutation_acknowledgements_must_match_the_original_write() {
        let invocation = read(
            "edit_note",
            json!({"request_id":"retry-001","id":"note:synthetic","revision":"a".repeat(64)}),
            OutputFormat::Json,
        );
        let valid = json!({"schema_version":1,"data":{"request_id":"retry-001","replayed":false,"outcome":{"id":"note:synthetic","operation":"edit","previous_revision":"a".repeat(64),"actor":"mcp:synthetic","status":"edited","resulting_edge_id":null,"cascade":null}},"error":null});
        assert!(validate_result(&invocation, &valid).is_ok());
        for pointer in [
            "/data/request_id",
            "/data/outcome/id",
            "/data/outcome/previous_revision",
            "/data/outcome/operation",
            "/data/outcome/status",
            "/data/outcome/actor",
        ] {
            let mut incompatible = valid.clone();
            *incompatible.pointer_mut(pointer).unwrap() = json!("different");
            assert!(
                validate_result(&invocation, &incompatible).is_err(),
                "{pointer}"
            );
        }
    }
}
