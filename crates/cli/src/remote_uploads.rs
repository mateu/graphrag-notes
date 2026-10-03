//! Client-side uploaded Markdown and owned job commands. No local DB bootstrap.
use crate::{
    cli::{Cli, Commands, JobOutputFormat, JobsCommand, SourceOutputFormat, SourcesCommand},
    commands::{
        editor::{format_flag, Draft},
        navigation::shell_quote,
    },
    output::{self, OutputFormat},
    remote::Invocation,
};
use anyhow::{Context, Result};
use graphrag_application::{
    ApplicationError, RemoteJobList, RemoteJobStatus, UploadAdmission, UploadedSource,
};
use serde_json::{json, Value};
use std::{io::Read, path::Path};

fn job_format(format: JobOutputFormat) -> OutputFormat {
    match format {
        JobOutputFormat::Human => OutputFormat::Human,
        JobOutputFormat::Json => OutputFormat::Json,
    }
}
fn source_format(format: SourceOutputFormat) -> OutputFormat {
    match format {
        SourceOutputFormat::Human => OutputFormat::Human,
        SourceOutputFormat::Json => OutputFormat::Json,
    }
}
fn path_text(path: &Path) -> Result<&str> {
    path.to_str().context("recovery paths must be valid UTF-8")
}

pub(super) fn prepare(cli: &Cli, server: &str) -> Result<Option<Invocation>> {
    let (tool, arguments, format) = match &cli.command {
        Commands::Upload {
            document_key,
            content_file,
            title,
            extract_entities,
            draft_dir,
            format,
        } => {
            let request = cli
                .request_id
                .clone()
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
            for (name, value, maximum) in [
                ("request ID", request.as_str(), 256),
                ("document key", document_key.as_str(), 512),
            ] {
                if value.trim().is_empty()
                    || value.trim() != value
                    || value.len() > maximum
                    || value.chars().any(char::is_control)
                {
                    anyhow::bail!("{name} must be bounded, nonempty text without control characters or surrounding whitespace");
                }
            }
            let directory = draft_dir.clone().unwrap_or_else(|| {
                std::env::var_os("HOME")
                    .map(std::path::PathBuf::from)
                    .unwrap_or_else(std::env::temp_dir)
                    .join(".graphrag/remote-drafts")
            });
            let directory_text = path_text(&directory)?.to_owned();
            let mut bytes = Vec::new();
            std::fs::File::open(content_file)
                .context("could not read client upload file")?
                .take(65_537)
                .read_to_end(&mut bytes)?;
            if bytes.len() > 65_536 {
                anyhow::bail!("uploaded Markdown must be at most 65536 UTF-8 bytes");
            }
            let draft = Draft::save(&bytes, &directory, |path| {
                let mut command=format!("graphrag --server={} --credential-env={} --request-id={} upload --document-key={} --content-file={} --draft-dir={} --format {}",shell_quote(server),shell_quote(&cli.credential_env),shell_quote(&request),shell_quote(document_key),shell_quote(&path.to_string_lossy()),shell_quote(&directory_text),format_flag(*format));
                if let Some(title) = title {
                    command.push_str(&format!(" --title={}", shell_quote(title)));
                }
                if *extract_entities {
                    command.push_str(" --extract-entities");
                }
                command
            })?;
            let content = draft.read()?;
            if content.trim().is_empty() || content.contains('\0') {
                anyhow::bail!(
                    "uploaded Markdown must contain text without NUL; private draft retained"
                );
            }
            eprintln!(
                "Upload request ID: {}\nDocument key: {}",
                output::safe_text(&request, false),
                output::safe_text(document_key, false)
            );
            return Ok(Some(Invocation {
                tool: "upload_source",
                arguments: json!({"request_id":request,"document_key":document_key,"content":content,"title":title,"provenance":null,"extract_entities":extract_entities}),
                format: *format,
                raw: false,
                draft: Some(draft),
            }));
        }
        Commands::Jobs { command } => {
            if cli.request_id.is_some() {
                anyhow::bail!("job control reuses the existing job ID; omit --request-id");
            }
            match command {
                JobsCommand::List { limit, format } => {
                    ("list_jobs", json!({"limit":limit}), job_format(*format))
                }
                JobsCommand::Show { id, format } => {
                    ("get_job", json!({"id":id}), job_format(*format))
                }
                JobsCommand::Cancel { id } => ("cancel_job", json!({"id":id}), OutputFormat::Human),
                JobsCommand::Resume { id } => ("resume_job", json!({"id":id}), OutputFormat::Human),
            }
        }
        Commands::Sources {
            command: SourcesCommand::Show { id_or_uri, format },
        } => {
            if cli.request_id.is_some() {
                anyhow::bail!("source inspection does not take --request-id");
            }
            (
                "get_source",
                json!({"id":id_or_uri}),
                source_format(*format),
            )
        }
        _ => return Ok(None),
    };
    Ok(Some(Invocation {
        tool,
        arguments,
        format,
        raw: false,
        draft: None,
    }))
}

pub(super) fn valid_read_result(tool: &str, value: &Value) -> bool {
    match tool {
        "upload_source" => serde_json::from_value::<UploadAdmission>(value["data"].clone()).is_ok(),
        "get_source" => serde_json::from_value::<UploadedSource>(value["data"].clone()).is_ok(),
        "list_jobs" => serde_json::from_value::<RemoteJobList>(value["data"].clone()).is_ok(),
        "get_job" | "cancel_job" | "resume_job" => {
            serde_json::from_value::<RemoteJobStatus>(value["data"].clone()).is_ok()
        }
        _ => false,
    }
}

pub(super) fn validate_result(invocation: &Invocation, value: &Value) -> Result<String> {
    let result: UploadAdmission = serde_json::from_value(value["data"].clone()).map_err(|_| {
        ApplicationError::Compatibility(
            "incompatible upload admission; retain the original request ID, document key and draft"
                .into(),
        )
    })?;
    if result.request_id
        != invocation.arguments["request_id"]
            .as_str()
            .unwrap_or_default()
        || !canonical_hash_id(&result.job_id, "processing_job")
        || !canonical_hash_id(&result.source_id, "source")
        || !result.source_uri.starts_with("mcp://upload/")
    {
        return Err(ApplicationError::Compatibility("upload admission does not match the request or stable identity contract; retain recovery".into()).into());
    }
    Ok(result.job_id)
}
fn canonical_hash_id(id: &str, table: &str) -> bool {
    id.strip_prefix(&format!("{table}:")).is_some_and(|key| {
        key.len() == 64
            && key
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn upload_acknowledgment_must_match_request_before_discarding_draft() {
        let invocation = Invocation {
            tool: "upload_source",
            arguments: json!({"request_id":"attempt-1"}),
            format: OutputFormat::Json,
            raw: false,
            draft: None,
        };
        let mut data = json!({"request_id":"attempt-1","job_id":format!("processing_job:{}","a".repeat(64)),"source_id":format!("source:{}","b".repeat(64)),"source_uri":format!("mcp://upload/{}","b".repeat(64)),"replayed":false});
        assert!(validate_result(&invocation, &json!({"data":data})).is_ok());
        data["request_id"] = json!("different");
        assert!(validate_result(&invocation, &json!({"data":data})).is_err());
        data["request_id"] = json!("attempt-1");
        data["job_id"] = json!("processing_job:wrong");
        assert!(validate_result(&invocation, &json!({"data":data})).is_err());
    }
}
