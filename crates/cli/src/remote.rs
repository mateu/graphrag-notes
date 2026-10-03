//! Remote dispatch runs before local configuration, database and provider bootstrap.
use crate::cli::{Cli, Commands, GraphModeArg, SearchModeArg, SearchScopeArg};
use crate::commands::{
    editor::{Draft, EditorOptions, EditorOutcome},
    navigation::shell_quote,
};
use crate::output::{self, OutputFormat};
use anyhow::{Context, Result};
use graphrag_application::ApplicationError;
use rmcp::{
    model::{CallToolRequestParams, ClientConfig},
    transport::{
        streamable_http_client::StreamableHttpClientTransportConfig, StreamableHttpClientTransport,
    },
    ServiceExt,
};
use serde_json::{json, Value};
use std::io::Read;

fn endpoint(value: &str) -> Result<reqwest_mcp::Url> {
    let mut url = reqwest_mcp::Url::parse(value).context("invalid --server URL")?;
    if !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        anyhow::bail!("server URL must not contain credentials, query parameters or fragments");
    }
    let loopback = url.host_str().is_some_and(|host| {
        host == "localhost"
            || host
                .trim_matches(['[', ']'])
                .parse::<std::net::IpAddr>()
                .is_ok_and(|ip| ip.is_loopback())
    });
    if url.scheme() != "https" && !(url.scheme() == "http" && loopback) {
        anyhow::bail!("use HTTPS for a remote server, or HTTP through a loopback encrypted tunnel");
    }
    if url.path() == "/" {
        url.set_path("/mcp");
    }
    Ok(url)
}

fn scope(value: &SearchScopeArg) -> &'static str {
    match value {
        SearchScopeArg::Notes => "notes",
        SearchScopeArg::Messages => "messages",
        SearchScopeArg::All => "all",
    }
}
fn graph(value: &GraphModeArg) -> &'static str {
    match value {
        GraphModeArg::Off => "off",
        GraphModeArg::Auto => "auto",
        GraphModeArg::On => "on",
    }
}

struct Invocation {
    tool: &'static str,
    arguments: Value,
    format: OutputFormat,
    raw: bool,
    draft: Option<Draft>,
}

struct CaptureInput<'a> {
    content: Option<&'a str>,
    file: Option<&'a std::path::Path>,
    title: Option<&'a str>,
    tags: &'a [String],
    options: &'a EditorOptions,
    format: OutputFormat,
}

fn prepare_capture(cli: &Cli, server: &str, input: CaptureInput<'_>) -> Result<Invocation> {
    let CaptureInput {
        content,
        file,
        title,
        tags,
        options,
        format,
    } = input;
    let request_id = cli
        .request_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let directory = options.draft_dir.clone().unwrap_or_else(|| {
        std::env::var_os("HOME")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(std::env::temp_dir)
            .join(".graphrag/remote-drafts")
    });
    let directory = crate::commands::editor::recovery_directory(&directory)?;
    let bytes = if let Some(content) = content {
        content.as_bytes().to_vec()
    } else if let Some(file) = file {
        let mut bytes = Vec::new();
        std::fs::File::open(file)
            .context("could not read client capture file")?
            .take(65_537)
            .read_to_end(&mut bytes)?;
        bytes
    } else if options.editor {
        Vec::new()
    } else {
        let mut bytes = Vec::new();
        std::io::stdin().take(65_537).read_to_end(&mut bytes)?;
        bytes
    };
    if bytes.len() > 65_536 {
        anyhow::bail!("remote capture content must be at most 65536 bytes");
    }
    let mut draft = Draft::save(&bytes, &directory, |path| {
        let mut command = format!("graphrag --server {} --credential-env {} --request-id={} capture --content-file {} --format {}", shell_quote(server), shell_quote(&cli.credential_env), shell_quote(&request_id), shell_quote(&path.to_string_lossy()), crate::commands::editor::format_flag(format));
        if options.draft_dir.is_some() {
            command.push_str(&format!(
                " --draft-dir {}",
                shell_quote(&directory.to_string_lossy())
            ));
        }
        if let Some(title) = title {
            command.push_str(&format!(" --title={}", shell_quote(title)));
        }
        for tag in tags {
            command.push_str(&format!(" --tags={}", shell_quote(tag)));
        }
        command
    })?;
    let original = draft.read()?;
    let content = if options.editor {
        let (outcome, content) = draft.edit(options, &original)?;
        match outcome {
            EditorOutcome::Cancelled => anyhow::bail!(
                "capture editor cancelled; draft retained and no remote request was sent"
            ),
            EditorOutcome::Unchanged => {
                draft.discard(None);
                anyhow::bail!("capture editor unchanged; no remote request was sent");
            }
            EditorOutcome::Changed => {}
        }
        content
    } else {
        original
    };
    if content.trim().is_empty() || content.len() > 65_536 {
        anyhow::bail!("remote capture content must contain text and be at most 65536 bytes");
    }
    eprintln!(
        "Capture request ID: {}",
        output::safe_text(&request_id, false)
    );
    Ok(Invocation {
        tool: "capture_note",
        arguments: json!({"request_id":request_id,"content":content,"title":title,"tags":tags,"provenance":null}),
        format,
        raw: false,
        draft: Some(draft),
    })
}

fn invocation(cli: &Cli, server: &str) -> Result<Invocation> {
    if cli.request_id.is_some()
        && !matches!(cli.command, Commands::Capture { .. } | Commands::Add { .. })
    {
        anyhow::bail!("--request-id is only valid for remote capture/add");
    }
    let (tool, arguments, format, raw) = match &cli.command {
        Commands::Search { query, mode, limit, scope: selected_scope, since_days, source_uri, context, graph: selected_graph, format } => {
            if *context { anyhow::bail!("use remote augment to build context with citations"); }
            ("search_notes",json!({"query":query,"mode":match mode { SearchModeArg::Keyword => "keyword", SearchModeArg::Hybrid => "hybrid" },"scope":scope(selected_scope),"limit":limit.unwrap_or(10),"graph":graph(selected_graph),"since_days":since_days,"source_uri":source_uri}),*format,false)
        }
        Commands::Inspect { id, neighbors, revision, format } => ("get_record",json!({"id":id,"revision":revision,"neighbors":neighbors}),*format,false),
        Commands::Augment { query, limit, scope:selected_scope, since_days, source_uri, entity, max_tokens, max_chunk_tokens, graph:selected_graph, raw, format } => ("build_context",json!({"query":query,"scope":scope(selected_scope),"graph":graph(selected_graph),"since_days":since_days,"source_uri":source_uri,"entity_filter":entity,"max_chunks":limit,"max_total_tokens":max_tokens,"max_chunk_tokens":max_chunk_tokens}),*format,*raw),
        Commands::Capture { args } => return prepare_capture(cli,server,CaptureInput { content:args.content.as_deref(),file:args.content_file.as_deref(),title:args.title.as_deref(),tags:&args.tags,options:&args.editor,format:args.format }),
        Commands::Add { content,title,tags } => {
            let tags: Vec<String> = tags.as_deref().map(|tags|tags.split(',').map(|tag|tag.trim().to_string()).collect()).unwrap_or_default();
            return prepare_capture(cli,server,CaptureInput { content:content.as_deref(),file:None,title:title.as_deref(),tags:&tags,options:&EditorOptions::default(),format:OutputFormat::Human });
        }
        _ => anyhow::bail!("remote mode currently supports search, inspect, augment, capture and add; use host-side commands for database maintenance"),
    };
    Ok(Invocation {
        tool,
        arguments,
        format,
        raw,
        draft: None,
    })
}

pub(crate) async fn run(cli: &Cli, server: &str) -> Result<()> {
    if cli.db_path.is_some()
        || cli.memory
        || cli.config.is_some()
        || cli.concurrency.is_some()
        || cli.retry_attempts.is_some()
        || cli.no_cache
        || cli.explain
    {
        anyhow::bail!("remote mode uses host configuration; omit local database, config, inference and explain overrides");
    }
    let url = endpoint(server)?;
    let mut invocation = invocation(cli, url.as_str())?;
    let token = std::env::var(&cli.credential_env)
        .map_err(|_| anyhow::anyhow!("remote credential environment variable is missing"))?;
    if token.is_empty()
        || token.chars().any(char::is_whitespace)
        || token.chars().any(char::is_control)
    {
        anyhow::bail!("remote credential is invalid");
    }
    let http = reqwest_mcp::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .redirect(reqwest_mcp::redirect::Policy::none())
        .build()?;
    let transport = StreamableHttpClientTransport::with_client(
        http,
        StreamableHttpClientTransportConfig::with_uri(url.as_str().to_owned()).auth_header(token),
    );
    let client = ClientConfig::default().serve(transport).await.map_err(|_|ApplicationError::ServiceUnreachable("service_unreachable: connection or MCP authentication/negotiation failed; check endpoint and credential".into()))?;
    let mut request = CallToolRequestParams::new(invocation.tool);
    request.arguments = invocation.arguments.as_object().cloned();
    let response = client.call_tool(request).await;
    let _ = client.cancel().await;
    let response = response.map_err(|_|ApplicationError::ServiceUnreachable("service_unreachable: tool response was lost; retain the capture request ID and draft when retrying".into()))?;
    let value = response
        .structured_content
        .context("incompatible service: structured MCP result is missing")?;
    validate_envelope(&value)?;
    if response.is_error == Some(true) {
        let code = value
            .pointer("/error/code")
            .and_then(Value::as_str)
            .unwrap_or("remote_error");
        let message = value
            .pointer("/error/message")
            .and_then(Value::as_str)
            .unwrap_or("remote operation failed");
        let message = format!(
            "{}: {}",
            output::safe_text(code, false),
            output::safe_text(message, false)
        );
        return Err(match code {
            "invalid_input" | "validation" => ApplicationError::Validation(message),
            "not_found" => ApplicationError::NotFound(message),
            "revision_conflict" | "conflict" => ApplicationError::RevisionConflict(message),
            "compatibility" => ApplicationError::Compatibility(message),
            "provider_unavailable" => ApplicationError::ProviderUnavailable(message),
            "service_unreachable" => ApplicationError::ServiceUnreachable(message),
            "cancelled" => ApplicationError::Cancelled,
            _ => ApplicationError::Internal(message),
        }
        .into());
    }
    if !value["error"].is_null() {
        return Err(ApplicationError::Compatibility(
            "incompatible service: error result was not marked as a tool failure".into(),
        )
        .into());
    }
    validate_read_result(invocation.tool, &value)?;
    // Remove recovery only after the authoritative successful result arrived,
    // before rendering: a broken output pipe must not invite a new mutation.
    acknowledge_capture(&mut invocation, &value)?;
    if invocation.raw {
        let context = value
            .pointer("/data/rendered_context")
            .or_else(|| value.get("rendered_context"))
            .and_then(Value::as_str)
            .context("incompatible context response")?;
        print!("{context}");
        return Ok(());
    }
    let human = serde_json::to_string_pretty(&value)?;
    output::print(invocation.format, invocation.tool, value, |writer| {
        writeln!(writer, "{}", output::safe_text(&human, true))
    })
}

fn validate_envelope(value: &Value) -> Result<()> {
    if value.get("schema_version").and_then(Value::as_u64) != Some(1)
        || value.get("data").is_none()
        || value.get("error").is_none()
    {
        return Err(ApplicationError::Compatibility("incompatible service result schema; retain capture recovery until the original outcome is verified".into()).into());
    }
    Ok(())
}

fn validate_read_result(tool: &str, value: &Value) -> Result<()> {
    let valid = match tool {
        "search_notes" => value.pointer("/data/records").is_some_and(|records| {
            serde_json::from_value::<Vec<graphrag_application::RecordSummary>>(records.clone())
                .is_ok()
        }),
        "get_record" => {
            serde_json::from_value::<graphrag_db::RecordInspection>(value["data"].clone()).is_ok()
        }
        "build_context" => {
            serde_json::from_value::<graphrag_application::ContextResponse>(value["data"].clone())
                .is_ok()
        }
        "capture_note" => true,
        _ => false,
    };
    if !valid {
        return Err(
            ApplicationError::Compatibility("incompatible service result contract".into()).into(),
        );
    }
    Ok(())
}

fn validate_capture_result(value: &Value, arguments: &Value) -> Result<String> {
    validate_envelope(value)?;
    if !value["error"].is_null() {
        anyhow::bail!("capture did not return an authoritative success; retain recovery");
    }
    let result: graphrag_application::RemoteCaptureResponse =
        serde_json::from_value(value["data"].clone()).map_err(|_| {
            ApplicationError::Compatibility(
                "incompatible capture result; retain the original request ID and draft".into(),
            )
        })?;
    let submitted: graphrag_application::RemoteCaptureRequest =
        serde_json::from_value(arguments.clone()).context("capture request payload missing")?;
    // Capture preserves text and tags exactly; only an omitted title may be
    // generated by the server. Replays return the same original wire snapshot.
    if result.request_id != submitted.request_id
        || result.record.content != submitted.content
        || result.record.tags != submitted.tags
        || (submitted.title.is_some() && result.record.title != submitted.title)
        || !result.record.id.starts_with("note:")
        || graphrag_application::validate_record_id(&result.record.id).is_err()
        || result.record.revision.len() != 64
        || !result
            .record
            .revision
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(ApplicationError::Compatibility("capture result does not match its request identity or record contract; retain recovery".into()).into());
    }
    Ok(result.record.id)
}

fn acknowledge_capture(invocation: &mut Invocation, value: &Value) -> Result<()> {
    if let Some(draft) = &mut invocation.draft {
        let id = validate_capture_result(value, &invocation.arguments)?;
        draft.discard(Some(&id));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn endpoint_requires_encryption_without_accepting_url_secrets() {
        assert_eq!(endpoint("http://127.0.0.1:3000").unwrap().path(), "/mcp");
        assert!(endpoint("https://notes.example.test/mcp").is_ok());
        for url in [
            "http://192.168.1.4:3000/mcp",
            "https://user:token@notes.test/mcp",
            "https://notes.test/mcp?token=x",
            "ftp://localhost/mcp",
        ] {
            assert!(endpoint(url).is_err());
        }
    }

    #[test]
    fn incompatible_capture_results_cannot_discard_recovery() {
        let arguments = json!({"request_id":"original","content":"body","title":null,"tags":[],"provenance":null});
        for value in [
            json!({"schema_version":999,"data":null,"error":null}),
            json!({"schema_version":1,"data":null,"error":null}),
            json!({"schema_version":1,"data":{"request_id":"different","replayed":false,"record":{"id":"note:synthetic","revision":"a".repeat(64),"title":null,"content":"body","tags":[],"created_at":"2026-10-03T00:00:00Z","updated_at":"2026-10-03T00:00:00Z","provenance":null}},"error":null}),
        ] {
            assert!(validate_capture_result(&value, &arguments).is_err());
        }
    }

    #[test]
    fn capture_acknowledgment_matches_exact_payload_for_fresh_and_replayed_results() {
        let arguments = json!({"request_id":"original","content":"exact\nbody","title":"Explicit title","tags":[" tag ","duplicate","duplicate"],"provenance":null});
        let mut valid = json!({"schema_version":1,"data":{"request_id":"original","replayed":false,"record":{"id":"note:synthetic","revision":"a".repeat(64),"title":"Explicit title","content":"exact\nbody","tags":[" tag ","duplicate","duplicate"],"created_at":"2026-10-03T00:00:00Z","updated_at":"2026-10-03T00:00:00Z","provenance":null}},"error":null});
        for replayed in [false, true] {
            valid["data"]["replayed"] = json!(replayed);
            assert_eq!(
                validate_capture_result(&valid, &arguments).unwrap(),
                "note:synthetic"
            );
            for (field, wrong) in [
                ("content", json!("different body")),
                ("tags", json!(["tag", "duplicate"])),
                ("title", json!("Different title")),
            ] {
                let mut incompatible = valid.clone();
                incompatible["data"]["record"][field] = wrong;
                let directory = tempfile::tempdir().unwrap();
                let draft =
                    Draft::save(b"exact\nbody", directory.path(), |_| "recovery".into()).unwrap();
                let path = draft.path.clone();
                let mut invocation = Invocation {
                    tool: "capture_note",
                    arguments: arguments.clone(),
                    format: OutputFormat::Json,
                    raw: false,
                    draft: Some(draft),
                };
                assert!(
                    acknowledge_capture(&mut invocation, &incompatible).is_err(),
                    "{field}, replayed={replayed}"
                );
                drop(invocation);
                assert_eq!(
                    std::fs::read(&path).unwrap(),
                    b"exact\nbody",
                    "mismatched acknowledgment must retain the only recovery draft"
                );
            }
        }
        let mut generated_title = arguments;
        generated_title["title"] = Value::Null;
        valid["data"]["record"]["title"] = json!("Generated title");
        assert!(validate_capture_result(&valid, &generated_title).is_ok());
    }
}
