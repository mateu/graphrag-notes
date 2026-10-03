//! Provider-free inspection and explicit, argument-safe local source opening.

use crate::output::{self, OutputEnvelope, OutputFormat};
use anyhow::{Context, Result};
use graphrag_db::{repository::RecordInspection, Repository};
use serde::Serialize;
use std::collections::HashMap;
use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use tracing::warn;

#[derive(Debug)]
pub(crate) enum NavigationError {
    Validation(String),
    NotFound(String),
}

impl fmt::Display for NavigationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Validation(message) | Self::NotFound(message) => f.write_str(message),
        }
    }
}
impl std::error::Error for NavigationError {}

pub(crate) fn validate_id(id: &str) -> Result<()> {
    graphrag_application::validate_record_id(id)
        .map_err(|error| NavigationError::Validation(error.to_string()).into())
}

pub(crate) fn validate_neighbors(neighbors: usize) -> Result<()> {
    graphrag_application::validate_neighbors(neighbors)
        .map_err(|error| NavigationError::Validation(error.to_string()).into())
}

fn check_revision(record: &RecordInspection, expected: Option<&str>) -> Result<()> {
    graphrag_application::validate_record_revision(record, expected)
        .map_err(|error| NavigationError::Validation(error.to_string()).into())
}

pub(crate) async fn inspect(
    repo: &Repository,
    id: &str,
    neighbors: usize,
    revision: Option<&str>,
    format: OutputFormat,
) -> Result<()> {
    validate_id(id)?;
    validate_neighbors(neighbors)?;
    let record = repo.inspect_record(id, neighbors).await?;
    check_revision(&record, revision)?;
    let mut envelope = OutputEnvelope::success("inspect", &record);
    envelope.warnings = record.warnings.clone();
    output::print_envelope(format, &envelope, |writer| {
        render_inspection(writer, &record)
    })
}

pub(crate) fn render_inspection(
    writer: &mut dyn Write,
    record: &RecordInspection,
) -> std::io::Result<()> {
    writeln!(
        writer,
        "[{}] {}",
        record.hit_type,
        readable_title(record.title.as_deref(), &record.content)
    )?;
    writeln!(writer, "ID: {}", record.id)?;
    if let Some(uri) = &record.provenance.source_uri {
        writeln!(writer, "Source: {uri}")?;
    }
    if !record.provenance.heading_path.is_empty() {
        writeln!(
            writer,
            "Heading: {}",
            record.provenance.heading_path.join(" > ")
        )?;
    }
    if let Some(line) = record.provenance.start_line {
        writeln!(
            writer,
            "Lines: {line}-{} (at last import)",
            record.provenance.end_line.unwrap_or(line)
        )?;
    }
    if let Some(uuid) = &record.provenance.conversation_uuid {
        writeln!(writer, "Conversation UUID: {uuid}")?;
    }
    if let Some(uuid) = &record.provenance.message_uuid {
        writeln!(writer, "Message UUID: {uuid}")?;
    } else if let Some(key) = &record.provenance.message_key {
        writeln!(writer, "Message key: {key}")?;
    }
    if let Some(index) = record.provenance.message_index {
        writeln!(
            writer,
            "Message #: {} ({})",
            index + 1,
            record.provenance.role.as_deref().unwrap_or("unknown role")
        )?;
    }
    writeln!(writer, "\n{}", record.content)?;
    for conversation in &record.conversations {
        writeln!(
            writer,
            "\nConversation: {}\n  ID: {}\n  UUID: {}",
            conversation
                .title
                .as_deref()
                .unwrap_or("(untitled conversation)"),
            conversation.id,
            conversation.uuid
        )?;
    }
    if !record.messages.is_empty() {
        writeln!(writer, "\nChat context:")?;
    }
    for message in &record.messages {
        let focus = if message.id == record.id {
            " [selected]"
        } else {
            ""
        };
        writeln!(
            writer,
            "  #{} [{}] {}{focus}",
            message.message_index + 1,
            message.role,
            message.id,
        )?;
        if let Some(uuid) = &message.message_uuid {
            writeln!(writer, "    Message UUID: {uuid}")?;
        } else {
            writeln!(writer, "    Message key: {}", message.message_key)?;
        }
        writeln!(writer, "{}", message.content)?;
    }
    if record.messages_truncated {
        writeln!(
            writer,
            "Additional messages are omitted; use --neighbors up to 20 or inspect a message ID."
        )?;
    }
    for warning in &record.warnings {
        writeln!(writer, "Warning: {warning}")?;
    }
    Ok(())
}

#[derive(Serialize)]
struct OpenOutput<'a> {
    id: &'a str,
    path: PathBuf,
    launched: bool,
    opener: Vec<String>,
    revision: &'a str,
    provenance: &'a graphrag_db::repository::InspectionProvenance,
}

/// Stored file URIs contain literal paths, including # and %, rather than URL
/// encoding. Decoding those characters would open a different file.
fn source_path(uri: &str) -> Result<PathBuf> {
    let literal = uri.strip_prefix("file://").unwrap_or(uri);
    let path = PathBuf::from(literal);
    if !path.is_absolute() || (uri.contains("://") && !uri.starts_with("file://")) {
        return Err(NavigationError::Validation(
            "This result has no local source file. Use inspect to read its stored content and chat context.".into(),
        ).into());
    }
    Ok(path)
}

fn opener_argv(explicit: Option<&Path>, configured: &[String]) -> Result<Vec<String>> {
    let argv = if let Some(path) = explicit {
        vec![path
            .to_str()
            .context("opener path must be valid UTF-8")?
            .to_string()]
    } else if !configured.is_empty() {
        configured.to_vec()
    } else if let Some(editor) = std::env::var_os("VISUAL").or_else(|| std::env::var_os("EDITOR")) {
        vec![editor.into_string().map_err(|_| {
            NavigationError::Validation("editor executable path must be valid UTF-8".into())
        })?]
    } else if cfg!(target_os = "macos") {
        vec!["open".into()]
    } else if cfg!(target_os = "linux") {
        vec!["xdg-open".into()]
    } else {
        return Err(NavigationError::Validation("Configure navigation.opener or pass --opener with an executable path on this platform.".into()).into());
    };
    validate_argv(&argv)?;
    Ok(argv)
}

pub(crate) fn validate_argv(argv: &[String]) -> Result<()> {
    if argv.first().is_none_or(|value| value.trim().is_empty())
        || argv.iter().any(|value| value.contains('\0'))
    {
        return Err(NavigationError::Validation(
            "The opener requires a nonempty executable path and arguments without NUL characters."
                .into(),
        )
        .into());
    }
    Ok(())
}

pub(crate) async fn open(
    repo: &Repository,
    id: &str,
    revision: Option<&str>,
    explicit_opener: Option<&Path>,
    configured: &[String],
    format: OutputFormat,
) -> Result<()> {
    validate_id(id)?;
    let record = repo.inspect_record(id, 0).await?;
    check_revision(&record, revision)?;
    let uri = record.provenance.source_uri.as_deref().ok_or_else(|| {
        NavigationError::Validation(
            "This result has no local source file. Use inspect to read the note or chat context."
                .into(),
        )
    })?;
    let path = source_path(uri)?;
    match std::fs::metadata(&path) {
        Ok(metadata) if metadata.is_file() => {},
        Ok(_) => return Err(NavigationError::Validation(format!("Source is not a regular file: {}. Use inspect to read the stored content.", path.display())).into()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Err(NavigationError::NotFound(format!("Source file not found: {}. Restore it at the recorded path, or use inspect to read the stored content. Reimport the source after restoring it.", path.display())).into()),
        Err(error) => return Err(error).with_context(|| format!("Cannot read source file: {}. Use inspect to read the stored content.", path.display())),
    }
    let argv = opener_argv(explicit_opener, configured)?;
    // Use literal argv. Neither imported content nor file paths become shell
    // code. Keep application output on stderr so our JSON remains parseable.
    let status = Command::new(&argv[0]).args(&argv[1..]).arg(&path)
        .stdin(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::from(std::io::stderr()))
        .stderr(std::process::Stdio::inherit())
        .status().with_context(|| format!("Cannot launch opener {}. Configure navigation.opener as an argv array or pass --opener with an executable path.", argv[0]))?;
    if !status.success() {
        anyhow::bail!("Source opener failed with {status}; the stored record was preserved.");
    }
    let data = OpenOutput {
        id: &record.id,
        path: path.clone(),
        launched: true,
        opener: argv,
        revision: &record.revision,
        provenance: &record.provenance,
    };
    output::print(format, "open", data, |writer| {
        writeln!(
            writer,
            "Opened {}",
            output::safe_text(&path.to_string_lossy(), false)
        )
    })
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct SearchNavigation {
    pub(crate) title: String,
    pub(crate) preview: String,
    pub(crate) inspect_command: String,
    pub(crate) open_command: Option<String>,
    pub(crate) revision: Option<String>,
    pub(crate) provenance: Option<graphrag_db::repository::InspectionProvenance>,
    pub(crate) warning: Option<String>,
}

pub(crate) async fn enrich_search(
    repo: &Repository,
    results: impl IntoIterator<Item = (String, Option<String>, String)>,
    query: &str,
    database: &Path,
    config: Option<&Path>,
) -> HashMap<String, SearchNavigation> {
    let mut navigation = HashMap::new();
    for (id, title, content) in results {
        let record = match repo.inspect_record(&id, 0).await {
            Ok(record) if record.content == content => Some(record),
            Ok(_) => {
                warn!(record_id = %id, "record changed while enriching search result");
                None
            }
            Err(error) => {
                warn!(record_id = %id, %error, "unable to inspect search result");
                None
            }
        };
        let revision = record.as_ref().map(|record| record.revision.clone());
        // A failed or raced enrichment must never offer a command that could
        // silently resolve a reused ID to content different from this result.
        let guard = revision.as_deref().unwrap_or("unavailable-search-snapshot");
        let mut base = String::from("graphrag");
        if let Some(path) = config {
            base.push_str(&format!(
                " --config {}",
                shell_quote(&absolute_database_path(path))
            ));
        }
        base.push_str(&format!(
            " --db-path {}",
            shell_quote(&absolute_database_path(database))
        ));
        let suffix = format!("{} --revision {}", shell_quote(&id), shell_quote(guard));
        let open_command = record
            .as_ref()
            .and_then(|record| record.provenance.source_uri.as_deref())
            .filter(|uri| source_path(uri).is_ok())
            .map(|_| format!("{base} open {suffix}"));
        let warning = record.as_ref().map(|record| record.warnings.join(" ")).filter(|value| !value.is_empty())
            .or_else(|| record.is_none().then(|| "Record changed or became unavailable. Run search again before inspecting or opening it.".into()));
        navigation.insert(
            id.clone(),
            SearchNavigation {
                title: readable_title(title.as_deref(), &content),
                preview: query_preview(&content, query, 200),
                inspect_command: format!("{base} inspect {suffix}"),
                open_command,
                revision,
                provenance: record.map(|record| record.provenance),
                warning,
            },
        );
    }
    navigation
}

fn absolute_database_path(path: &Path) -> String {
    if path.is_absolute() {
        path.to_string_lossy().into_owned()
    } else {
        std::env::current_dir()
            .unwrap_or_default()
            .join(path)
            .to_string_lossy()
            .into_owned()
    }
}

/// Preserve an explicitly selected config (including environment selection),
/// or the existing platform default, in follow-up commands copied elsewhere.
pub(crate) fn selected_config_path(explicit: Option<&Path>) -> Option<PathBuf> {
    explicit
        .map(Path::to_path_buf)
        .or_else(|| {
            std::env::var("GRAPHRAG_CONFIG")
                .ok()
                .filter(|value| !value.trim().is_empty())
                .map(|value| {
                    if let Some(tail) = value.strip_prefix("~/") {
                        if let Some(home) = std::env::var_os("HOME") {
                            return PathBuf::from(home).join(tail);
                        }
                    }
                    PathBuf::from(value)
                })
        })
        .or_else(|| graphrag_config::default_config_path().filter(|path| path.exists()))
}

pub(crate) fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

pub(crate) fn readable_title(title: Option<&str>, content: &str) -> String {
    let text = title
        .filter(|title| !title.trim().is_empty())
        .unwrap_or_else(|| {
            content
                .lines()
                .find(|line| !line.trim().is_empty())
                .unwrap_or("(empty record)")
        });
    text.chars()
        .take(100)
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Fold with an original-character index map; Unicode lowercase expansion
/// never creates an invalid UTF-8 slice or shifts a preview's source position.
pub(crate) fn query_preview(content: &str, query: &str, limit: usize) -> String {
    let chars = content.chars().collect::<Vec<_>>();
    if limit == 0 {
        return String::new();
    }
    let folded = chars
        .iter()
        .enumerate()
        .flat_map(|(index, character)| character.to_lowercase().map(move |folded| (folded, index)))
        .collect::<Vec<_>>();
    let mut terms = query
        .split(|character: char| !character.is_alphanumeric())
        .filter(|term| term.chars().count() >= 2)
        .filter(|term| {
            !matches!(
                term.to_lowercase().as_str(),
                "what" | "the" | "is" | "and" | "of" | "to" | "for" | "in" | "on" | "with" | "how"
            )
        })
        .map(|term| term.to_lowercase().chars().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    terms.sort_by_key(|term| std::cmp::Reverse(term.len()));
    let center = terms
        .iter()
        .find_map(|term| {
            folded
                .windows(term.len())
                .find(|window| {
                    window
                        .iter()
                        .map(|(character, _)| character)
                        .eq(term.iter())
                })
                .map(|window| window[0].1)
        })
        .unwrap_or(0);
    let end = chars
        .len()
        .min(center.saturating_sub(limit / 3).saturating_add(limit));
    let start = end.saturating_sub(limit);
    let preview = chars[start..end]
        .iter()
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    format!(
        "{}{}{}",
        if start > 0 { "…" } else { "" },
        preview,
        if end < chars.len() { "…" } else { "" }
    )
}

pub(crate) fn print_search_navigation(navigation: &SearchNavigation) {
    if let Some(provenance) = &navigation.provenance {
        if let Some(uri) = &provenance.source_uri {
            println!("   Source: {uri}");
        }
        if !provenance.heading_path.is_empty() {
            println!("   Heading: {}", provenance.heading_path.join(" > "));
        }
        if let Some(line) = provenance.start_line {
            println!(
                "   Lines: {line}-{} (at last import)",
                provenance.end_line.unwrap_or(line)
            );
        }
        if let Some(uuid) = &provenance.conversation_uuid {
            println!("   Conversation UUID: {uuid}");
        }
        if let Some(uuid) = &provenance.message_uuid {
            println!("   Message UUID: {uuid}");
        } else if let Some(key) = &provenance.message_key {
            println!("   Message key: {key}");
        }
    }
    println!("   {}", navigation.preview);
    println!("   Inspect: {}", navigation.inspect_command);
    if let Some(command) = &navigation.open_command {
        println!("   Open source: {command}");
    }
    if let Some(warning) = &navigation.warning {
        println!("   Warning: {warning}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_finds_late_unicode_query_without_byte_truncation() {
        let content = format!(
            "{} İSTANBUL 日本語 launch details {}",
            "前文🙂 ".repeat(100),
            "続き ".repeat(100)
        );
        let preview = query_preview(&content, "istanbul launch", 60);
        assert!(preview.contains("launch details"));
        assert!(preview.starts_with('…'));
        assert!(preview.ends_with('…'));
        assert!(query_preview("日本語🙂", "other", 200) == "日本語🙂");
        assert_eq!(query_preview("anything", "", 0), "");
    }

    #[test]
    fn literal_file_uri_and_shell_hint_preserve_special_characters() {
        let path = "/tmp/a # b%20 '$()`日本語.md";
        assert_eq!(
            source_path(&format!("file://{path}")).unwrap(),
            PathBuf::from(path)
        );
        assert!(source_path("https://example.org/notes").is_err());
        assert!(source_path("file://remote/path").is_err());
        assert_eq!(shell_quote("a'b"), "'a'\\''b'");
        assert!(validate_id("1").is_err());
        assert!(validate_id("note:").is_err());
        assert!(validate_id("note:1").is_ok());
    }
}
