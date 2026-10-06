//! Exact, copyable recovery for explicitly selected hybrid retrieval.
use crate::cli::{Cli, Commands, SearchModeArg, SearchScopeArg};
use crate::output::OutputFormat;
use graphrag_config::RuntimeConfig;
use std::path::Path;

fn absolute(path: &Path) -> String {
    if path.is_absolute() {
        path.to_string_lossy().into_owned()
    } else {
        std::env::current_dir()
            .map(|directory| directory.join(path))
            .unwrap_or_else(|_| path.into())
            .to_string_lossy()
            .into_owned()
    }
}

pub(crate) fn keyword_command(cli: &Cli, config: &RuntimeConfig) -> Option<String> {
    let Commands::Search {
        query,
        mode: SearchModeArg::Hybrid,
        limit,
        scope,
        since_days,
        source_uri,
        context,
        format,
        ..
    } = &cli.command
    else {
        return None;
    };
    let mut command = "graphrag".to_string();
    let config_path = crate::commands::navigation::selected_config_path(cli.config.as_deref());
    if let Some(path) = config_path {
        command.push_str(&format!(
            " --config {}",
            crate::commands::navigation::shell_quote(&absolute(&path))
        ));
    }
    if cli.memory {
        command.push_str(" --memory");
    } else {
        command.push_str(&format!(
            " --db-path {}",
            crate::commands::navigation::shell_quote(&absolute(&config.database.path))
        ));
    }
    if cli.explain {
        command.push_str(" --explain");
    }
    let scope = match scope {
        SearchScopeArg::Notes => "notes",
        SearchScopeArg::Messages => "messages",
        SearchScopeArg::All => "all",
    };
    let format = match format {
        OutputFormat::Human => "human",
        OutputFormat::Json => "json",
        OutputFormat::Jsonl => "jsonl",
    };
    command.push_str(&format!(
        " search --mode keyword --scope {scope} --limit {} --graph off --format {format}",
        limit.unwrap_or(config.search.default_limit)
    ));
    if let Some(days) = since_days {
        command.push_str(&format!(" --since-days {days}"));
    }
    if let Some(uri) = source_uri {
        command.push_str(&format!(
            " --source-uri={}",
            crate::commands::navigation::shell_quote(uri)
        ));
    }
    if *context {
        command.push_str(" --context");
    }
    // A quoted leading '-' is still an option to Clap unless the query is
    // placed after the option terminator. All retrieval flags precede it.
    command.push_str(&format!(
        " -- {}",
        crate::commands::navigation::shell_quote(query)
    ));
    Some(command)
}

/// Preserve safe retrieval arguments and connection references, never a bearer value.
pub(crate) fn remote_keyword_command(cli: &Cli, server: &str) -> Option<String> {
    let Commands::Search {
        query,
        mode: SearchModeArg::Hybrid,
        limit,
        scope,
        since_days,
        source_uri,
        format,
        context,
        ..
    } = &cli.command
    else {
        return None;
    };
    if *context {
        return None;
    }
    let quote = crate::commands::navigation::shell_quote;
    let scope = match scope {
        SearchScopeArg::Notes => "notes",
        SearchScopeArg::Messages => "messages",
        SearchScopeArg::All => "all",
    };
    let format = match format {
        OutputFormat::Human => "human",
        OutputFormat::Json => "json",
        OutputFormat::Jsonl => "jsonl",
    };
    let mut command = format!("graphrag --server {} --credential-env {} search --mode keyword --graph off --scope {scope} --limit {} --format {format}", quote(server), quote(&cli.credential_env), limit.unwrap_or(10));
    if let Some(days) = since_days {
        command.push_str(&format!(" --since-days {days}"));
    }
    if let Some(uri) = source_uri {
        command.push_str(&format!(" --source-uri={}", quote(uri)));
    }
    command.push_str(&format!(" -- {}", quote(query)));
    Some(command)
}

#[cfg(test)]
mod remote_tests {
    use super::*;
    use clap::Parser;
    #[test]
    fn remote_fallback_preserves_query_filters_and_connection_without_secrets() {
        let cli = Cli::parse_from([
            "graphrag",
            "--server",
            "http://127.0.0.1:3000/mcp",
            "--credential-env",
            "PRIVATE_NOTES_TOKEN",
            "search",
            "--scope",
            "notes",
            "--limit",
            "7",
            "--since-days",
            "14",
            "--source-uri",
            "mcp://upload/example",
            "--graph",
            "on",
            "--format",
            "json",
            "--",
            "-Atlas's $(literal)",
        ]);
        let command = remote_keyword_command(&cli, cli.server.as_deref().unwrap()).unwrap();
        assert!(
            command.contains("--mode keyword --graph off --scope notes --limit 7 --format json")
        );
        assert!(command.contains("--credential-env 'PRIVATE_NOTES_TOKEN'"));
        assert!(command.contains("--since-days 14 --source-uri='mcp://upload/example'"));
        assert!(command.ends_with("-- '-Atlas'\\''s $(literal)'"));
    }
}
