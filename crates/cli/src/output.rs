use anyhow::Result;
use clap::ValueEnum;
use serde::Serialize;
use std::io::{self, Write};

/// Stable machine-output contract shared by newly modular CLI commands.
pub const CLI_OUTPUT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum OutputFormat {
    Human,
    Json,
    Jsonl,
}

#[derive(Debug, Serialize)]
pub struct OutputEnvelope<T> {
    pub schema_version: u32,
    pub command: String,
    pub success: bool,
    pub data: T,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

impl<T> OutputEnvelope<T> {
    pub fn success(command: impl Into<String>, data: T) -> Self {
        Self {
            schema_version: CLI_OUTPUT_SCHEMA_VERSION,
            command: command.into(),
            success: true,
            data,
            warnings: Vec::new(),
            errors: Vec::new(),
        }
    }
}

/// Write requested command data exclusively to stdout. Logging and provider
/// progress are configured separately on stderr, so JSON/JSONL remain safe
/// for shell pipelines and LLM-backed automation.
pub fn print<T: Serialize>(
    format: OutputFormat,
    command: &str,
    data: T,
    human: impl FnOnce(&mut dyn Write) -> io::Result<()>,
) -> Result<()> {
    let mut stdout = io::stdout().lock();
    match format {
        OutputFormat::Human => human(&mut stdout)?,
        OutputFormat::Json => {
            serde_json::to_writer_pretty(&mut stdout, &OutputEnvelope::success(command, data))?;
            writeln!(stdout)?;
        }
        OutputFormat::Jsonl => {
            serde_json::to_writer(&mut stdout, &OutputEnvelope::success(command, data))?;
            writeln!(stdout)?;
        }
    }
    Ok(())
}

/// Emit one versioned envelope per item for streaming/list command forms.
pub fn print_jsonl<T: Serialize>(command: &str, values: impl IntoIterator<Item = T>) -> Result<()> {
    let mut stdout = io::stdout().lock();
    for value in values {
        serde_json::to_writer(&mut stdout, &OutputEnvelope::success(command, value))?;
        writeln!(stdout)?;
    }
    Ok(())
}

/// Documented process outcomes. Command handlers return ordinary errors; the
/// top-level runner maps them to these values as output coverage expands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ExitCode {
    Success = 0,
    Validation = 2,
    NotFound = 3,
    Compatibility = 4,
    PartialFailure = 5,
    Internal = 1,
}
