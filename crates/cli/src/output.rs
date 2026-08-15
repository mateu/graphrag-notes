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

/// Emit one versioned envelope per result while retaining pipeline metadata on
/// every record. An empty result set still emits a single envelope with a
/// `null` result, so JSONL consumers never lose the command filters or
/// diagnostics that explain an empty response.
pub fn print_jsonl_with_pipeline<T: Serialize>(
    command: &str,
    values: impl IntoIterator<Item = T>,
    pipeline: serde_json::Value,
) -> Result<()> {
    let mut stdout = io::stdout().lock();
    write_jsonl_with_pipeline(&mut stdout, command, values, &pipeline)
}

#[derive(Serialize)]
struct JsonlPipelineData<'a, T> {
    result: Option<T>,
    pipeline: &'a serde_json::Value,
}

fn write_jsonl_with_pipeline<T: Serialize>(
    writer: &mut dyn Write,
    command: &str,
    values: impl IntoIterator<Item = T>,
    pipeline: &serde_json::Value,
) -> Result<()> {
    let mut wrote_result = false;
    for value in values {
        serde_json::to_writer(
            &mut *writer,
            &OutputEnvelope::success(
                command,
                JsonlPipelineData {
                    result: Some(value),
                    pipeline,
                },
            ),
        )?;
        writeln!(writer)?;
        wrote_result = true;
    }
    if !wrote_result {
        serde_json::to_writer(
            &mut *writer,
            &OutputEnvelope::success(
                command,
                JsonlPipelineData::<T> {
                    result: None,
                    pipeline,
                },
            ),
        )?;
        writeln!(writer)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_jsonl_retains_metadata_for_results_and_empty_sets() {
        let pipeline = serde_json::json!({"filters": {"scope": "Notes"}});
        let mut output = Vec::new();
        write_jsonl_with_pipeline(
            &mut output,
            "search",
            [
                serde_json::json!({"id": "note:1"}),
                serde_json::json!({"id": "note:2"}),
            ],
            &pipeline,
        )
        .unwrap();
        let records = std::str::from_utf8(&output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0]["data"]["result"]["id"], "note:1");
        assert_eq!(records[1]["data"]["result"]["id"], "note:2");
        assert!(records
            .iter()
            .all(|record| record["data"]["pipeline"] == pipeline));

        output.clear();
        write_jsonl_with_pipeline(
            &mut output,
            "augment",
            std::iter::empty::<serde_json::Value>(),
            &pipeline,
        )
        .unwrap();
        let record = serde_json::from_slice::<serde_json::Value>(&output).unwrap();
        assert!(record["data"]["result"].is_null());
        assert_eq!(record["data"]["pipeline"], pipeline);
    }
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
