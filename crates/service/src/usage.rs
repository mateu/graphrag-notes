//! One privacy-safe usage record per tool call, for measuring how clients use
//! the shared corpus. Records name the authenticated principal, the tool, its
//! outcome and latency, a few allowlisted request options and result
//! statistics. They never contain query text, record IDs, content, titles,
//! source URIs or client-supplied free text.

use rmcp::model::{CallToolResponse, ErrorData};
use serde_json::{Map, Value};

/// `tracing` target for usage records; filter on it to extract them.
pub(crate) const TARGET: &str = "graphrag_usage";

#[derive(Debug, Default, PartialEq)]
pub(crate) struct Usage {
    pub outcome: String,
    pub mode: Option<&'static str>,
    pub graph: Option<&'static str>,
    pub scope: Option<&'static str>,
    pub limit: Option<u64>,
    pub query_chars: Option<usize>,
    pub query_terms: Option<usize>,
    pub hits: Option<usize>,
    pub top_score: Option<f64>,
    pub graph_considered: Option<u64>,
    pub graph_selected: Option<u64>,
    pub dropped_for_relevance: Option<u64>,
    pub total_tokens: Option<u64>,
}

/// Map a client-supplied option to a known value, so arbitrary strings are
/// never logged.
fn allowlisted(
    args: &Map<String, Value>,
    key: &str,
    values: &[&'static str],
) -> Option<&'static str> {
    let value = args.get(key)?.as_str()?;
    Some(
        values
            .iter()
            .copied()
            .find(|known| *known == value)
            .unwrap_or("other"),
    )
}

fn count(value: Option<&Value>) -> Option<u64> {
    value.and_then(Value::as_u64)
}

pub(crate) fn summarize(
    tool: &str,
    args: Option<&Map<String, Value>>,
    result: &Result<CallToolResponse, ErrorData>,
) -> Usage {
    let empty = Map::new();
    let args = args.unwrap_or(&empty);
    let mut usage = Usage::default();
    if matches!(tool, "search_notes" | "build_context") {
        usage.mode = allowlisted(args, "mode", &["keyword", "hybrid"]);
        usage.graph = allowlisted(args, "graph", &["off", "auto", "on"]);
        usage.scope = allowlisted(args, "scope", &["notes", "messages", "all"]);
        usage.limit = count(args.get("limit")).or_else(|| count(args.get("max_chunks")));
        if let Some(query) = args.get("query").and_then(Value::as_str) {
            usage.query_chars = Some(query.chars().count());
            usage.query_terms = Some(query.split_whitespace().count());
        }
    }
    let structured = match result {
        Err(_) => {
            usage.outcome = "protocol_error".into();
            return usage;
        }
        Ok(CallToolResponse::Complete(result)) => result.structured_content.as_ref(),
        Ok(_) => {
            usage.outcome = "deferred".into();
            return usage;
        }
    };
    let error_code = structured
        .and_then(|value| value.pointer("/error/code"))
        .and_then(Value::as_str);
    usage.outcome = match error_code {
        Some(code) => code.to_string(),
        None => "ok".into(),
    };
    let Some(data) = structured.and_then(|value| value.get("data")) else {
        return usage;
    };
    let items = match tool {
        "search_notes" => data.get("records"),
        "build_context" => data.get("chunks"),
        _ => None,
    };
    if let Some(items) = items.and_then(Value::as_array) {
        usage.hits = Some(items.len());
        usage.top_score = items
            .iter()
            .filter_map(|item| item.get("score").and_then(Value::as_f64))
            .reduce(f64::max);
    }
    if tool == "build_context" {
        let diagnostics = data.get("diagnostics");
        let field = |name: &str| count(diagnostics.and_then(|d| d.get(name)));
        usage.graph_considered = field("graph_candidates_considered");
        usage.graph_selected = field("graph_candidates_selected");
        usage.dropped_for_relevance = field("dropped_for_relevance");
        usage.total_tokens = count(data.get("total_tokens"));
    }
    usage
}

pub(crate) fn record(principal: &str, tool: &str, elapsed_ms: f64, usage: &Usage) {
    tracing::info!(
        target: TARGET,
        principal,
        tool,
        outcome = usage.outcome.as_str(),
        elapsed_ms = (elapsed_ms * 10.0).round() / 10.0,
        mode = usage.mode,
        graph = usage.graph,
        scope = usage.scope,
        limit = usage.limit,
        query_chars = usage.query_chars,
        query_terms = usage.query_terms,
        hits = usage.hits,
        top_score = usage.top_score,
        graph_considered = usage.graph_considered,
        graph_selected = usage.graph_selected,
        dropped_for_relevance = usage.dropped_for_relevance,
        total_tokens = usage.total_tokens,
        "tool call"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::model::CallToolResult;
    use serde_json::json;

    fn args(value: Value) -> Map<String, Value> {
        value.as_object().unwrap().clone()
    }

    fn ok(value: Value) -> Result<CallToolResponse, ErrorData> {
        Ok(CallToolResult::structured(value).into())
    }

    #[test]
    fn search_counts_hits_and_top_score_without_the_query_text() {
        let args = args(
            json!({"query": "secret plans for tuesday", "mode": "hybrid", "graph": "on", "scope": "notes", "limit": 3}),
        );
        let result = ok(
            json!({"schema_version": 1, "error": null, "data": {"records": [
                {"id": "note:a", "score": 0.4}, {"id": "note:b", "score": 0.9}
            ]}}),
        );
        let usage = summarize("search_notes", Some(&args), &result);
        assert_eq!(usage.outcome, "ok");
        assert_eq!(
            (usage.mode, usage.graph, usage.scope, usage.limit),
            (Some("hybrid"), Some("on"), Some("notes"), Some(3))
        );
        assert_eq!((usage.query_chars, usage.query_terms), (Some(24), Some(4)));
        assert_eq!((usage.hits, usage.top_score), (Some(2), Some(0.9)));
        assert!(!format!("{usage:?}").contains("secret"));
    }

    #[test]
    fn empty_search_is_ok_with_zero_hits() {
        let result = ok(json!({"schema_version": 1, "error": null, "data": {"records": []}}));
        let usage = summarize("search_notes", None, &result);
        assert_eq!(
            (usage.outcome.as_str(), usage.hits, usage.top_score),
            ("ok", Some(0), None)
        );
    }

    #[test]
    fn build_context_reports_graph_and_budget_diagnostics() {
        let result = ok(json!({"schema_version": 1, "error": null, "data": {
            "chunks": [{"score": 0.02}], "total_tokens": 1013,
            "diagnostics": {"graph_candidates_considered": 7, "graph_candidates_selected": 1, "dropped_for_relevance": 2}
        }}));
        let usage = summarize(
            "build_context",
            Some(&args(json!({"query": "x", "max_chunks": 4}))),
            &result,
        );
        assert_eq!(usage.hits, Some(1));
        assert_eq!(
            (
                usage.graph_considered,
                usage.graph_selected,
                usage.dropped_for_relevance
            ),
            (Some(7), Some(1), Some(2))
        );
        assert_eq!((usage.total_tokens, usage.limit), (Some(1013), Some(4)));
    }

    #[test]
    fn tool_failures_record_their_stable_code() {
        let result: Result<CallToolResponse, ErrorData> =
            Ok(crate::tools::failure("forbidden", "no", false).into());
        assert_eq!(
            summarize("capture_note", None, &result).outcome,
            "forbidden"
        );
        let protocol = Err(ErrorData::invalid_params(
            "Unknown tool; refresh tools/list.",
            None,
        ));
        assert_eq!(summarize("nope", None, &protocol).outcome, "protocol_error");
    }

    #[test]
    fn unknown_option_values_are_not_logged_verbatim() {
        let args = args(json!({"mode": "injected free text", "graph": 5}));
        let usage = summarize(
            "search_notes",
            Some(&args),
            &ok(json!({"data": {"records": []}})),
        );
        assert_eq!((usage.mode, usage.graph), (Some("other"), None));
    }
}
