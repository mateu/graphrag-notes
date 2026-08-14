//! Pure evaluation schema, metrics, reporting, and baseline comparison.
//!
//! This module intentionally does not depend on the database or inference clients so metric
//! behavior can be tested from precomputed ranked results.

use anyhow::{bail, Context, Result};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub const EVAL_SCHEMA_VERSION: u32 = 2;
const MAX_NDCG_GRADE: u32 = 63;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum EvalScope {
    Notes,
    Messages,
    All,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EvalRelevance {
    pub id: String,
    #[serde(default)]
    pub grade: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EvalAugmentCase {
    /// Version of this individual case. Missing means the legacy v1 shape.
    #[serde(default)]
    pub schema_version: Option<u32>,
    pub query: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub scope: Option<EvalScope>,
    #[serde(default)]
    pub limit: Option<usize>,
    /// Metric cutoff. Defaults to the resolved retrieval limit.
    #[serde(default)]
    pub k: Option<usize>,
    #[serde(default)]
    pub since_days: Option<u32>,
    #[serde(default)]
    pub source_uri: Option<String>,
    #[serde(default)]
    pub entity: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub max_chunk_tokens: Option<usize>,

    /// Legacy ungraded relevance field. Multiple IDs are supported.
    #[serde(default)]
    pub expected_ids: Vec<String>,
    /// New relevance field. A grade makes the case eligible for nDCG.
    #[serde(default, alias = "relevant_records")]
    pub relevance: Vec<EvalRelevance>,
    /// Legacy substring expectation. It is reported independently from exact-ID ranking metrics.
    #[serde(default)]
    pub expected_contains: Vec<String>,
    #[serde(default)]
    pub expected_source_uris: Vec<String>,
    #[serde(default)]
    pub expected_conversation_uuids: Vec<String>,
    #[serde(default)]
    pub forbidden_ids: Vec<String>,
    #[serde(default)]
    pub forbidden_contains: Vec<String>,
}

/// The explicitly versioned format is deliberately strict: CI expectations
/// must never be silently discarded because of a misspelled field.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct VersionedEvalAugmentCase {
    pub schema_version: u32,
    pub query: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub scope: Option<EvalScope>,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub k: Option<usize>,
    #[serde(default)]
    pub since_days: Option<u32>,
    #[serde(default)]
    pub source_uri: Option<String>,
    #[serde(default)]
    pub entity: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub max_chunk_tokens: Option<usize>,
    #[serde(default)]
    pub expected_ids: Vec<String>,
    #[serde(default, alias = "relevant_records")]
    pub relevance: Vec<VersionedEvalRelevance>,
    #[serde(default)]
    pub expected_contains: Vec<String>,
    #[serde(default)]
    pub expected_source_uris: Vec<String>,
    #[serde(default)]
    pub expected_conversation_uuids: Vec<String>,
    #[serde(default)]
    pub forbidden_ids: Vec<String>,
    #[serde(default)]
    pub forbidden_contains: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct VersionedEvalRelevance {
    pub id: String,
    #[serde(default)]
    pub grade: Option<u32>,
}

impl From<VersionedEvalAugmentCase> for EvalAugmentCase {
    fn from(case: VersionedEvalAugmentCase) -> Self {
        Self {
            schema_version: Some(case.schema_version),
            query: case.query,
            name: case.name,
            scope: case.scope,
            limit: case.limit,
            k: case.k,
            since_days: case.since_days,
            source_uri: case.source_uri,
            entity: case.entity,
            max_tokens: case.max_tokens,
            max_chunk_tokens: case.max_chunk_tokens,
            expected_ids: case.expected_ids,
            relevance: case
                .relevance
                .into_iter()
                .map(|item| EvalRelevance {
                    id: item.id,
                    grade: item.grade,
                })
                .collect(),
            expected_contains: case.expected_contains,
            expected_source_uris: case.expected_source_uris,
            expected_conversation_uuids: case.expected_conversation_uuids,
            forbidden_ids: case.forbidden_ids,
            forbidden_contains: case.forbidden_contains,
        }
    }
}

impl EvalAugmentCase {
    pub fn display_name(&self) -> &str {
        self.name.as_deref().unwrap_or(&self.query)
    }

    pub fn resolved_k(&self, default_limit: usize) -> usize {
        self.k.or(self.limit).unwrap_or(default_limit)
    }

    pub fn relevance(&self) -> BTreeMap<String, u32> {
        let mut relevant = BTreeMap::new();
        for id in &self.expected_ids {
            if let Some(id) = normalize_id(id) {
                relevant.entry(id).or_insert(1);
            }
        }
        for item in &self.relevance {
            if let Some(id) = normalize_id(&item.id) {
                let grade = item.grade.unwrap_or(1);
                if grade == 0 {
                    continue;
                }
                relevant
                    .entry(id)
                    .and_modify(|existing| *existing = (*existing).max(grade))
                    .or_insert(grade);
            }
        }
        relevant
    }

    pub fn has_grades(&self) -> bool {
        self.relevance.iter().any(|item| item.grade.is_some())
    }

    pub fn has_checks(&self) -> bool {
        !self.relevance().is_empty()
            || !normalized_strings(&self.expected_contains).is_empty()
            || !normalized_strings(&self.expected_source_uris).is_empty()
            || !normalized_strings(&self.expected_conversation_uuids).is_empty()
            || !normalized_ids(&self.forbidden_ids).is_empty()
            || !normalized_strings(&self.forbidden_contains).is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RankedResult {
    pub id: String,
    #[serde(default)]
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_uri: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conversation_uuid: Option<String>,
    #[serde(default)]
    pub approx_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CaseMetrics {
    pub k: usize,
    pub retrieved: usize,
    pub relevant_total: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recall_at_k: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub precision_at_k: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reciprocal_rank: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ndcg_at_k: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provenance_accuracy: Option<f64>,
    pub substring_expectation_matched: Option<bool>,
    pub forbidden_result_found: bool,
    /// `None` denotes an unscored case with no positive or negative expectation.
    pub checks_passed: Option<bool>,
    pub chunks: usize,
    pub tokens: usize,
    pub latency_ms: u64,
}

pub fn evaluate_ranked_results(
    case: &EvalAugmentCase,
    results: &[RankedResult],
    k: usize,
    latency_ms: u64,
) -> CaseMetrics {
    let relevance = case.relevance();
    let ranked = &results[..results.len().min(k)];
    let ranked_ids: Vec<String> = ranked
        .iter()
        .filter_map(|result| normalize_id(&result.id))
        .collect();
    let retrieved_relevant: BTreeSet<String> = ranked_ids
        .iter()
        .filter(|id| relevance.contains_key(*id))
        .cloned()
        .collect();
    let relevant_total = relevance.len();

    let recall_at_k =
        (!relevance.is_empty()).then(|| retrieved_relevant.len() as f64 / relevant_total as f64);
    let precision_at_k =
        (!relevance.is_empty() && k > 0).then(|| retrieved_relevant.len() as f64 / k as f64);
    let reciprocal_rank = (!relevance.is_empty()).then(|| {
        ranked_ids
            .iter()
            .position(|id| relevance.contains_key(id))
            .map(|index| 1.0 / (index + 1) as f64)
            .unwrap_or(0.0)
    });
    let ndcg_at_k = (case.has_grades() && !relevance.is_empty()).then(|| {
        let dcg = ranked_ids
            .iter()
            .enumerate()
            .map(|(index, id)| {
                let grade = relevance.get(id).copied().unwrap_or(0) as f64;
                ndcg_gain(grade as u32) / ((index + 2) as f64).log2()
            })
            .sum::<f64>();
        let mut ideal_grades: Vec<u32> = relevance.values().copied().collect();
        ideal_grades.sort_unstable_by(|a, b| b.cmp(a));
        let ideal_dcg = ideal_grades
            .into_iter()
            .take(k)
            .enumerate()
            .map(|(index, grade)| ndcg_gain(grade) / ((index + 2) as f64).log2())
            .sum::<f64>();
        if ideal_dcg == 0.0 {
            0.0
        } else {
            dcg / ideal_dcg
        }
    });

    let expected_text = normalized_strings(&case.expected_contains);
    let substring_expectation_matched = (!expected_text.is_empty()).then(|| {
        expected_text.iter().any(|needle| {
            results
                .iter()
                .any(|result| result_prompt_text(result).contains(needle))
        })
    });
    let forbidden_ids = normalized_ids(&case.forbidden_ids);
    let forbidden_text = normalized_strings(&case.forbidden_contains);
    // Negative expectations protect the entire prompt context, not just the
    // top-k slice used for ranking metrics.
    let forbidden_result_found = results.iter().any(|result| {
        normalize_id(&result.id).is_some_and(|id| forbidden_ids.contains(&id))
            || forbidden_text
                .iter()
                .any(|needle| result_prompt_text(result).contains(needle))
    });

    let expected_sources = normalized_strings(&case.expected_source_uris);
    let expected_conversations = normalized_strings(&case.expected_conversation_uuids);
    let has_provenance_expectation =
        !expected_sources.is_empty() || !expected_conversations.is_empty();
    let provenance_accuracy = has_provenance_expectation.then(|| {
        let matching_results = results
            .iter()
            .filter(|result| {
                let source_matches = expected_sources.is_empty()
                    || result.source_uri.as_deref().is_some_and(|actual| {
                        expected_sources.contains(&actual.trim().to_lowercase())
                    });
                let conversation_matches = expected_conversations.is_empty()
                    || result.conversation_uuid.as_deref().is_some_and(|actual| {
                        expected_conversations.contains(&actual.trim().to_lowercase())
                    });
                source_matches && conversation_matches
            })
            .count();
        matching_results as f64 / results.len().max(1) as f64
    });

    // Each explicitly configured positive expectation is independently
    // required. A matching ID must not hide missing expected prompt text (or
    // vice versa) when a case specifies both categories.
    let relevance_passed = relevance.is_empty() || recall_at_k.is_some_and(|recall| recall > 0.0);
    let text_passed =
        expected_text.is_empty() || substring_expectation_matched.is_some_and(|matched| matched);
    let provenance_passed = provenance_accuracy.is_none_or(|accuracy| accuracy == 1.0);
    let checks_passed = case
        .has_checks()
        .then(|| relevance_passed && text_passed && !forbidden_result_found && provenance_passed);

    CaseMetrics {
        k,
        retrieved: ranked.len(),
        relevant_total,
        recall_at_k,
        precision_at_k,
        reciprocal_rank,
        ndcg_at_k,
        provenance_accuracy,
        substring_expectation_matched,
        forbidden_result_found,
        checks_passed,
        chunks: results.len(),
        tokens: results.iter().map(|result| result.approx_tokens).sum(),
        latency_ms,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvalCaseReport {
    pub name: String,
    pub query: String,
    pub metrics: CaseMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvalSummary {
    pub cases_total: usize,
    pub cases_with_checks: usize,
    pub cases_passed: usize,
    pub cases_missed: usize,
    pub cases_unscored: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recall_at_k: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub precision_at_k: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mrr: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ndcg_at_k: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provenance_accuracy: Option<f64>,
    pub avg_chunks_per_case: f64,
    pub avg_tokens_per_case: f64,
    pub avg_latency_ms: f64,
    pub total_latency_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvalMetadata {
    pub schema_version: u32,
    pub provider: String,
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvalRunReport {
    pub metadata: EvalMetadata,
    pub cases: Vec<EvalCaseReport>,
    pub summary: EvalSummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline: Option<BaselineComparison>,
}

impl EvalRunReport {
    pub fn from_cases(metadata: EvalMetadata, cases: Vec<EvalCaseReport>) -> Self {
        let total = cases.len();
        let with_checks = cases
            .iter()
            .filter(|case| case.metrics.checks_passed.is_some())
            .count();
        let passed = cases
            .iter()
            .filter(|case| case.metrics.checks_passed == Some(true))
            .count();
        let average = |values: Vec<f64>| {
            (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
        };
        let total_latency_ms = cases.iter().map(|case| case.metrics.latency_ms).sum();
        let denom = total.max(1) as f64;
        Self {
            metadata,
            summary: EvalSummary {
                cases_total: total,
                cases_with_checks: with_checks,
                cases_passed: passed,
                cases_missed: with_checks - passed,
                cases_unscored: total - with_checks,
                recall_at_k: average(
                    cases
                        .iter()
                        .filter_map(|case| case.metrics.recall_at_k)
                        .collect(),
                ),
                precision_at_k: average(
                    cases
                        .iter()
                        .filter_map(|case| case.metrics.precision_at_k)
                        .collect(),
                ),
                mrr: average(
                    cases
                        .iter()
                        .filter_map(|case| case.metrics.reciprocal_rank)
                        .collect(),
                ),
                ndcg_at_k: average(
                    cases
                        .iter()
                        .filter_map(|case| case.metrics.ndcg_at_k)
                        .collect(),
                ),
                provenance_accuracy: average(
                    cases
                        .iter()
                        .filter_map(|case| case.metrics.provenance_accuracy)
                        .collect(),
                ),
                avg_chunks_per_case: cases
                    .iter()
                    .map(|case| case.metrics.chunks as f64)
                    .sum::<f64>()
                    / denom,
                avg_tokens_per_case: cases
                    .iter()
                    .map(|case| case.metrics.tokens as f64)
                    .sum::<f64>()
                    / denom,
                avg_latency_ms: total_latency_ms as f64 / denom,
                total_latency_ms,
            },
            cases,
            baseline: None,
        }
    }

    pub fn human_report(&self) -> String {
        let mut out = String::from("Eval summary:\n");
        out.push_str(&format!("  Cases total: {}\n", self.summary.cases_total));
        out.push_str(&format!(
            "  Cases with checks: {}\n",
            self.summary.cases_with_checks
        ));
        out.push_str(&format!(
            "  Cases passed/missed/unscored: {}/{}/{}\n",
            self.summary.cases_passed, self.summary.cases_missed, self.summary.cases_unscored
        ));
        for (label, value) in [
            ("Recall@k", self.summary.recall_at_k),
            ("Precision@k", self.summary.precision_at_k),
            ("MRR", self.summary.mrr),
            ("nDCG@k", self.summary.ndcg_at_k),
            ("Provenance accuracy", self.summary.provenance_accuracy),
        ] {
            if let Some(value) = value {
                out.push_str(&format!("  {label}: {:.4}\n", value));
            }
        }
        out.push_str(&format!(
            "  Avg chunks/case: {:.2}\n",
            self.summary.avg_chunks_per_case
        ));
        out.push_str(&format!(
            "  Avg tokens/case: {:.2}\n",
            self.summary.avg_tokens_per_case
        ));
        out.push_str(&format!(
            "  Avg latency/case: {:.2}ms\n",
            self.summary.avg_latency_ms
        ));
        if let Some(comparison) = &self.baseline {
            out.push_str("Baseline deltas:\n");
            for delta in &comparison.metrics {
                out.push_str(&format!(
                    "  {}: {:+.4} (baseline {:.4}, current {:.4})\n",
                    delta.metric, delta.delta, delta.baseline, delta.current
                ));
            }
            for regression in &comparison.regressions {
                out.push_str(&format!("  Regression: {regression}\n"));
            }
        }
        out
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BaselineMetricDelta {
    pub metric: String,
    pub baseline: f64,
    pub current: f64,
    /// Current minus baseline. Negative values are regressions.
    pub delta: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BaselineComparison {
    pub metrics: Vec<BaselineMetricDelta>,
    pub regressions: Vec<String>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum EvalOutputFormat {
    Human,
    Json,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RegressionMetric {
    RecallAtK,
    PrecisionAtK,
    Mrr,
    NdcgAtK,
    ProvenanceAccuracy,
}

impl RegressionMetric {
    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "recall" | "recall_at_k" => Some(Self::RecallAtK),
            "precision" | "precision_at_k" => Some(Self::PrecisionAtK),
            "mrr" | "reciprocal_rank" => Some(Self::Mrr),
            "ndcg" | "ndcg_at_k" => Some(Self::NdcgAtK),
            "provenance" | "provenance_accuracy" => Some(Self::ProvenanceAccuracy),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::RecallAtK => "recall_at_k",
            Self::PrecisionAtK => "precision_at_k",
            Self::Mrr => "mrr",
            Self::NdcgAtK => "ndcg_at_k",
            Self::ProvenanceAccuracy => "provenance_accuracy",
        }
    }

    fn value(self, summary: &EvalSummary) -> Option<f64> {
        match self {
            Self::RecallAtK => summary.recall_at_k,
            Self::PrecisionAtK => summary.precision_at_k,
            Self::Mrr => summary.mrr,
            Self::NdcgAtK => summary.ndcg_at_k,
            Self::ProvenanceAccuracy => summary.provenance_accuracy,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RegressionThreshold {
    pub metric: RegressionMetric,
    pub max_drop: f64,
}

pub fn parse_regression_thresholds(values: &[String]) -> Result<Vec<RegressionThreshold>> {
    values
        .iter()
        .map(|value| {
            let (metric, max_drop) = value.split_once('=').with_context(|| {
                format!("Invalid regression threshold `{value}`; use METRIC=MAX_DROP")
            })?;
            let metric = RegressionMetric::parse(metric)
                .with_context(|| format!("Unknown regression metric `{metric}`"))?;
            let max_drop = max_drop
                .parse::<f64>()
                .with_context(|| format!("Invalid maximum drop `{max_drop}` in `{value}`"))?;
            if !(0.0..=1.0).contains(&max_drop) {
                bail!("Maximum drop must be between 0 and 1 in `{value}`");
            }
            Ok(RegressionThreshold { metric, max_drop })
        })
        .collect()
}

pub fn compare_baseline(
    current: &EvalRunReport,
    baseline: &EvalRunReport,
    thresholds: &[RegressionThreshold],
) -> Result<Vec<String>> {
    let mut regressions = Vec::new();
    for threshold in thresholds {
        let metric = threshold.metric.label();
        let current_value = threshold.metric.value(&current.summary).with_context(|| {
            format!("Configured regression metric `{metric}` is unavailable in the current report")
        })?;
        let baseline_value = threshold.metric.value(&baseline.summary).with_context(|| {
            format!("Configured regression metric `{metric}` is unavailable in the baseline report")
        })?;
        let drop = baseline_value - current_value;
        if drop > threshold.max_drop {
            regressions.push(format!(
                "{} regressed by {:.4} (baseline {:.4}, current {:.4}, allowed {:.4})",
                metric, drop, baseline_value, current_value, threshold.max_drop
            ));
        }
    }
    Ok(regressions)
}

pub fn build_baseline_comparison(
    current: &EvalRunReport,
    baseline: &EvalRunReport,
    thresholds: &[RegressionThreshold],
) -> Result<BaselineComparison> {
    let metrics = [
        RegressionMetric::RecallAtK,
        RegressionMetric::PrecisionAtK,
        RegressionMetric::Mrr,
        RegressionMetric::NdcgAtK,
        RegressionMetric::ProvenanceAccuracy,
    ]
    .into_iter()
    .filter_map(|metric| {
        let baseline_value = metric.value(&baseline.summary)?;
        let current_value = metric.value(&current.summary)?;
        Some(BaselineMetricDelta {
            metric: metric.label().to_string(),
            baseline: baseline_value,
            current: current_value,
            delta: current_value - baseline_value,
        })
    })
    .collect();
    Ok(BaselineComparison {
        metrics,
        regressions: compare_baseline(current, baseline, thresholds)?,
    })
}

pub fn load_eval_cases(path: &Path) -> Result<Vec<EvalAugmentCase>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read eval file: {}", path.display()))?;
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    if trimmed.starts_with('[') {
        let values: Vec<Value> = serde_json::from_str(trimmed).with_context(|| {
            format!(
                "Failed to parse eval JSON array from file: {}",
                path.display()
            )
        })?;
        let cases = values
            .into_iter()
            .map(parse_eval_case)
            .collect::<Result<Vec<_>>>()?;
        return validate_case_versions(cases);
    }
    let cases = content
        .lines()
        .enumerate()
        .filter_map(|(index, line)| {
            let line = line.trim();
            (!line.is_empty() && !line.starts_with('#')).then_some((index + 1, line))
        })
        .map(|(line_number, line)| {
            serde_json::from_str::<Value>(line)
                .with_context(|| {
                    format!(
                        "Failed to parse eval JSON object at {}:{}",
                        path.display(),
                        line_number
                    )
                })
                .and_then(parse_eval_case)
        })
        .collect::<Result<Vec<_>>>()?;
    validate_case_versions(cases)
}

pub fn load_baseline(path: &Path) -> Result<EvalRunReport> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read baseline file: {}", path.display()))?;
    let report: EvalRunReport = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse baseline report: {}", path.display()))?;
    validate_baseline_version(report)
}

fn validate_baseline_version(report: EvalRunReport) -> Result<EvalRunReport> {
    if report.metadata.schema_version > EVAL_SCHEMA_VERSION {
        bail!(
            "Baseline report uses unsupported schema version {}; this binary supports up to {EVAL_SCHEMA_VERSION}",
            report.metadata.schema_version
        );
    }
    Ok(report)
}

fn normalize_id(value: &str) -> Option<String> {
    let normalized = value.trim().to_lowercase();
    (!normalized.is_empty()).then_some(normalized)
}

fn validate_case_versions(cases: Vec<EvalAugmentCase>) -> Result<Vec<EvalAugmentCase>> {
    for case in &cases {
        if let Some(version) = case.schema_version {
            if version > EVAL_SCHEMA_VERSION {
                bail!(
                    "Eval case `{}` uses unsupported schema version {version}; this binary supports up to {EVAL_SCHEMA_VERSION}",
                    case.display_name()
                );
            }
        }
        if let Some(grade) = case
            .relevance
            .iter()
            .filter_map(|item| item.grade)
            .find(|grade| *grade > MAX_NDCG_GRADE)
        {
            bail!(
                "Eval case `{}` has relevance grade {grade}, but grades must be at most {MAX_NDCG_GRADE}",
                case.display_name()
            );
        }
    }
    Ok(cases)
}

fn parse_eval_case(value: Value) -> Result<EvalAugmentCase> {
    if value.get("schema_version").is_some() {
        Ok(serde_json::from_value::<VersionedEvalAugmentCase>(value)?.into())
    } else {
        Ok(serde_json::from_value(value)?)
    }
}

fn ndcg_gain(grade: u32) -> f64 {
    2_f64.powi(grade.min(MAX_NDCG_GRADE) as i32) - 1.0
}

fn normalized_ids(values: &[String]) -> BTreeSet<String> {
    values
        .iter()
        .filter_map(|value| normalize_id(value))
        .collect()
}

fn normalized_strings(values: &[String]) -> BTreeSet<String> {
    values
        .iter()
        .map(|value| value.trim().to_lowercase())
        .filter(|value| !value.is_empty())
        .collect()
}

fn result_prompt_text(result: &RankedResult) -> String {
    result.text.to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(value: serde_json::Value) -> EvalAugmentCase {
        serde_json::from_value(value).unwrap()
    }

    fn result(id: &str) -> RankedResult {
        RankedResult {
            id: id.into(),
            text: String::new(),
            source_uri: None,
            conversation_uuid: None,
            approx_tokens: 10,
        }
    }

    #[test]
    fn computes_hand_checked_rank_metrics() {
        let case = case(serde_json::json!({
            "query": "q", "expected_ids": ["note:a", "note:b"],
            "relevance": [{"id": "note:a", "grade": 3}, {"id": "note:b", "grade": 1}]
        }));
        let metrics = evaluate_ranked_results(
            &case,
            &[result("note:x"), result("NOTE:A"), result("note:b")],
            3,
            7,
        );
        assert_eq!(metrics.recall_at_k, Some(1.0));
        assert!((metrics.precision_at_k.unwrap() - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(metrics.reciprocal_rank, Some(0.5));
        assert!((metrics.ndcg_at_k.unwrap() - 0.644_286_9).abs() < 1e-6);
        assert_eq!(metrics.latency_ms, 7);
    }

    #[test]
    fn ties_keep_input_order_and_missing_results_score_zero() {
        let case = case(serde_json::json!({"query": "q", "expected_ids": ["note:hit"]}));
        let metrics =
            evaluate_ranked_results(&case, &[result("note:miss"), result("note:hit")], 10, 0);
        assert_eq!(metrics.recall_at_k, Some(1.0));
        assert_eq!(metrics.precision_at_k, Some(0.1));
        assert_eq!(metrics.reciprocal_rank, Some(0.5));
        let missing = evaluate_ranked_results(&case, &[], 10, 0);
        assert_eq!(missing.recall_at_k, Some(0.0));
        assert_eq!(missing.precision_at_k, Some(0.0));
        assert_eq!(missing.reciprocal_rank, Some(0.0));
    }

    #[test]
    fn grade_zero_is_not_a_relevant_hit_and_budget_uses_all_chunks() {
        let case = case(serde_json::json!({
            "query": "q",
            "relevance": [{"id": "note:zero", "grade": 0}, {"id": "note:hit", "grade": 2}]
        }));
        let mut first = result("note:zero");
        first.approx_tokens = 3;
        let mut second = result("note:hit");
        second.approx_tokens = 5;

        let metrics = evaluate_ranked_results(&case, &[first, second], 1, 0);
        assert_eq!(metrics.relevant_total, 1);
        assert_eq!(metrics.recall_at_k, Some(0.0));
        assert_eq!(metrics.precision_at_k, Some(0.0));
        assert_eq!(metrics.reciprocal_rank, Some(0.0));
        assert_eq!(metrics.chunks, 2);
        assert_eq!(metrics.tokens, 8);
    }

    #[test]
    fn negative_and_provenance_checks_cover_the_full_augmentation_context() {
        let case = case(serde_json::json!({
            "query": "q",
            "expected_source_uris": ["file://allowed"],
            "forbidden_ids": ["note:forbidden"]
        }));
        let mut allowed = result("note:allowed");
        allowed.source_uri = Some("FILE://ALLOWED".into());
        let mut leaked = result("note:forbidden");
        leaked.source_uri = Some("file://unexpected".into());

        let metrics = evaluate_ranked_results(&case, &[allowed, leaked], 1, 0);
        assert!(metrics.forbidden_result_found);
        assert_eq!(metrics.provenance_accuracy, Some(0.5));
        assert_eq!(metrics.checks_passed, Some(false));
    }

    #[test]
    fn positive_substring_checks_cover_the_full_augmentation_context() {
        let case = case(serde_json::json!({
            "query": "q",
            "expected_contains": ["expected prompt text"]
        }));
        let mut after_k = result("note:later");
        after_k.text = "expected prompt text".into();

        let metrics = evaluate_ranked_results(&case, &[result("note:first"), after_k], 1, 0);
        assert_eq!(metrics.substring_expectation_matched, Some(true));
        assert_eq!(metrics.checks_passed, Some(true));
    }

    #[test]
    fn ungraded_cases_do_not_report_ndcg_and_empty_cases_are_unscored() {
        let ungraded = case(serde_json::json!({"query": "q", "expected_ids": ["note:a"]}));
        assert_eq!(
            evaluate_ranked_results(&ungraded, &[result("note:a")], 1, 0).ndcg_at_k,
            None
        );
        let empty = case(serde_json::json!({"query": "q"}));
        let metrics = evaluate_ranked_results(&empty, &[result("note:a")], 1, 0);
        assert_eq!(metrics.checks_passed, None);
        assert_eq!(metrics.recall_at_k, None);
    }

    #[test]
    fn exact_ids_and_negative_checks_are_normalized_and_text_is_separate() {
        let case = case(serde_json::json!({
            "query": "q", "expected_ids": [" NOTE:Wanted "], "expected_contains": ["needle"],
            "forbidden_ids": ["note:bad"], "forbidden_contains": ["secret"]
        }));
        let mut wanted = result("note:WANTED");
        wanted.text = "contains NEEDLE".into();
        let metrics = evaluate_ranked_results(&case, &[wanted], 1, 0);
        assert_eq!(metrics.recall_at_k, Some(1.0));
        assert_eq!(metrics.substring_expectation_matched, Some(true));
        assert!(!metrics.forbidden_result_found);
        assert_eq!(metrics.checks_passed, Some(true));
        let mut bad = result("note:bad");
        bad.text = "secret".into();
        assert!(evaluate_ranked_results(&case, &[bad], 1, 0).forbidden_result_found);

        let mut metadata_only = result("note:innocuous");
        metadata_only.source_uri = Some("file:///private/notes.md".into());
        assert!(!evaluate_ranked_results(&case, &[metadata_only], 1, 0).forbidden_result_found);
    }

    #[test]
    fn every_configured_positive_expectation_must_match() {
        let case = case(serde_json::json!({
            "query": "q",
            "expected_ids": ["note:expected"],
            "expected_contains": ["required prompt text"]
        }));
        let mut correct_id = result("note:expected");
        correct_id.text = "different text".into();
        let mut correct_text = result("note:different");
        correct_text.text = "required prompt text".into();

        assert_eq!(
            evaluate_ranked_results(&case, &[correct_id], 1, 0).checks_passed,
            Some(false)
        );
        assert_eq!(
            evaluate_ranked_results(&case, &[correct_text], 1, 0).checks_passed,
            Some(false)
        );
    }

    #[test]
    fn exact_ids_are_unicode_case_insensitive() {
        let case = case(serde_json::json!({
            "query": "q",
            "expected_ids": ["note:CAFÉ"],
            "forbidden_ids": ["note:СЕКРЕТ"]
        }));
        let hit = result("note:café");
        let forbidden = result("note:секрет");

        assert_eq!(
            evaluate_ranked_results(&case, &[hit], 1, 0).recall_at_k,
            Some(1.0)
        );
        assert!(evaluate_ranked_results(&case, &[forbidden], 1, 0).forbidden_result_found);
    }

    #[test]
    fn text_expectations_are_unicode_case_insensitive() {
        let case = case(serde_json::json!({
            "query": "q",
            "expected_contains": ["café"],
            "forbidden_contains": ["секрет"]
        }));
        let mut result = result("note:unicode");
        result.text = "CAFÉ СЕКРЕТ".into();

        let metrics = evaluate_ranked_results(&case, &[result], 1, 0);
        assert_eq!(metrics.substring_expectation_matched, Some(true));
        assert!(metrics.forbidden_result_found);
    }

    #[test]
    fn provenance_expectations_are_unicode_case_insensitive() {
        let case = case(serde_json::json!({
            "query": "q",
            "expected_source_uris": ["file:///CAFÉ.md"]
        }));
        let mut result = result("note:unicode");
        result.source_uri = Some("file:///café.md".into());

        assert_eq!(
            evaluate_ranked_results(&case, &[result], 1, 0).provenance_accuracy,
            Some(1.0)
        );
    }

    #[test]
    fn provenance_requires_every_configured_dimension() {
        let case = case(serde_json::json!({
            "query": "q",
            "expected_source_uris": ["file://allowed"],
            "expected_conversation_uuids": ["conversation-1"]
        }));
        let mut wrong_conversation = result("message:one");
        wrong_conversation.source_uri = Some("file://allowed".into());
        wrong_conversation.conversation_uuid = Some("conversation-2".into());
        let mut correct = result("message:two");
        correct.source_uri = Some("file://allowed".into());
        correct.conversation_uuid = Some("conversation-1".into());

        assert_eq!(
            evaluate_ranked_results(&case, &[wrong_conversation, correct.clone()], 2, 0)
                .provenance_accuracy,
            Some(0.5)
        );
        assert_eq!(
            evaluate_ranked_results(&case, &[correct], 1, 0).checks_passed,
            Some(true)
        );
    }

    #[test]
    fn versioned_cases_reject_unknown_fields_and_unsafe_grades() {
        let unknown_field = parse_eval_case(serde_json::json!({
            "schema_version": 2,
            "query": "q",
            "forbidden_contians": ["typo"]
        }))
        .unwrap_err();
        assert!(unknown_field.to_string().contains("unknown field"));

        let unsafe_grade = parse_eval_case(serde_json::json!({
            "schema_version": 2,
            "query": "q",
            "relevance": [{"id": "note:a", "grade": 64}]
        }))
        .unwrap();
        assert!(validate_case_versions(vec![unsafe_grade])
            .unwrap_err()
            .to_string()
            .contains("grades must be at most 63"));
        assert!(ndcg_gain(u32::MAX).is_finite());
    }

    #[test]
    fn parses_legacy_jsonl_shape() {
        let parsed: EvalAugmentCase = serde_json::from_str(
            r#"{"name":"old","query":"q","scope":"all","expected_contains":["x"]}"#,
        )
        .unwrap();
        assert_eq!(parsed.schema_version, None);
        assert_eq!(parsed.scope, Some(EvalScope::All));
        assert_eq!(parsed.expected_contains, ["x"]);
    }

    #[test]
    fn provenance_and_baseline_thresholds_work() {
        let case = case(serde_json::json!({"query":"q", "expected_conversation_uuids":["chat-1"]}));
        let mut hit = result("message:1");
        hit.conversation_uuid = Some("CHAT-1".into());
        assert_eq!(
            evaluate_ranked_results(&case, &[hit.clone()], 1, 0).provenance_accuracy,
            Some(1.0)
        );

        let metadata = EvalMetadata {
            schema_version: EVAL_SCHEMA_VERSION,
            provider: "test".into(),
            model: "test".into(),
        };
        let baseline = EvalRunReport::from_cases(
            metadata.clone(),
            vec![EvalCaseReport {
                name: "q".into(),
                query: "q".into(),
                metrics: evaluate_ranked_results(&case, &[hit], 1, 0),
            }],
        );
        let current = EvalRunReport::from_cases(
            metadata,
            vec![EvalCaseReport {
                name: "q".into(),
                query: "q".into(),
                metrics: evaluate_ranked_results(&case, &[], 1, 0),
            }],
        );
        let thresholds = parse_regression_thresholds(&["provenance=0.1".into()]).unwrap();
        assert_eq!(
            compare_baseline(&current, &baseline, &thresholds)
                .unwrap()
                .len(),
            1
        );
        let permissive = parse_regression_thresholds(&["provenance=1.0".into()]).unwrap();
        assert!(compare_baseline(&current, &baseline, &permissive)
            .unwrap()
            .is_empty());
        assert!(parse_regression_thresholds(&["recall=1.2".into()]).is_err());

        let ndcg = parse_regression_thresholds(&["ndcg=0.1".into()]).unwrap();
        assert!(compare_baseline(&current, &baseline, &ndcg).is_err());
    }

    #[test]
    fn json_report_round_trips_stably() {
        let report = EvalRunReport::from_cases(
            EvalMetadata {
                schema_version: EVAL_SCHEMA_VERSION,
                provider: "fixture".into(),
                model: "fixture-v1".into(),
            },
            vec![],
        );
        let json = serde_json::to_string_pretty(&report).unwrap();
        assert_eq!(
            serde_json::from_str::<EvalRunReport>(&json).unwrap(),
            report
        );
        assert!(json.contains("\"schema_version\": 2"));
    }

    #[test]
    fn baseline_comparison_lists_each_available_metric_delta() {
        let case = case(serde_json::json!({"query": "q", "expected_ids": ["note:a"]}));
        let metadata = EvalMetadata {
            schema_version: EVAL_SCHEMA_VERSION,
            provider: "fixture".into(),
            model: "fixture".into(),
        };
        let baseline = EvalRunReport::from_cases(
            metadata.clone(),
            vec![EvalCaseReport {
                name: "q".into(),
                query: "q".into(),
                metrics: evaluate_ranked_results(&case, &[result("note:a")], 1, 0),
            }],
        );
        let current = EvalRunReport::from_cases(
            metadata,
            vec![EvalCaseReport {
                name: "q".into(),
                query: "q".into(),
                metrics: evaluate_ranked_results(&case, &[], 1, 0),
            }],
        );
        let comparison = build_baseline_comparison(
            &current,
            &baseline,
            &parse_regression_thresholds(&["recall=0.5".into()]).unwrap(),
        )
        .unwrap();
        assert_eq!(comparison.metrics.len(), 3);
        assert_eq!(comparison.regressions.len(), 1);
        assert!(comparison.metrics.iter().all(|delta| delta.delta <= 0.0));
    }

    #[test]
    fn sanitized_fixture_covers_all_scopes_and_v2_features() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/eval/cases-v2.jsonl");
        let cases = load_eval_cases(&path).unwrap();
        assert_eq!(cases.len(), 6);
        assert!(cases
            .iter()
            .any(|case| case.scope == Some(EvalScope::Notes)));
        assert!(cases
            .iter()
            .any(|case| case.scope == Some(EvalScope::Messages)));
        assert!(cases.iter().any(|case| case.scope == Some(EvalScope::All)));
        assert!(cases.iter().any(|case| !case.forbidden_ids.is_empty()));
        assert!(cases
            .iter()
            .any(|case| !case.expected_conversation_uuids.is_empty()));
    }

    #[test]
    fn rejects_future_case_schema_versions() {
        let future = case(serde_json::json!({"schema_version": 3, "query": "q"}));
        assert!(validate_case_versions(vec![future]).is_err());
    }

    #[test]
    fn rejects_baselines_from_newer_evaluator_schemas() {
        let report = EvalRunReport::from_cases(
            EvalMetadata {
                schema_version: EVAL_SCHEMA_VERSION + 1,
                provider: "future".into(),
                model: "future".into(),
            },
            vec![],
        );
        assert!(validate_baseline_version(report).is_err());
    }
}
