//! Synthesis provider boundary. Production uses the existing `[inference]`
//! text-generation endpoint (TGI or Ollama, the extraction provider); tests
//! use [`FixtureSynthesisGenerator`]. The provider only drafts: every claim it
//! returns is checked by the Rust verifier before anyone sees it.

use super::gather::EvidenceSet;
use super::lineage::Lineage;
use super::{SynthesisInput, SynthesisPlan, EVIDENCE_UNASSESSED, MIN_QUOTE_CHARS};
use crate::{AgentError, InferenceCapabilities, InferenceProviderConfig, Result};
use async_trait::async_trait;
use graphrag_core::record_id_to_string;
use reqwest::Client;
use serde_json::{json, Value};
use std::fmt::Write as _;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Upper bound on a provider HTTP response body.
const MAX_PROVIDER_RESPONSE_BYTES: usize = 1024 * 1024;

const SYSTEM_PROMPT: &str = "You are the Alchemist, a read-only synthesis assistant for a personal notes graph. Explain only what the numbered evidence chunks say. Every claim must cite at least one chunk label and copy an exact, verbatim quote from that chunk; never paraphrase or join text inside a quote. Do not resolve disagreements: list conflicting claims by index under conflicts. Do not propose new relations, edges or notes. Put unanswered questions under open_questions. Return only JSON that matches the schema.";

/// One bounded generation call.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationRequest {
    pub system: String,
    pub prompt: String,
    pub max_output_tokens: usize,
    /// JSON schema the provider should constrain its output to.
    pub schema: Value,
}

/// Drafts a structured synthesis. Implementations must not persist anything.
#[async_trait]
pub trait SynthesisGenerator: Send + Sync {
    async fn generate(&self, request: &GenerationRequest) -> Result<String>;
    fn capabilities(&self) -> InferenceCapabilities;
}

pub type SharedSynthesisGenerator = Arc<dyn SynthesisGenerator>;

/// The draft schema shared by the prompt and provider-side constraints.
pub fn draft_json_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "claims": {"type": "array", "items": {"type": "object", "properties": {
                "text": {"type": "string"},
                "citations": {"type": "array", "items": {"type": "object", "properties": {
                    "chunk": {"type": "string"},
                    "quote": {"type": "string"}
                }, "required": ["chunk", "quote"]}}
            }, "required": ["text", "citations"]}},
            "conflicts": {"type": "array", "items": {"type": "object", "properties": {
                "topic": {"type": "string"},
                "claims": {"type": "array", "items": {"type": "integer"}}
            }, "required": ["topic", "claims"]}},
            "open_questions": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["claims", "conflicts", "open_questions"]
    })
}

/// Render the bounded prompt: pinned excerpts, their origins and the
/// accepted relations among them.
pub(crate) fn build_request(
    plan: &SynthesisPlan,
    evidence: &EvidenceSet,
    lineage: &Lineage,
) -> GenerationRequest {
    let mut prompt = String::new();
    match &plan.input {
        SynthesisInput::Query { query, .. } => {
            let _ = writeln!(prompt, "Question: {query}");
        }
        SynthesisInput::Seed { .. } => {
            let _ = writeln!(
                prompt,
                "Explain what the notes around seed chunk C1 collectively say."
            );
        }
    }
    let _ = writeln!(
        prompt,
        "\nEvidence chunks (cite by label; quotes must be copied verbatim, at least {MIN_QUOTE_CHARS} characters):"
    );
    for (index, chunk) in evidence.chunks.iter().enumerate() {
        let _ = write!(prompt, "\n[{}] {}", chunk.label, chunk.id);
        if let Some(title) = &chunk.title {
            let _ = write!(prompt, " | title: {title}");
        }
        if let Some(date) = chunk.created_at {
            let _ = write!(prompt, " | note date: {}", date.date_naive());
        }
        let _ = writeln!(
            prompt,
            " | origin: {}\n<<<\n{}\n>>>",
            lineage.origin_label(index),
            chunk.text
        );
    }
    if !evidence.relations.is_empty() {
        let label = |id: &str| {
            evidence
                .chunks
                .iter()
                .find(|chunk| chunk.id == id)
                .map_or(id.to_string(), |chunk| chunk.label.clone())
        };
        let _ = writeln!(
            prompt,
            "\nAccepted relations among the evidence (reviewed edges; evidence {EVIDENCE_UNASSESSED}):"
        );
        for edge in &evidence.relations {
            let _ = writeln!(
                prompt,
                "- {} {} {}",
                label(&record_id_to_string(&edge.in_id)),
                edge.edge_type,
                label(&record_id_to_string(&edge.out_id)),
            );
        }
        let _ = writeln!(prompt, "Chunks sharing an origin are copies, summaries or parts of one source; they do not corroborate each other.");
    }
    let _ = writeln!(
        prompt,
        "\nReturn JSON: {{\"claims\":[{{\"text\":string,\"citations\":[{{\"chunk\":label,\"quote\":verbatim}}]}}],\"conflicts\":[{{\"topic\":string,\"claims\":[claim index]}}],\"open_questions\":[string]}}"
    );
    GenerationRequest {
        system: SYSTEM_PROMPT.into(),
        prompt,
        max_output_tokens: plan.budgets.max_output_tokens,
        schema: draft_json_schema(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Provider {
    Tgi,
    Ollama,
}

/// Text generation through the configured `[inference]` extraction
/// provider and model. Opt-in: it is only constructed when `[alchemist]` is
/// enabled.
#[derive(Clone)]
pub struct ConfiguredSynthesisGenerator {
    client: Client,
    provider: Provider,
    base_url: String,
    model: String,
    ollama_options: Option<Value>,
}

impl ConfiguredSynthesisGenerator {
    pub fn from_config(config: &InferenceProviderConfig) -> Self {
        let provider = if config.extraction_provider.eq_ignore_ascii_case("ollama") {
            Provider::Ollama
        } else {
            Provider::Tgi
        };
        let timeout = match provider {
            Provider::Ollama => config.ollama_timeout_secs,
            Provider::Tgi => config.timeout_secs,
        };
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(timeout.max(1)))
                .build()
                .expect("valid inference HTTP client configuration"),
            provider,
            base_url: config.extraction_url.trim_end_matches('/').to_string(),
            model: config.extraction_model.clone(),
            ollama_options: config.ollama_options.clone(),
        }
    }

    async fn post(&self, path: &str, body: Value) -> Result<Value> {
        let response = self
            .client
            .post(format!("{}{path}", self.base_url))
            .json(&body)
            .send()
            .await?
            .error_for_status()?;
        let bytes = response.bytes().await?;
        if bytes.len() > MAX_PROVIDER_RESPONSE_BYTES {
            return Err(AgentError::InferenceService(
                "synthesis provider response exceeds 1 MiB".into(),
            ));
        }
        serde_json::from_slice(&bytes).map_err(|error| {
            AgentError::InferenceService(format!(
                "synthesis provider returned invalid JSON: {error}"
            ))
        })
    }
}

#[async_trait]
impl SynthesisGenerator for ConfiguredSynthesisGenerator {
    async fn generate(&self, request: &GenerationRequest) -> Result<String> {
        let text = match self.provider {
            Provider::Ollama => {
                let mut options = match &self.ollama_options {
                    Some(Value::Object(options)) => options.clone(),
                    _ => serde_json::Map::new(),
                };
                options.insert("num_predict".into(), json!(request.max_output_tokens));
                options.insert("temperature".into(), json!(0));
                let response = self
                    .post(
                        "/api/chat",
                        json!({
                            "model": self.model,
                            "messages": [
                                {"role": "system", "content": request.system},
                                {"role": "user", "content": request.prompt}
                            ],
                            "stream": false,
                            "format": request.schema,
                            "options": options
                        }),
                    )
                    .await?;
                response["message"]["content"].as_str().map(str::to_string)
            }
            Provider::Tgi => {
                let response = self
                    .post(
                        "/generate",
                        json!({
                            "inputs": format!("{}\n\n{}", request.system, request.prompt),
                            "parameters": {
                                "max_new_tokens": request.max_output_tokens,
                                "return_full_text": false,
                                "grammar": {"type": "json", "value": request.schema}
                            }
                        }),
                    )
                    .await?;
                response
                    .get("generated_text")
                    .or_else(|| {
                        response
                            .get(0)
                            .and_then(|first| first.get("generated_text"))
                    })
                    .and_then(Value::as_str)
                    .map(str::to_string)
            }
        };
        text.ok_or_else(|| {
            AgentError::InferenceService("synthesis provider returned no generated text".into())
        })
    }

    fn capabilities(&self) -> InferenceCapabilities {
        let provider = match self.provider {
            Provider::Tgi => "tgi",
            Provider::Ollama => "ollama",
        };
        InferenceCapabilities {
            provider: provider.into(),
            model: self.model.clone(),
            endpoint: self.base_url.clone(),
            known_dimension: None,
            cache_identity: format!("alchemist-synthesis-v1;provider={provider}"),
        }
    }
}

type Responder = dyn Fn(&GenerationRequest) -> Result<String> + Send + Sync;

/// Deterministic offline provider for tests and fixtures. It records every
/// request so tests can prove which evidence the model was shown.
#[derive(Clone)]
pub struct FixtureSynthesisGenerator {
    responder: Arc<Responder>,
    delay: Option<Duration>,
    requests: Arc<Mutex<Vec<GenerationRequest>>>,
}

impl FixtureSynthesisGenerator {
    pub fn new(
        responder: impl Fn(&GenerationRequest) -> Result<String> + Send + Sync + 'static,
    ) -> Self {
        Self {
            responder: Arc::new(responder),
            delay: None,
            requests: Arc::default(),
        }
    }

    /// Always return `response`.
    pub fn fixed(response: impl Into<String>) -> Self {
        let response = response.into();
        Self::new(move |_| Ok(response.clone()))
    }

    /// Sleep before answering, to exercise deadlines.
    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = Some(delay);
        self
    }

    pub fn requests(&self) -> Vec<GenerationRequest> {
        self.requests.lock().expect("fixture request log").clone()
    }
}

#[async_trait]
impl SynthesisGenerator for FixtureSynthesisGenerator {
    async fn generate(&self, request: &GenerationRequest) -> Result<String> {
        self.requests
            .lock()
            .expect("fixture request log")
            .push(request.clone());
        if let Some(delay) = self.delay {
            tokio::time::sleep(delay).await;
        }
        (self.responder)(request)
    }

    fn capabilities(&self) -> InferenceCapabilities {
        InferenceCapabilities {
            provider: "fixture".into(),
            model: "fixture-synthesis".into(),
            endpoint: "fixture://synthesis".into(),
            known_dimension: None,
            cache_identity: "alchemist-synthesis-v1;provider=fixture".into(),
        }
    }
}
