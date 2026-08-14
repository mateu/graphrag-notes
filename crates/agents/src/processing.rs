//! Bounded, retrying, cache-aware local inference adapters.
//!
//! This module deliberately contains no queue/daemon. Callers retain ownership
//! of item ordering and durable checkpoints while these adapters provide the
//! provider-wide safety envelope shared by imports and extraction passes.

use crate::{
    AgentError, Embedder, EntityExtraction, EntityExtractor, InferenceCapabilities, Result,
};
use async_trait::async_trait;
use graphrag_db::{InferenceCacheEntry, Repository};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::future::Future;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use std::time::Duration;
use tokio::sync::Semaphore;
use unicode_normalization::UnicodeNormalization;

const EMBEDDING_CACHE_VERSION: &str = "embedding-v1";
const EXTRACTION_CACHE_VERSION: &str = "entity-extraction-v1";

/// Provider-wide limits for one class of local inference operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessingConfig {
    pub concurrency: usize,
    pub request_timeout: Duration,
    /// Includes the initial request. A value of one disables retries.
    pub retry_attempts: usize,
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
    pub use_cache: bool,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            concurrency: 4,
            request_timeout: Duration::from_secs(30),
            retry_attempts: 3,
            initial_backoff: Duration::from_millis(250),
            max_backoff: Duration::from_secs(5),
            use_cache: true,
        }
    }
}

impl ProcessingConfig {
    pub fn normalized(mut self) -> Self {
        self.concurrency = self.concurrency.max(1);
        self.retry_attempts = self.retry_attempts.max(1);
        self
    }
}

/// Counters intended for stable command summaries rather than tracing-only
/// observability. A clone shares counters with all adapter clones.
#[derive(Debug, Default)]
pub struct ProcessingStats {
    requests: AtomicU64,
    retries: AtomicU64,
    cache_hits: AtomicU64,
    failures: AtomicU64,
    jitter_seeds: AtomicU64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ProcessingStatsSnapshot {
    pub requests: u64,
    pub retries: u64,
    pub cache_hits: u64,
    pub failures: u64,
}

impl ProcessingStats {
    pub fn snapshot(&self) -> ProcessingStatsSnapshot {
        ProcessingStatsSnapshot {
            requests: self.requests.load(Ordering::Relaxed),
            retries: self.retries.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            failures: self.failures.load(Ordering::Relaxed),
        }
    }
}

/// Retry classes are deliberately conservative. Invalid model output and
/// compatibility failures remain immediate failures instead of repeated load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryClassification {
    Transient,
    Permanent,
}

pub fn classify_retry(error: &AgentError) -> RetryClassification {
    match error {
        AgentError::Http(error) => {
            if error.is_timeout() || error.is_connect() {
                return RetryClassification::Transient;
            }
            match error.status().map(|status| status.as_u16()) {
                Some(429) | Some(500..=599) => RetryClassification::Transient,
                _ => RetryClassification::Permanent,
            }
        }
        AgentError::InferenceService(message) => classify_message(message),
        // These categories include dimension checks and deterministic parser
        // failures, both explicitly excluded from retries.
        AgentError::Database(_) | AgentError::NotFound(_) | AgentError::Processing(_) => {
            RetryClassification::Permanent
        }
    }
}

fn classify_message(message: &str) -> RetryClassification {
    let message = message.to_ascii_lowercase();
    if message.contains("timeout")
        || message.contains("timed out")
        || message.contains("connection")
        || message.contains("429")
        || message.contains("too many requests")
        || message.contains("rate limit")
        || message.contains(" 5")
        || message.starts_with("5")
    {
        RetryClassification::Transient
    } else {
        RetryClassification::Permanent
    }
}

/// Deterministic, bounded jitter for a retry attempt. Determinism keeps unit
/// tests and automation reproducible while avoiding synchronized retry storms.
pub fn retry_delay(config: &ProcessingConfig, retry_index: usize) -> Duration {
    retry_delay_with_seed(config, retry_index, 0)
}

/// A deterministic seed gives concurrent operations different bounded jitter
/// without making tests or automation nondeterministic.
pub fn retry_delay_with_seed(config: &ProcessingConfig, retry_index: usize, seed: u64) -> Duration {
    let exponent = u32::try_from(retry_index.saturating_sub(1))
        .unwrap_or(u32::MAX)
        .min(20);
    let base = config
        .initial_backoff
        .checked_mul(1_u32 << exponent)
        .unwrap_or(config.max_backoff)
        .min(config.max_backoff);
    if base.is_zero() {
        return base;
    }
    let nanos = base.as_nanos();
    // 80..120% jitter, derived solely from the attempt number.
    let factor = 80_u128 + ((retry_index as u128 * 73 + u128::from(seed) * 31 + 19) % 41);
    let jittered = nanos.saturating_mul(factor) / 100;
    Duration::from_nanos(u64::try_from(jittered.min(u128::from(u64::MAX))).unwrap_or(u64::MAX))
        .min(config.max_backoff)
}

async fn execute_with_retry<T, F, Fut>(
    config: &ProcessingConfig,
    limiter: &Semaphore,
    stats: &ProcessingStats,
    mut operation: F,
) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    let _permit = limiter
        .acquire()
        .await
        .map_err(|_| AgentError::Processing("inference concurrency limiter closed".into()))?;
    let mut attempt = 1_usize;
    let jitter_seed = stats.jitter_seeds.fetch_add(1, Ordering::Relaxed);
    loop {
        stats.requests.fetch_add(1, Ordering::Relaxed);
        let result = tokio::time::timeout(config.request_timeout, operation())
            .await
            .unwrap_or_else(|_| {
                Err(AgentError::InferenceService(
                    "inference request timed out".into(),
                ))
            });
        match result {
            Ok(value) => return Ok(value),
            Err(error)
                if classify_retry(&error) == RetryClassification::Transient
                    && attempt < config.retry_attempts =>
            {
                stats.retries.fetch_add(1, Ordering::Relaxed);
                tokio::time::sleep(retry_delay_with_seed(config, attempt, jitter_seed)).await;
                attempt += 1;
            }
            Err(error) => {
                stats.failures.fetch_add(1, Ordering::Relaxed);
                return Err(error);
            }
        }
    }
}

#[derive(Debug, Clone)]
struct CacheKey {
    key: String,
    operation: &'static str,
    provider: String,
    model: String,
    version: String,
    input_hash: String,
}

impl CacheKey {
    fn new(
        operation: &'static str,
        capability: &InferenceCapabilities,
        version: String,
        input: &str,
    ) -> Self {
        // NFC plus newline normalization makes canonically equivalent content
        // share an entry without conflating meaningful internal whitespace.
        let normalized = input
            .replace("\r\n", "\n")
            .replace('\r', "\n")
            .nfc()
            .collect::<String>();
        let input_hash = hash(&normalized);
        let semantic = format!(
            "operation={operation}\0provider={}\0endpoint={}\0model={}\0provider_settings={}\0version={version}\0input={input_hash}",
            capability.provider, capability.endpoint, capability.model, capability.cache_identity
        );
        Self {
            key: hash(&semantic),
            operation,
            provider: capability.provider.clone(),
            model: capability.model.clone(),
            version,
            input_hash,
        }
    }

    fn entry(&self, value: serde_json::Value) -> InferenceCacheEntry {
        InferenceCacheEntry {
            cache_key: self.key.clone(),
            operation: self.operation.to_string(),
            provider: self.provider.clone(),
            model: self.model.clone(),
            version: self.version.clone(),
            input_hash: self.input_hash.clone(),
            value,
        }
    }
}

fn hash(input: &str) -> String {
    let digest = Sha256::digest(input.as_bytes());
    format!("{digest:x}")
}

/// Cache-aware, bounded embedding adapter. It caches each input separately so
/// partial batch hits retain their value and preserve caller ordering.
#[derive(Clone)]
pub struct ResilientEmbedder {
    inner: Arc<dyn Embedder>,
    cache: Option<Repository>,
    config: ProcessingConfig,
    limiter: Arc<Semaphore>,
    stats: Arc<ProcessingStats>,
}

impl ResilientEmbedder {
    pub fn new(
        inner: Arc<dyn Embedder>,
        cache: Option<Repository>,
        config: ProcessingConfig,
    ) -> Self {
        let config = config.normalized();
        Self {
            inner,
            cache,
            limiter: Arc::new(Semaphore::new(config.concurrency)),
            config,
            stats: Arc::new(ProcessingStats::default()),
        }
    }

    pub fn stats(&self) -> ProcessingStatsSnapshot {
        self.stats.snapshot()
    }

    fn key(&self, text: &str, is_query: bool) -> CacheKey {
        let role = if is_query { "query" } else { "passage" };
        CacheKey::new(
            "embedding",
            &self.inner.capabilities(),
            format!("{EMBEDDING_CACHE_VERSION}:{role}"),
            text,
        )
    }

    async fn cache_embedding(&self, key: &CacheKey) -> Result<Option<Vec<f32>>> {
        let Some(cache) = &self.cache else {
            return Ok(None);
        };
        if !self.config.use_cache {
            return Ok(None);
        }
        let Some(value) = cache.get_inference_cache(&key.key).await? else {
            return Ok(None);
        };
        let cached: CachedEmbedding = serde_json::from_value(value).map_err(|error| {
            AgentError::Processing(format!("invalid cached embedding: {error}"))
        })?;
        self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
        Ok(Some(cached.embedding))
    }

    async fn store_embedding(&self, key: &CacheKey, embedding: &[f32]) -> Result<()> {
        if self.config.use_cache {
            if let Some(cache) = &self.cache {
                cache
                    .put_inference_cache(
                        key.entry(
                            serde_json::to_value(CachedEmbedding {
                                embedding: embedding.to_vec(),
                            })
                            .map_err(|error| {
                                AgentError::Processing(format!(
                                    "serialize embedding cache value: {error}"
                                ))
                            })?,
                        ),
                    )
                    .await?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct CachedEmbedding {
    embedding: Vec<f32>,
}

#[async_trait]
impl Embedder for ResilientEmbedder {
    async fn embed(&self, text: &str, is_query: bool) -> Result<Vec<f32>> {
        let key = self.key(text, is_query);
        if let Some(embedding) = self.cache_embedding(&key).await? {
            return Ok(embedding);
        }
        let text = text.to_string();
        let inner = self.inner.clone();
        let embedding = execute_with_retry(&self.config, &self.limiter, &self.stats, || {
            let inner = inner.clone();
            let text = text.clone();
            async move { inner.embed(&text, is_query).await }
        })
        .await?;
        self.store_embedding(&key, &embedding).await?;
        Ok(embedding)
    }

    async fn embed_batch(&self, texts: &[String], is_query: bool) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        // Ollama embeds one prompt per request. Route through `embed` so each
        // provider request owns its timeout, retry budget, cache lookup, and
        // semaphore permit.
        if self
            .inner
            .capabilities()
            .provider
            .eq_ignore_ascii_case("ollama")
        {
            let mut embeddings = Vec::with_capacity(texts.len());
            for text in texts {
                embeddings.push(self.embed(text, is_query).await?);
            }
            return Ok(embeddings);
        }
        let keys = texts
            .iter()
            .map(|text| self.key(text, is_query))
            .collect::<Vec<_>>();
        let mut output = vec![None; texts.len()];
        let mut misses = Vec::new();
        let mut miss_positions: HashMap<String, Vec<usize>> = HashMap::new();
        for (index, key) in keys.iter().enumerate() {
            if let Some(embedding) = self.cache_embedding(key).await? {
                output[index] = Some(embedding);
            } else if let Some(indices) = miss_positions.get_mut(&key.key) {
                indices.push(index);
            } else {
                miss_positions.insert(key.key.clone(), vec![index]);
                misses.push(index);
            }
        }
        if !misses.is_empty() {
            let inputs = misses
                .iter()
                .map(|index| texts[*index].clone())
                .collect::<Vec<_>>();
            let batch_size = self.inner.max_batch_size().unwrap_or(inputs.len()).max(1);
            for (chunk_index, chunk) in inputs.chunks(batch_size).enumerate() {
                let inner = self.inner.clone();
                let chunk = chunk.to_vec();
                let chunk_embeddings =
                    execute_with_retry(&self.config, &self.limiter, &self.stats, || {
                        let inner = inner.clone();
                        let chunk = chunk.clone();
                        async move { inner.embed_batch(&chunk, is_query).await }
                    })
                    .await?;
                if chunk_embeddings.len() != chunk.len() {
                    return Err(AgentError::Processing(format!(
                        "embedding provider returned {} embeddings for a batch of {} inputs",
                        chunk_embeddings.len(),
                        chunk.len()
                    )));
                }
                let start = chunk_index * batch_size;
                for (offset, embedding) in chunk_embeddings.into_iter().enumerate() {
                    let miss_index = misses[start + offset];
                    self.store_embedding(&keys[miss_index], &embedding).await?;
                    for position in &miss_positions[&keys[miss_index].key] {
                        output[*position] = Some(embedding.clone());
                    }
                }
            }
        }
        output
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                value.ok_or_else(|| {
                    AgentError::Processing(format!("missing embedding result at index {index}"))
                })
            })
            .collect()
    }

    async fn health(&self) -> Result<bool> {
        self.inner.health().await
    }
    fn capabilities(&self) -> InferenceCapabilities {
        self.inner.capabilities()
    }
}

/// Cache-aware, bounded entity-extraction adapter. Structured extraction has
/// its own namespace and schema version, so it can never collide with vectors.
#[derive(Clone)]
pub struct ResilientEntityExtractor {
    inner: Arc<dyn EntityExtractor>,
    cache: Option<Repository>,
    config: ProcessingConfig,
    limiter: Arc<Semaphore>,
    stats: Arc<ProcessingStats>,
}

impl ResilientEntityExtractor {
    pub fn new(
        inner: Arc<dyn EntityExtractor>,
        cache: Option<Repository>,
        config: ProcessingConfig,
    ) -> Self {
        let config = config.normalized();
        Self {
            inner,
            cache,
            limiter: Arc::new(Semaphore::new(config.concurrency)),
            config,
            stats: Arc::new(ProcessingStats::default()),
        }
    }
    pub fn stats(&self) -> ProcessingStatsSnapshot {
        self.stats.snapshot()
    }
    fn key(&self, text: &str) -> CacheKey {
        CacheKey::new(
            "entity_extraction",
            &self.inner.capabilities(),
            EXTRACTION_CACHE_VERSION.into(),
            text,
        )
    }
}

#[async_trait]
impl EntityExtractor for ResilientEntityExtractor {
    async fn extract(&self, text: &str) -> Result<EntityExtraction> {
        let key = self.key(text);
        if self.config.use_cache {
            if let Some(cache) = &self.cache {
                if let Some(value) = cache.get_inference_cache(&key.key).await? {
                    let extraction = serde_json::from_value(value).map_err(|error| {
                        AgentError::Processing(format!("invalid cached extraction: {error}"))
                    })?;
                    self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
                    return Ok(extraction);
                }
            }
        }
        // Ollama strict JSON extraction can issue several progressively larger
        // generation requests. Retain cache, semaphore, counters, and retry
        // accounting, but do not place one timeout around that whole sequence:
        // the client owns the configured timeout for each HTTP request.
        if self
            .inner
            .capabilities()
            .provider
            .eq_ignore_ascii_case("ollama")
        {
            let _permit = self.limiter.acquire().await.map_err(|_| {
                AgentError::Processing("inference concurrency limiter closed".into())
            })?;
            let mut attempt = 1_usize;
            loop {
                self.stats.requests.fetch_add(1, Ordering::Relaxed);
                match self.inner.extract(text).await {
                    Ok(extraction) => {
                        if self.config.use_cache {
                            if let Some(cache) = &self.cache {
                                cache
                                    .put_inference_cache(key.entry(
                                        serde_json::to_value(&extraction).map_err(|error| {
                                            AgentError::Processing(format!(
                                                "serialize extraction cache value: {error}"
                                            ))
                                        })?,
                                    ))
                                    .await?;
                            }
                        }
                        return Ok(extraction);
                    }
                    Err(error)
                        if classify_retry(&error) == RetryClassification::Transient
                            && attempt < self.config.retry_attempts =>
                    {
                        self.stats.retries.fetch_add(1, Ordering::Relaxed);
                        let seed = self.stats.jitter_seeds.fetch_add(1, Ordering::Relaxed);
                        tokio::time::sleep(retry_delay_with_seed(&self.config, attempt, seed))
                            .await;
                        attempt += 1;
                    }
                    Err(error) => {
                        self.stats.failures.fetch_add(1, Ordering::Relaxed);
                        return Err(error);
                    }
                }
            }
        }
        let text = text.to_string();
        let inner = self.inner.clone();
        let extraction = execute_with_retry(&self.config, &self.limiter, &self.stats, || {
            let inner = inner.clone();
            let text = text.clone();
            async move { inner.extract(&text).await }
        })
        .await?;
        if self.config.use_cache {
            if let Some(cache) = &self.cache {
                cache
                    .put_inference_cache(key.entry(serde_json::to_value(&extraction).map_err(
                        |error| {
                            AgentError::Processing(format!(
                                "serialize extraction cache value: {error}"
                            ))
                        },
                    )?))
                    .await?;
            }
        }
        Ok(extraction)
    }
    async fn health(&self) -> Result<bool> {
        self.inner.health().await
    }
    fn capabilities(&self) -> InferenceCapabilities {
        self.inner.capabilities()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DeterministicEmbedder, FixtureEntityExtractor};
    use graphrag_db::init_memory;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn retry_table_is_conservative() {
        assert_eq!(
            classify_retry(&AgentError::InferenceService("HTTP 429".into())),
            RetryClassification::Transient
        );
        assert_eq!(
            classify_retry(&AgentError::InferenceService(
                "invalid JSON response".into()
            )),
            RetryClassification::Permanent
        );
        assert_eq!(
            classify_retry(&AgentError::Processing("dimension mismatch".into())),
            RetryClassification::Permanent
        );
    }

    #[test]
    fn delay_is_bounded_and_grows() {
        let config = ProcessingConfig {
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_millis(500),
            ..Default::default()
        };
        assert!(retry_delay(&config, 1) >= Duration::from_millis(80));
        assert!(retry_delay(&config, 3) <= Duration::from_millis(500));
        assert_ne!(
            retry_delay_with_seed(&config, 1, 1),
            retry_delay_with_seed(&config, 1, 2)
        );
    }

    #[tokio::test]
    async fn embedding_cache_key_changes_for_model_role_and_content() {
        let db = init_memory().await.unwrap();
        let repo = Repository::new(db);
        let inner: Arc<dyn Embedder> =
            Arc::new(DeterministicEmbedder::default().with_identity("fixture", "model-a"));
        let adapter = ResilientEmbedder::new(
            inner,
            Some(repo),
            ProcessingConfig {
                initial_backoff: Duration::ZERO,
                ..Default::default()
            },
        );
        let first = adapter.embed("café", false).await.unwrap();
        let equivalent = adapter.embed("cafe\u{301}", false).await.unwrap();
        assert_eq!(first, equivalent);
        assert_eq!(adapter.stats().cache_hits, 1);
        let query = adapter.embed("café", true).await.unwrap();
        assert_ne!(query, first);
        assert_eq!(
            adapter.stats().cache_hits,
            1,
            "query and passage cache namespaces differ"
        );
    }

    #[tokio::test]
    async fn structured_cache_key_includes_schema_and_provider_settings() {
        let repo = Repository::new(init_memory().await.unwrap());
        let first: Arc<dyn EntityExtractor> = Arc::new(
            FixtureEntityExtractor::default().with_cache_identity("strict=true;max_entities=5"),
        );
        let first =
            ResilientEntityExtractor::new(first, Some(repo.clone()), ProcessingConfig::default());
        first.extract("same input").await.unwrap();
        assert_eq!(first.stats().cache_hits, 0);

        let changed: Arc<dyn EntityExtractor> = Arc::new(
            FixtureEntityExtractor::default().with_cache_identity("strict=false;max_entities=5"),
        );
        let changed =
            ResilientEntityExtractor::new(changed, Some(repo), ProcessingConfig::default());
        changed.extract("same input").await.unwrap();
        assert_eq!(
            changed.stats().cache_hits,
            0,
            "setting change must miss cache"
        );
    }

    #[tokio::test]
    async fn retries_stop_at_attempt_ceiling() {
        let inner: Arc<dyn Embedder> =
            Arc::new(DeterministicEmbedder::default().fail_next_requests(10, "timeout"));
        let adapter = ResilientEmbedder::new(
            inner,
            None,
            ProcessingConfig {
                retry_attempts: 3,
                initial_backoff: Duration::ZERO,
                ..Default::default()
            },
        );
        assert!(adapter.embed("x", false).await.is_err());
        let stats = adapter.stats();
        assert_eq!(stats.requests, 3);
        assert_eq!(stats.retries, 2);
        assert_eq!(stats.failures, 1);
    }

    #[derive(Clone)]
    struct ConcurrencyProbe {
        active: Arc<AtomicUsize>,
        peak: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Embedder for ConcurrencyProbe {
        async fn embed(&self, _text: &str, _is_query: bool) -> Result<Vec<f32>> {
            let now = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(now, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(10)).await;
            self.active.fetch_sub(1, Ordering::SeqCst);
            Ok(vec![0.0; 1024])
        }
        async fn embed_batch(&self, texts: &[String], query: bool) -> Result<Vec<Vec<f32>>> {
            let mut values = Vec::with_capacity(texts.len());
            for text in texts {
                values.push(self.embed(text, query).await?);
            }
            Ok(values)
        }
        async fn health(&self) -> Result<bool> {
            Ok(true)
        }
        fn capabilities(&self) -> InferenceCapabilities {
            InferenceCapabilities {
                provider: "probe".into(),
                model: "probe".into(),
                endpoint: "offline://probe".into(),
                known_dimension: Some(1024),
                cache_identity: "probe-v1".into(),
            }
        }
    }

    #[derive(Clone)]
    struct OllamaBatchProbe {
        provider: String,
        max_batch_size: Option<usize>,
        fail_on_batch: Option<usize>,
        individual_requests: Arc<AtomicUsize>,
        batch_requests: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Embedder for OllamaBatchProbe {
        async fn embed(&self, _text: &str, _is_query: bool) -> Result<Vec<f32>> {
            self.individual_requests.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(20)).await;
            Ok(vec![0.0; 1024])
        }

        async fn embed_batch(&self, texts: &[String], query: bool) -> Result<Vec<Vec<f32>>> {
            let batch_number = self.batch_requests.fetch_add(1, Ordering::SeqCst) + 1;
            if self.fail_on_batch == Some(batch_number) {
                return Err(AgentError::InferenceService("timeout".into()));
            }
            if self.provider.eq_ignore_ascii_case("tei") {
                tokio::time::sleep(Duration::from_millis(20)).await;
                return Ok(vec![vec![0.0; 1024]; texts.len()]);
            }
            let mut values = Vec::with_capacity(texts.len());
            for text in texts {
                values.push(self.embed(text, query).await?);
            }
            Ok(values)
        }

        async fn health(&self) -> Result<bool> {
            Ok(true)
        }

        fn capabilities(&self) -> InferenceCapabilities {
            InferenceCapabilities {
                provider: self.provider.clone(),
                model: "probe".into(),
                endpoint: "offline://ollama-probe".into(),
                known_dimension: Some(1024),
                cache_identity: "ollama-probe-v1".into(),
            }
        }

        fn max_batch_size(&self) -> Option<usize> {
            self.max_batch_size
        }
    }

    #[tokio::test]
    async fn concurrency_never_exceeds_configured_bound() {
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let inner: Arc<dyn Embedder> = Arc::new(ConcurrencyProbe {
            active,
            peak: peak.clone(),
        });
        let adapter = ResilientEmbedder::new(
            inner,
            None,
            ProcessingConfig {
                concurrency: 2,
                ..Default::default()
            },
        );
        let (a, b, c, d, e) = tokio::join!(
            adapter.embed("a", false),
            adapter.embed("b", false),
            adapter.embed("c", false),
            adapter.embed("d", false),
            adapter.embed("e", false),
        );
        for result in [a, b, c, d, e] {
            result.unwrap();
        }
        assert_eq!(peak.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn ollama_batch_times_out_each_provider_request_not_the_full_batch() {
        let individual_requests = Arc::new(AtomicUsize::new(0));
        let batch_requests = Arc::new(AtomicUsize::new(0));
        let inner: Arc<dyn Embedder> = Arc::new(OllamaBatchProbe {
            provider: "ollama".into(),
            max_batch_size: None,
            fail_on_batch: None,
            individual_requests: individual_requests.clone(),
            batch_requests: batch_requests.clone(),
        });
        let adapter = ResilientEmbedder::new(
            inner,
            None,
            ProcessingConfig {
                request_timeout: Duration::from_millis(35),
                retry_attempts: 1,
                ..Default::default()
            },
        );
        let texts = vec!["first".to_string(), "second".to_string()];

        let embeddings = adapter.embed_batch(&texts, false).await.unwrap();

        assert_eq!(embeddings.len(), 2);
        assert_eq!(individual_requests.load(Ordering::SeqCst), 2);
        assert_eq!(batch_requests.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn tei_batch_times_out_each_provider_request_not_the_full_batch() {
        let individual_requests = Arc::new(AtomicUsize::new(0));
        let batch_requests = Arc::new(AtomicUsize::new(0));
        let inner: Arc<dyn Embedder> = Arc::new(OllamaBatchProbe {
            provider: "tei".into(),
            max_batch_size: Some(2),
            fail_on_batch: None,
            individual_requests: individual_requests.clone(),
            batch_requests: batch_requests.clone(),
        });
        let adapter = ResilientEmbedder::new(
            inner,
            None,
            ProcessingConfig {
                request_timeout: Duration::from_millis(35),
                retry_attempts: 1,
                ..Default::default()
            },
        );
        let embeddings = adapter
            .embed_batch(
                &[
                    "first".to_string(),
                    "second".to_string(),
                    "third".to_string(),
                    "fourth".to_string(),
                    "fifth".to_string(),
                ],
                false,
            )
            .await
            .unwrap();
        assert_eq!(embeddings.len(), 5);
        assert_eq!(individual_requests.load(Ordering::SeqCst), 0);
        assert_eq!(batch_requests.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn successful_tei_chunks_remain_cached_after_a_later_chunk_fails() {
        let repo = Repository::new(init_memory().await.unwrap());
        let first_batches = Arc::new(AtomicUsize::new(0));
        let failing: Arc<dyn Embedder> = Arc::new(OllamaBatchProbe {
            provider: "tei".into(),
            max_batch_size: Some(2),
            fail_on_batch: Some(2),
            individual_requests: Arc::new(AtomicUsize::new(0)),
            batch_requests: first_batches.clone(),
        });
        let config = ProcessingConfig {
            retry_attempts: 1,
            ..Default::default()
        };
        let texts = vec!["one".into(), "two".into(), "three".into()];
        assert!(
            ResilientEmbedder::new(failing, Some(repo.clone()), config.clone())
                .embed_batch(&texts, false)
                .await
                .is_err()
        );
        assert_eq!(first_batches.load(Ordering::SeqCst), 2);

        let restart_batches = Arc::new(AtomicUsize::new(0));
        let healthy: Arc<dyn Embedder> = Arc::new(OllamaBatchProbe {
            provider: "tei".into(),
            max_batch_size: Some(2),
            fail_on_batch: None,
            individual_requests: Arc::new(AtomicUsize::new(0)),
            batch_requests: restart_batches.clone(),
        });
        let embeddings = ResilientEmbedder::new(healthy, Some(repo), config)
            .embed_batch(&texts, false)
            .await
            .unwrap();
        assert_eq!(embeddings.len(), 3);
        assert_eq!(restart_batches.load(Ordering::SeqCst), 1);
    }
}
