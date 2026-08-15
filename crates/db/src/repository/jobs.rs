//! Durable processing-job and inference-cache ownership.
//!
//! Cache writes rely on the schema's unique semantic key. They remain a
//! single UPSERT so concurrent misses converge without changing the value.

use super::*;

impl Repository {
    /// Read a durable local inference result by its fully semantic cache key.
    pub async fn get_inference_cache(&self, cache_key: &str) -> Result<Option<serde_json::Value>> {
        #[derive(Deserialize, SurrealValue)]
        struct CacheValue {
            cache_value: serde_json::Value,
        }
        let row: Option<CacheValue> = self
            .db
            .query("SELECT cache_value FROM inference_cache WHERE cache_key = $cache_key LIMIT 1")
            .bind(("cache_key", cache_key.to_string()))
            .await?
            .take(0)?;
        Ok(row.map(|row| row.cache_value))
    }

    /// Store a JSON result under its semantic key. The unique index makes
    /// concurrent misses converge without changing the cached result.
    pub async fn put_inference_cache(&self, entry: InferenceCacheEntry) -> Result<()> {
        self.db
            .query(
                "UPSERT type::record('inference_cache', $cache_key) SET cache_key = $cache_key, operation = $operation, \
                 provider = $provider, model = $model, version = $version, input_hash = $input_hash, \
                 cache_value = $cache_value, updated_at = time::now(), created_at = IF created_at = NONE THEN time::now() ELSE created_at END",
            )
            .bind(("cache_key", entry.cache_key))
            .bind(("operation", entry.operation))
            .bind(("provider", entry.provider))
            .bind(("model", entry.model))
            .bind(("version", entry.version))
            .bind(("input_hash", entry.input_hash))
            .bind(("cache_value", entry.value))
            .await?
            .check()?;
        Ok(())
    }
}
