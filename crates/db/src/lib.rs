//! Database layer for GraphRAG Notes
//!
//! Provides SurrealDB integration with schema management and queries.

pub mod compatibility;
pub mod error;
pub mod fusion;
pub mod migrations;
pub mod repository;
pub mod schema;

pub use error::{DbError, Result};
pub use repository::{
    parse_portable_record_id, parse_record_id, uploaded_processing_compatible,
    uploaded_processing_policy_sha256, uploaded_source_revision, InferenceCacheEntry,
    InspectedConversation, InspectedMessage, InspectionProvenance, MutationGuard,
    MutationNoteSnapshot, ProcessingJob, ProcessingJobStatus, ProcessingJobType,
    ProcessingJobUpdate, RecordInspection, RemoteCaptureInput, RemoteCaptureReceipt,
    RemoteEndpointProposalInput, RemoteEnrichmentInput, RemoteJobAdmission, RemoteJobLease,
    RemoteMutationEffect, RemoteMutationInput, RemoteUploadInput, RemoteUploadJob,
    RemoteUploadJobStatus, Repository, SourceDeleteSummary, SourceImportAction, SourceImportPlan,
    MAX_INSPECTION_NEIGHBORS, MAX_REMOTE_UPLOAD_BYTES, MAX_REMOTE_UPLOAD_CHUNKS, PORTABLE_TABLES,
};
pub use repository::{
    source_entity_enrichment_metadata_eligible, SourceEnrichmentPlan, SourceEnrichmentStatus,
};

use std::collections::HashMap;
use std::future::{Future, IntoFuture};
use std::ops::Deref;
#[cfg(feature = "rocksdb")]
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
#[cfg(feature = "rocksdb")]
use surrealdb::engine::local::RocksDb;
use surrealdb::engine::local::{Db, Mem};
use surrealdb::method::IntoVariables;
use surrealdb::opt::{QueryResult, QueryStream};
use surrealdb::types::SurrealValue;
use surrealdb::Surreal;
use surrealdb_types::Variables;
use tokio::sync::Mutex;
use uuid::Uuid;

/// A database client together with process-local coordination scoped to that
/// client identity. Cloning a connection preserves the same coordination;
/// separately opened stores receive independent locks.
#[derive(Clone, Debug)]
pub struct DbConnection {
    client: Surreal<Db>,
    proposal_lifecycle_lock: Arc<Mutex<()>>,
    remote_job_transition_lock: Arc<Mutex<()>>,
    source_enrichment_worker_lock: Arc<Mutex<()>>,
}

const TRANSACTION_CONFLICT_RETRY_ATTEMPTS: usize = 8;

fn is_retryable_transaction_conflict(error: &surrealdb::Error) -> bool {
    matches!(
        error.details(),
        surrealdb_types::ErrorDetails::Query(Some(
            surrealdb_types::QueryError::TransactionConflict
                | surrealdb_types::QueryError::NotExecuted
        ))
    )
}

/// A statement-level response is replayable only when every failed statement
/// reports an engine-aborted transaction. Application `THROW` sentinels are
/// not typed engine conflicts, so mixed responses are never retried.
pub(crate) fn statement_errors_are_retryable_conflict(
    errors: &HashMap<usize, surrealdb::Error>,
) -> bool {
    !errors.is_empty() && errors.values().all(is_retryable_transaction_conflict)
}

/// A query response whose statement errors may have been inspected by the
/// retry boundary. SurrealDB removes errors from `IndexedResults` when read,
/// so retain them for the caller-facing response contract.
#[derive(Debug)]
pub struct DbResponse {
    inner: surrealdb::IndexedResults,
    extracted_errors: HashMap<usize, surrealdb::Error>,
    statement_count: usize,
}

impl DbResponse {
    fn new(mut inner: surrealdb::IndexedResults) -> Self {
        let statement_count = inner.num_statements();
        let extracted_errors = inner.take_errors();
        Self {
            inner,
            extracted_errors,
            statement_count,
        }
    }

    fn uninspected(inner: surrealdb::IndexedResults) -> Self {
        let statement_count = inner.num_statements();
        Self {
            inner,
            extracted_errors: HashMap::new(),
            statement_count,
        }
    }

    pub fn take<R>(&mut self, index: usize) -> surrealdb::Result<R>
    where
        R: SurrealValue,
        usize: QueryResult<R>,
    {
        if let Some(error) = self.extracted_errors.remove(&index) {
            return Err(error);
        }
        self.inner.take(index)
    }

    pub fn stream<R>(
        &mut self,
        index: impl QueryStream<R>,
    ) -> surrealdb::Result<surrealdb::method::QueryStream<R>> {
        self.inner.stream(index)
    }

    pub fn take_errors(&mut self) -> HashMap<usize, surrealdb::Error> {
        let mut errors = std::mem::take(&mut self.extracted_errors);
        errors.extend(self.inner.take_errors());
        errors
    }

    pub fn check(mut self) -> surrealdb::Result<Self> {
        if let Some(index) = self.extracted_errors.keys().min().copied() {
            return Err(self
                .extracted_errors
                .remove(&index)
                .expect("selected extracted query error"));
        }
        self.inner = self.inner.check()?;
        Ok(self)
    }

    pub fn num_statements(&self) -> usize {
        self.statement_count
    }

    fn into_parts(
        self,
    ) -> (
        surrealdb::IndexedResults,
        HashMap<usize, surrealdb::Error>,
        usize,
    ) {
        (self.inner, self.extracted_errors, self.statement_count)
    }
}

/// Return true only for queries whose replay cannot duplicate a mutation.
///
/// This deliberately recognizes a small, unambiguous subset of SurrealQL. An
/// unterminated string/comment means no retry.
fn is_replayable_query(query: &str) -> bool {
    let statements = match query_statements(query) {
        Some(statements) => statements,
        None => return false,
    };

    match statements.as_slice() {
        [statement] => {
            !(statement.starts_with("THROW")
                && query.to_ascii_uppercase().contains("NOTEREVISIONCONFLICT"))
        }
        [first, .., last] => first == "BEGIN TRANSACTION" && last == "COMMIT TRANSACTION",
        [] => false,
    }
}

/// Extract uppercase statement text without treating string/comment semicolons
/// as statement boundaries.
fn query_statements(query: &str) -> Option<Vec<String>> {
    #[derive(Clone, Copy)]
    enum State {
        Code,
        SingleQuote,
        DoubleQuote,
        Backtick,
        LineComment,
        BlockComment,
    }

    let mut state = State::Code;
    let mut statements = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = query.chars().collect();
    let mut index = 0;
    while index < chars.len() {
        let character = chars[index];
        let next = chars.get(index + 1).copied();
        match state {
            State::Code => match (character, next) {
                ('-', Some('-')) => {
                    state = State::LineComment;
                    index += 1;
                }
                ('/', Some('*')) => {
                    state = State::BlockComment;
                    index += 1;
                }
                ('\'', _) => state = State::SingleQuote,
                ('"', _) => state = State::DoubleQuote,
                ('`', _) => state = State::Backtick,
                (';', _) => {
                    let statement = current.trim();
                    if !statement.is_empty() {
                        statements.push(statement.to_ascii_uppercase());
                    }
                    current.clear();
                }
                _ => current.push(character),
            },
            State::SingleQuote | State::DoubleQuote | State::Backtick => {
                let quote = match state {
                    State::SingleQuote => '\'',
                    State::DoubleQuote => '"',
                    State::Backtick => '`',
                    _ => unreachable!(),
                };
                if character == '\\' {
                    index += 1;
                    if index == chars.len() {
                        return None;
                    }
                } else if character == quote {
                    state = State::Code;
                }
            }
            State::LineComment => {
                if character == '\n' {
                    state = State::Code;
                    current.push(' ');
                }
            }
            State::BlockComment => {
                if character == '*' && next == Some('/') {
                    state = State::Code;
                    current.push(' ');
                    index += 1;
                }
            }
        }
        index += 1;
    }
    if !matches!(state, State::Code | State::LineComment) {
        return None;
    }
    let statement = current.trim();
    if !statement.is_empty() {
        statements.push(statement.to_ascii_uppercase());
    }
    Some(statements)
}

/// A replayable query builder for complete SurrealQL transactions and single statements.
///
/// The underlying query is rerun only after SurrealDB reports that the entire
/// transaction rolled back. A single statement is also an implicit transaction.
/// Other multi-statement scripts execute once because an earlier mutation may
/// already have committed.
#[must_use = "queries do nothing unless you `.await` or poll them"]
pub struct DbQuery {
    client: Surreal<Db>,
    query: String,
    variables: std::result::Result<Variables, String>,
    retry_transaction_conflicts: bool,
}

impl DbQuery {
    fn new(client: Surreal<Db>, query: String) -> Self {
        Self {
            client,
            retry_transaction_conflicts: is_replayable_query(&query),
            query,
            variables: Ok(Variables::new()),
        }
    }

    /// Chain another statement, retaining the ordinary SurrealDB query API.
    pub fn query(mut self, query: impl Into<String>) -> Self {
        self.query.push(';');
        self.query.push_str(&query.into());
        self.retry_transaction_conflicts = is_replayable_query(&self.query);
        self
    }

    /// Execute with the normal per-statement statistics response.
    pub fn with_stats(self) -> DbQueryWithStats {
        DbQueryWithStats(self)
    }

    /// Execute as a result stream. Streaming intentionally does not retry:
    /// callers can observe provisional rows before the transaction is known to
    /// commit, so replay would violate the stream contract.
    pub fn stream_items(self) -> surrealdb::Result<surrealdb::method::ItemStream> {
        let variables = self.variables.map_err(surrealdb::Error::internal)?;
        self.client.query(self.query).bind(variables).stream_items()
    }

    pub fn bind(mut self, vars: impl IntoVariables) -> Self {
        self.variables = match (self.variables, vars.into_variables()) {
            (Ok(mut current), Ok(next)) => {
                current.extend(next);
                Ok(current)
            }
            (Ok(_), Err(error)) | (Err(_), Err(error)) => Err(error.to_string()),
            (Err(error), Ok(_)) => Err(error),
        };
        self
    }
}

async fn retry_transaction_conflicts<T, F, Fut>(mut operation: F) -> surrealdb::Result<T>
where
    F: FnMut() -> Fut,
    Fut: IntoFuture<Output = surrealdb::Result<T>>,
{
    for attempt in 1..=TRANSACTION_CONFLICT_RETRY_ATTEMPTS {
        match operation().into_future().await {
            Err(error)
                if is_retryable_transaction_conflict(&error)
                    && attempt < TRANSACTION_CONFLICT_RETRY_ATTEMPTS =>
            {
                // SurrealDB starts an index-compaction pass after a 250 ms
                // commit-trigger debounce. Quadratic spacing lets that pass
                // finish instead of repeatedly colliding with it.
                tokio::time::sleep(Duration::from_millis((attempt * attempt * 25) as u64)).await;
            }
            result => return result,
        }
    }
    unreachable!("the final transaction retry attempt always returns")
}

/// Query response wrapper retaining SurrealDB's per-statement execution stats.
#[must_use = "queries do nothing unless you `.await` or poll them"]
pub struct DbQueryWithStats(DbQuery);

#[must_use = "queries do nothing unless you `.await` or poll them"]
pub struct DbResponseWithStats {
    inner: surrealdb::method::WithStats<surrealdb::IndexedResults>,
    extracted_errors: HashMap<usize, surrealdb::Error>,
    statement_count: usize,
}

impl DbResponseWithStats {
    pub fn take<R>(
        &mut self,
        index: usize,
    ) -> Option<(surrealdb::method::Stats, surrealdb::Result<R>)>
    where
        R: SurrealValue,
        usize: QueryResult<R>,
    {
        if let Some(error) = self.extracted_errors.remove(&index) {
            return Some((surrealdb::method::Stats::default(), Err(error)));
        }
        self.inner.take(index)
    }

    pub fn take_errors(&mut self) -> HashMap<usize, surrealdb::Error> {
        let mut errors = std::mem::take(&mut self.extracted_errors);
        errors.extend(
            self.inner
                .take_errors()
                .into_iter()
                .map(|(index, (_, error))| (index, error)),
        );
        errors
    }

    pub fn check(mut self) -> surrealdb::Result<Self> {
        if let Some(index) = self.extracted_errors.keys().min().copied() {
            return Err(self
                .extracted_errors
                .remove(&index)
                .expect("selected extracted query error"));
        }
        self.inner = self.inner.check()?;
        Ok(self)
    }

    pub fn num_statements(&self) -> usize {
        self.statement_count
    }
}

impl IntoFuture for DbQueryWithStats {
    type Output = surrealdb::Result<DbResponseWithStats>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let response = self.0.await?;
            let (inner, extracted_errors, statement_count) = response.into_parts();
            Ok(DbResponseWithStats {
                inner: surrealdb::method::WithStats(inner),
                extracted_errors,
                statement_count,
            })
        })
    }
}

impl IntoFuture for DbQuery {
    type Output = surrealdb::Result<DbResponse>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let variables = self.variables.map_err(surrealdb::Error::internal)?;
            let execute = || {
                let query = self
                    .client
                    .query(self.query.clone())
                    .bind(variables.clone());
                async move { query.await }
            };
            if !self.retry_transaction_conflicts {
                return execute().await.map(DbResponse::uninspected);
            }
            retry_transaction_conflicts(|| async {
                let response = DbResponse::new(execute().await?);
                if statement_errors_are_retryable_conflict(&response.extracted_errors) {
                    return Err(surrealdb::Error::query(
                        "retryable statement-level transaction conflict".into(),
                        surrealdb::types::QueryError::TransactionConflict,
                    ));
                }
                Ok(response)
            })
            .await
        })
    }
}

impl DbConnection {
    /// Wrap a configured SurrealDB client in the application connection type.
    pub fn new(client: Surreal<Db>) -> Self {
        Self {
            client,
            proposal_lifecycle_lock: Arc::new(Mutex::new(())),
            remote_job_transition_lock: Arc::new(Mutex::new(())),
            source_enrichment_worker_lock: Arc::new(Mutex::new(())),
        }
    }

    /// Run a query, automatically retrying complete transactions and one
    /// statement when SurrealDB confirms they were aborted by a write conflict.
    pub fn query(&self, query: impl Into<String>) -> DbQuery {
        DbQuery::new(self.client.clone(), query.into())
    }

    pub(crate) fn proposal_lifecycle_lock(&self) -> Arc<Mutex<()>> {
        Arc::clone(&self.proposal_lifecycle_lock)
    }

    pub(crate) fn source_enrichment_worker_lock(&self) -> Arc<Mutex<()>> {
        Arc::clone(&self.source_enrichment_worker_lock)
    }

    pub(crate) fn remote_job_transition_lock(&self) -> Arc<Mutex<()>> {
        Arc::clone(&self.remote_job_transition_lock)
    }
}

impl Deref for DbConnection {
    type Target = Surreal<Db>;

    fn deref(&self) -> &Self::Target {
        &self.client
    }
}

/// Initialize database with RocksDB (persistent)
#[cfg(feature = "rocksdb")]
pub async fn init_persistent(path: impl AsRef<Path>) -> Result<DbConnection> {
    let db = connect_persistent(path).await?;
    setup_database(&db).await?;
    Ok(db)
}

/// Open an existing persistent database without applying application
/// migrations.  Operational diagnostics use this so inspection cannot mutate
/// a user's schema or data.  Opening an absent path is intentionally left to
/// callers: `graphrag doctor` reports that state without creating a store.
#[cfg(feature = "rocksdb")]
pub async fn connect_persistent(path: impl AsRef<Path>) -> Result<DbConnection> {
    let path = path.as_ref();
    let client = Surreal::new::<RocksDb>(path).await.map_err(|error| {
        let message = error.to_string().to_ascii_lowercase();
        if message.contains("lock")
            && (message.contains("resource temporarily unavailable")
                || message.contains("lock hold by")
                || message.contains("already held")
                || message.contains("held by another"))
        {
            DbError::DatabaseBusy(path.display().to_string())
        } else {
            DbError::Surreal(error)
        }
    })?;
    let db = DbConnection::new(client);
    db.use_ns("graphrag").use_db("notes").await?;
    Ok(db)
}

/// Initialize database in-memory (for testing)
pub async fn init_memory() -> Result<DbConnection> {
    let db = DbConnection::new(Surreal::new::<Mem>(()).await?);
    // SurrealDB's in-memory engine can be shared by concurrently running
    // clients. Give each test/ephemeral client its own namespace so a
    // transaction in one caller cannot observe or conflict with another
    // caller's fixtures. Persistent stores retain the stable application
    // namespace through `setup_database` below.
    let namespace = format!("graphrag_memory_{}", Uuid::new_v4().simple());
    setup_database_at(&db, &namespace, "notes").await?;
    Ok(db)
}

/// Setup database namespace, database, and schema
async fn setup_database(db: &DbConnection) -> Result<()> {
    setup_database_at(db, "graphrag", "notes").await
}

async fn setup_database_at(db: &DbConnection, namespace: &str, database: &str) -> Result<()> {
    db.use_ns(namespace).use_db(database).await?;
    schema::initialize_schema(db).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn conflict_then_success(
        attempts: &std::sync::atomic::AtomicUsize,
    ) -> surrealdb::Result<&'static str> {
        use std::sync::atomic::Ordering;
        if attempts.fetch_add(1, Ordering::SeqCst) == 0 {
            Err(surrealdb::Error::query(
                "transaction conflict".into(),
                surrealdb::types::QueryError::TransactionConflict,
            ))
        } else {
            Ok("committed")
        }
    }

    #[test]
    fn statement_level_conflicts_require_only_engine_conflicts() {
        let conflict = surrealdb::Error::query(
            "transaction conflict".into(),
            surrealdb::types::QueryError::TransactionConflict,
        );
        let sentinel = surrealdb::Error::thrown("graphrag-note-editor-revision-conflict".into());
        let mut only_conflict = HashMap::new();
        only_conflict.insert(0, conflict.clone());
        assert!(statement_errors_are_retryable_conflict(&only_conflict));

        let mut sentinel_only = HashMap::new();
        sentinel_only.insert(0, sentinel.clone());
        assert!(!statement_errors_are_retryable_conflict(&sentinel_only));

        let mut mixed = only_conflict;
        mixed.insert(1, sentinel);
        assert!(!statement_errors_are_retryable_conflict(&mixed));
    }

    #[tokio::test]
    async fn single_statement_retries_native_conflicts() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        assert!(is_replayable_query(
            "CREATE note SET title = 'one; statement';"
        ));
        let attempts = AtomicUsize::new(0);
        assert_eq!(
            retry_transaction_conflicts(|| conflict_then_success(&attempts))
                .await
                .unwrap(),
            "committed"
        );
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn complete_transaction_retries_native_conflicts() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        assert!(is_replayable_query(
            "BEGIN TRANSACTION; CREATE note; COMMIT TRANSACTION;"
        ));
        let attempts = AtomicUsize::new(0);
        assert_eq!(
            retry_transaction_conflicts(|| conflict_then_success(&attempts))
                .await
                .unwrap(),
            "committed"
        );
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn extracted_statement_errors_remain_visible_at_their_original_indices() {
        let db = init_memory().await.unwrap();
        let response = db
            .client
            .query("THROW 'first'; RETURN 'kept'; THROW 'third';")
            .await
            .unwrap();
        let mut response = DbResponse::new(response);
        assert_eq!(response.num_statements(), 3);
        assert!(response.take::<surrealdb::types::Value>(0).is_err());
        let _: surrealdb::types::Value = response.take(1).unwrap();
        let errors = response.take_errors();
        assert_eq!(errors.len(), 1);
        assert!(errors.contains_key(&2));
    }

    #[tokio::test]
    async fn extracted_statement_errors_check_uses_lowest_index() {
        let db = init_memory().await.unwrap();
        let response = db
            .client
            .query("THROW 'first'; THROW 'second';")
            .await
            .unwrap();
        let error = DbResponse::new(response).check().unwrap_err();
        assert!(error.to_string().contains("first"));
    }

    #[test]
    fn multi_statement_script_is_not_retried() {
        assert!(!is_replayable_query("CREATE note; CREATE source;"));
    }

    #[test]
    fn revision_sentinel_is_not_retried() {
        assert!(!is_replayable_query("THROW 'NoteRevisionConflict';"));
    }

    #[tokio::test]
    async fn non_conflict_error_is_not_retried() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let attempts = AtomicUsize::new(0);
        let error = retry_transaction_conflicts(|| {
            attempts.fetch_add(1, Ordering::SeqCst);
            async {
                Err::<(), _>(surrealdb::Error::query(
                    "timed out".into(),
                    surrealdb::types::QueryError::TimedOut {
                        duration: Duration::from_secs(1),
                    },
                ))
            }
        })
        .await
        .unwrap_err();
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
        assert!(!is_retryable_transaction_conflict(&error));
    }

    #[tokio::test]
    async fn test_init_memory() {
        let db = init_memory().await.expect("Failed to init memory db");
        // Just verify it connects
        let _: Vec<serde_json::Value> = db.select("note").await.unwrap();
    }

    #[tokio::test]
    async fn memory_clients_use_isolated_namespaces() {
        let first = Repository::new(init_memory().await.unwrap());
        let created = first
            .create_note(graphrag_core::Note::new("isolated memory fixture"))
            .await
            .unwrap();
        let second = Repository::new(init_memory().await.unwrap());
        assert!(second
            .get_note(&graphrag_core::record_id_to_string(
                created.id.as_ref().unwrap(),
            ))
            .await
            .unwrap()
            .is_none());
    }

    #[cfg(feature = "rocksdb")]
    #[tokio::test]
    async fn persistent_database_keeps_schema_version_after_reopen() {
        let temporary_directory = tempfile::tempdir().unwrap();

        let db = init_persistent(temporary_directory.path()).await.unwrap();
        assert_eq!(
            migrations::current_version(&db).await.unwrap(),
            migrations::LATEST_SCHEMA_VERSION
        );
        // The embedded engine owns the RocksDB lock through this client. Drop
        // it before attempting to reopen the same directory.
        drop(db);

        // SurrealDB releases the embedded-engine lock asynchronously after the
        // final client session is dropped. Its 3.3 index workers can outlive
        // the client for more than the old 50 ms grace period.
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        let reopened = init_persistent(temporary_directory.path()).await.unwrap();
        assert_eq!(
            migrations::current_version(&reopened).await.unwrap(),
            migrations::LATEST_SCHEMA_VERSION
        );
    }
}
