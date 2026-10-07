//! Default-off boundary evidence for the fictional policy-migration fixture.
//! These labels diagnose failures; they do not classify an operation as safe
//! to retry. Results, statements, transaction boundaries and errors are intact.

use crate::DbError;
use std::future::IntoFuture;
use surrealdb::types::{ErrorDetails, QueryError};

pub(super) trait ProbeError {
    fn probe_class(&self) -> &'static str;
}

impl ProbeError for surrealdb::Error {
    fn probe_class(&self) -> &'static str {
        match self.details() {
            ErrorDetails::Query(Some(QueryError::TransactionConflict)) => "transaction_conflict",
            ErrorDetails::Query(Some(QueryError::NotExecuted)) => "not_executed",
            _ => self.kind_str(),
        }
    }
}

impl ProbeError for DbError {
    fn probe_class(&self) -> &'static str {
        match self {
            Self::Surreal(error) => error.probe_class(),
            Self::RemoteJobOwnershipLost(_) => "ownership_lost",
            Self::RemoteJobCancelled(_) => "cancelled",
            Self::RemoteJobSourceConflict(_) => "source_conflict",
            Self::QueryFailed(_) => "query_failed",
            _ => "database_error",
        }
    }
}

fn emit(
    operation: &'static str,
    stage: &'static str,
    outcome: &'static str,
    error_class: &'static str,
) {
    if tracing::enabled!(target: "graphrag_migration_probe", tracing::Level::INFO)
        && std::env::var("GRAPHRAG_MIGRATION_CONFLICT_PROBE").as_deref() == Ok("1")
    {
        tracing::info!(target: "graphrag_migration_probe", operation, stage, outcome, error_class);
    }
}

pub(super) fn migration_probe_result<T, E: ProbeError>(
    operation: &'static str,
    stage: &'static str,
    result: std::result::Result<T, E>,
) -> std::result::Result<T, E> {
    match &result {
        Ok(_) => emit(operation, stage, "passed", "none"),
        Err(error) => emit(operation, stage, "failed", error.probe_class()),
    }
    result
}

pub(super) async fn migration_probe<T, E: ProbeError>(
    operation: &'static str,
    stage: &'static str,
    future: impl IntoFuture<Output = std::result::Result<T, E>>,
) -> std::result::Result<T, E> {
    emit(operation, stage, "started", "none");
    migration_probe_result(operation, stage, future.await)
}

pub(super) fn migration_statement_errors(
    operation: &'static str,
    errors: &std::collections::HashMap<usize, surrealdb::Error>,
) {
    const STAGES: [&str; 32] = [
        "statement_0",
        "statement_1",
        "statement_2",
        "statement_3",
        "statement_4",
        "statement_5",
        "statement_6",
        "statement_7",
        "statement_8",
        "statement_9",
        "statement_10",
        "statement_11",
        "statement_12",
        "statement_13",
        "statement_14",
        "statement_15",
        "statement_16",
        "statement_17",
        "statement_18",
        "statement_19",
        "statement_20",
        "statement_21",
        "statement_22",
        "statement_23",
        "statement_24",
        "statement_25",
        "statement_26",
        "statement_27",
        "statement_28",
        "statement_29",
        "statement_30",
        "statement_31",
    ];
    // Sort diagnostic slots only; the original HashMap and error conversion
    // are retained verbatim by check_write.
    let mut slots = errors.keys().copied().collect::<Vec<_>>();
    slots.sort_unstable();
    for slot in slots {
        emit(
            operation,
            STAGES
                .get(slot)
                .copied()
                .unwrap_or("statement_over_probe_bound"),
            "failed",
            errors[&slot].probe_class(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn probe_preserves_typed_error_and_single_execution() {
        // A raw message which happens to mention a conflict is not promoted
        // into the structured transaction-conflict class.
        let internal =
            surrealdb::Error::internal("fictional Transaction conflict: private-value".into());
        assert_eq!(internal.probe_class(), "Internal");
        let conflict = surrealdb::Error::query(
            "fictional private-value".into(),
            QueryError::TransactionConflict,
        );
        assert_eq!(conflict.probe_class(), "transaction_conflict");
        let not_executed =
            surrealdb::Error::query("fictional private-value".into(), QueryError::NotExecuted);
        assert_eq!(not_executed.probe_class(), "not_executed");
        assert_eq!(
            DbError::QueryFailed("fictional Transaction conflict: private-value".into())
                .probe_class(),
            "query_failed"
        );
        let calls = AtomicUsize::new(0);
        let expected = conflict.clone();
        let result: std::result::Result<(), surrealdb::Error> =
            migration_probe("fixture", "passthrough", async {
                calls.fetch_add(1, Ordering::SeqCst);
                Err(conflict)
            })
            .await;
        assert_eq!(result, Err(expected));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let success: std::result::Result<u64, surrealdb::Error> =
            migration_probe("fixture", "passthrough", async {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(7)
            })
            .await;
        assert_eq!(success, Ok(7));
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }
}
