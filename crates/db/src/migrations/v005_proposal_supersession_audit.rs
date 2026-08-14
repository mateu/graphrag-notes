use super::Migration;

/// Keep the original review decision immutable when a proposal is later
/// superseded because its accepted edge was undone or a source endpoint was
/// removed.
pub(super) const MIGRATION: Migration = Migration {
    version: 5,
    name: "proposal_supersession_audit",
    sql: r#"
DEFINE FIELD IF NOT EXISTS superseded_at ON proposed_edge TYPE option<datetime>;
DEFINE FIELD IF NOT EXISTS supersession_reason ON proposed_edge TYPE option<string>;
DEFINE FIELD IF NOT EXISTS acceptance_is_manual ON proposed_edge TYPE option<bool>;
"#,
};
