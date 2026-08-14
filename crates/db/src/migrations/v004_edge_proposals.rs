use super::Migration;

/// Persisted Gardener proposals and audit metadata for accepted note edges.
///
/// This migration is deliberately self-contained so it can be renumbered when
/// other Wave 1 migrations land first.
pub(super) const MIGRATION: Migration = Migration {
    version: 4,
    name: "edge_proposals",
    sql: r#"
DEFINE TABLE IF NOT EXISTS proposed_edge SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS dedupe_key ON proposed_edge TYPE string;
DEFINE FIELD IF NOT EXISTS in ON proposed_edge TYPE record<note>;
DEFINE FIELD IF NOT EXISTS out ON proposed_edge TYPE record<note>;
DEFINE FIELD IF NOT EXISTS edge_type ON proposed_edge TYPE string;
DEFINE FIELD IF NOT EXISTS confidence ON proposed_edge TYPE float;
DEFINE FIELD IF NOT EXISTS reason ON proposed_edge TYPE string;
DEFINE FIELD IF NOT EXISTS generator ON proposed_edge TYPE string;
DEFINE FIELD IF NOT EXISTS generator_version ON proposed_edge TYPE option<string>;
DEFINE FIELD IF NOT EXISTS model ON proposed_edge TYPE option<string>;
DEFINE FIELD IF NOT EXISTS status ON proposed_edge TYPE string DEFAULT 'pending';
DEFINE FIELD IF NOT EXISTS created_at ON proposed_edge TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS updated_at ON proposed_edge TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS reviewed_at ON proposed_edge TYPE option<datetime>;
DEFINE FIELD IF NOT EXISTS reviewer ON proposed_edge TYPE option<string>;
DEFINE FIELD IF NOT EXISTS action_reason ON proposed_edge TYPE option<string>;
DEFINE FIELD IF NOT EXISTS resulting_edge_id ON proposed_edge TYPE option<record>;
DEFINE INDEX IF NOT EXISTS idx_proposed_edge_dedupe ON proposed_edge FIELDS dedupe_key UNIQUE;
DEFINE INDEX IF NOT EXISTS idx_proposed_edge_status ON proposed_edge FIELDS status;

-- Audit fields are additive for existing edge rows. New writes use a
-- canonical dedupe key; related_to is canonicalized lexically by note id.
DEFINE FIELD IF NOT EXISTS reason ON supports TYPE option<string>;
DEFINE FIELD IF NOT EXISTS provenance ON supports TYPE option<string>;
DEFINE FIELD IF NOT EXISTS proposal_id ON supports TYPE option<record<proposed_edge>>;
DEFINE FIELD IF NOT EXISTS dedupe_key ON supports TYPE option<string>;
DEFINE INDEX IF NOT EXISTS idx_supports_dedupe ON supports FIELDS dedupe_key UNIQUE;

DEFINE FIELD IF NOT EXISTS reason ON contradicts TYPE option<string>;
DEFINE FIELD IF NOT EXISTS provenance ON contradicts TYPE option<string>;
DEFINE FIELD IF NOT EXISTS proposal_id ON contradicts TYPE option<record<proposed_edge>>;
DEFINE FIELD IF NOT EXISTS dedupe_key ON contradicts TYPE option<string>;
DEFINE INDEX IF NOT EXISTS idx_contradicts_dedupe ON contradicts FIELDS dedupe_key UNIQUE;

DEFINE FIELD IF NOT EXISTS reason ON derived_from TYPE option<string>;
DEFINE FIELD IF NOT EXISTS provenance ON derived_from TYPE option<string>;
DEFINE FIELD IF NOT EXISTS proposal_id ON derived_from TYPE option<record<proposed_edge>>;
DEFINE FIELD IF NOT EXISTS confidence ON derived_from TYPE option<float>;
DEFINE FIELD IF NOT EXISTS is_manual ON derived_from TYPE bool DEFAULT false;
DEFINE FIELD IF NOT EXISTS dedupe_key ON derived_from TYPE option<string>;
DEFINE INDEX IF NOT EXISTS idx_derived_from_dedupe ON derived_from FIELDS dedupe_key UNIQUE;

DEFINE FIELD IF NOT EXISTS provenance ON related_to TYPE option<string>;
DEFINE FIELD IF NOT EXISTS proposal_id ON related_to TYPE option<record<proposed_edge>>;
DEFINE FIELD IF NOT EXISTS dedupe_key ON related_to TYPE option<string>;
DEFINE INDEX IF NOT EXISTS idx_related_to_dedupe ON related_to FIELDS dedupe_key UNIQUE;
"#,
};
