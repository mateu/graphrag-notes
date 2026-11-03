use super::Migration;

/// Typed relation evidence (#205): an optional, versioned assessment object on
/// proposals and on the four accepted note-edge tables. Purely additive. No
/// row is rewritten, legacy rows keep NONE and display as unassessed.
///
/// The Rust validators decide what may be proposed and accepted. These
/// ASSERTs are a last line of defence for out-of-contract writes and tampered
/// portable archives: the accepted graph never holds an analogy/hypothesis,
/// a same-origin `supports`, a `related_to` without explicit link evidence,
/// another assessment version, or a calibration claim.
pub(super) const MIGRATION: Migration = Migration {
    version: 28,
    name: "relation_evidence_assessment",
    sql: r#"
DEFINE FIELD IF NOT EXISTS evidence_assessment ON proposed_edge TYPE option<object> FLEXIBLE
    ASSERT $value = NONE OR ($value.version = 1 AND $value.calibration = 'none'
        AND $value.evidence_mode IN ['explicit_statement', 'derivation', 'analogy', 'hypothesis']);
DEFINE FIELD IF NOT EXISTS evidence_assessment ON supports TYPE option<object> FLEXIBLE
    ASSERT $value = NONE OR ($value.version = 1 AND $value.calibration = 'none'
        AND $value.relation = 'supports'
        AND $value.evidence_mode IN ['explicit_statement', 'derivation']
        AND $value.lineage.same_origin = false);
DEFINE FIELD IF NOT EXISTS evidence_assessment ON contradicts TYPE option<object> FLEXIBLE
    ASSERT $value = NONE OR ($value.version = 1 AND $value.calibration = 'none'
        AND $value.relation = 'contradicts'
        AND $value.evidence_mode IN ['explicit_statement', 'derivation']);
DEFINE FIELD IF NOT EXISTS evidence_assessment ON derived_from TYPE option<object> FLEXIBLE
    ASSERT $value = NONE OR ($value.version = 1 AND $value.calibration = 'none'
        AND $value.relation = 'derived_from'
        AND $value.evidence_mode IN ['explicit_statement', 'derivation']);
DEFINE FIELD IF NOT EXISTS evidence_assessment ON related_to TYPE option<object> FLEXIBLE
    ASSERT $value = NONE OR ($value.version = 1 AND $value.calibration = 'none'
        AND $value.relation = 'related_to'
        AND $value.evidence_mode = 'explicit_statement'
        AND $value.link IS NOT NONE);
"#,
};
