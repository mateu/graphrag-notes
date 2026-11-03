//! Origin groups: copies, summaries and parts of one origin are not
//! independent corroboration. Grouping is deterministic and conservative.

use super::gather::EvidenceChunk;
use super::{Citation, EvidenceStrength, LineageReason, OriginGroup};
use graphrag_core::record_id_to_string;
use graphrag_db::repository::NoteEdgeRow;
use std::collections::{BTreeSet, HashMap};

pub(crate) struct Lineage {
    /// Evidence chunk index → index into `groups`.
    origin_of: Vec<usize>,
    pub groups: Vec<OriginGroup>,
}

impl Lineage {
    pub fn origin_label(&self, chunk: usize) -> &str {
        &self.groups[self.origin_of[chunk]].origin
    }
}

struct UnionFind(Vec<usize>);

impl UnionFind {
    fn find(&mut self, item: usize) -> usize {
        let parent = self.0[item];
        if parent == item {
            return item;
        }
        let root = self.find(parent);
        self.0[item] = root;
        root
    }

    fn union(&mut self, a: usize, b: usize) {
        let (a, b) = (self.find(a), self.find(b));
        // Keep the earliest chunk as root so labels follow evidence order.
        if a < b {
            self.0[b] = a;
        } else if b < a {
            self.0[a] = b;
        }
    }
}

fn normalized(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Group evidence chunks by shared origin. Chunks join one group when an
/// accepted `derived_from` edge links them, they belong to the same source
/// document or conversation, or their content is identical.
pub(crate) fn origin_groups(chunks: &[EvidenceChunk], relations: &[NoteEdgeRow]) -> Lineage {
    let index: HashMap<&str, usize> = chunks
        .iter()
        .enumerate()
        .map(|(position, chunk)| (chunk.id.as_str(), position))
        .collect();
    let mut sets = UnionFind((0..chunks.len()).collect());
    let mut links: Vec<(usize, usize, LineageReason)> = Vec::new();
    for edge in relations
        .iter()
        .filter(|edge| edge.edge_type == "derived_from")
    {
        let from = index
            .get(record_id_to_string(&edge.in_id).as_str())
            .copied();
        let to = index
            .get(record_id_to_string(&edge.out_id).as_str())
            .copied();
        if let (Some(from), Some(to)) = (from, to) {
            links.push((from, to, LineageReason::DerivedFrom));
        }
    }
    type LineageKey = fn(&EvidenceChunk) -> Option<String>;
    let keyed: [(LineageReason, LineageKey); 3] = [
        (LineageReason::SameSource, |chunk| chunk.source_id.clone()),
        (LineageReason::SameConversation, |chunk| {
            chunk.conversation_id.clone()
        }),
        (LineageReason::IdenticalContent, |chunk| {
            Some(normalized(&chunk.content)).filter(|text| !text.is_empty())
        }),
    ];
    for (reason, key) in keyed {
        let mut first: HashMap<String, usize> = HashMap::new();
        for (position, chunk) in chunks.iter().enumerate() {
            if let Some(key) = key(chunk) {
                match first.get(&key) {
                    Some(earlier) => links.push((*earlier, position, reason)),
                    None => {
                        first.insert(key, position);
                    }
                }
            }
        }
    }
    for (a, b, _) in &links {
        sets.union(*a, *b);
    }
    let mut group_of_root: HashMap<usize, usize> = HashMap::new();
    let mut groups: Vec<OriginGroup> = Vec::new();
    let mut origin_of = Vec::with_capacity(chunks.len());
    let mut reasons: Vec<BTreeSet<LineageReason>> = Vec::new();
    for (position, chunk) in chunks.iter().enumerate() {
        let root = sets.find(position);
        let group = *group_of_root.entry(root).or_insert_with(|| {
            groups.push(OriginGroup {
                origin: format!("origin-{}", groups.len() + 1),
                chunks: Vec::new(),
                reasons: Vec::new(),
            });
            reasons.push(BTreeSet::new());
            groups.len() - 1
        });
        groups[group].chunks.push(chunk.id.clone());
        origin_of.push(group);
    }
    for (a, _, reason) in links {
        reasons[origin_of[a]].insert(reason);
    }
    for (group, reasons) in groups.iter_mut().zip(reasons) {
        group.reasons = reasons.into_iter().collect();
    }
    Lineage { origin_of, groups }
}

/// Evidence-strength wording from the number of distinct origins among a
/// claim's verified citations. Deliberately no probabilities.
pub(crate) fn strength(citations: &[Citation], contested: bool) -> EvidenceStrength {
    let origins: BTreeSet<&str> = citations
        .iter()
        .map(|citation| citation.origin.as_str())
        .collect();
    let chunks: BTreeSet<&str> = citations
        .iter()
        .map(|citation| citation.chunk_id.as_str())
        .collect();
    let mut wording = match (origins.len(), chunks.len()) {
        (0, _) => "No verified citation.".to_string(),
        (1, 1) => "Stated by one cited chunk; not independently corroborated.".to_string(),
        (1, cited) => format!(
            "Stated by {cited} chunks that share one origin (copy, summary or same source); not independently corroborated."
        ),
        (origins, _) => format!("Stated by {origins} lineage-independent origins."),
    };
    if contested {
        wording.push_str(" Contested: an accepted contradicts edge or a conflicting claim touches this evidence.");
    }
    EvidenceStrength {
        independent_origins: origins.len(),
        cited_chunks: chunks.len(),
        contested,
        wording,
    }
}
