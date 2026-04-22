//! Integration tests for GraphRAG Notes
//!
//! Note: Tests that require local inference backends are marked with #[ignore]
//! Run them with: cargo test -- --ignored

mod common;

use graphrag_core::{EdgeType, Note, NoteType};
use graphrag_db::{init_memory, Repository};

/// Test database initialization and basic CRUD
#[tokio::test]
async fn test_database_crud() {
    let db = init_memory().await.expect("Failed to init db");
    let repo = Repository::new(db);

    // Create a note
    let note = Note::new("Test content for CRUD operations")
        .with_title("Test Note")
        .with_type(NoteType::Claim)
        .with_tags(vec!["test".into()]);

    let created = repo.create_note(note).await.expect("Failed to create note");
    assert!(created.id.is_some());
    assert_eq!(created.content, "Test content for CRUD operations");

    // List notes
    let notes = repo.list_notes(10).await.expect("Failed to list notes");
    assert_eq!(notes.len(), 1);

    // Get stats
    let stats = repo.get_stats().await.expect("Failed to get stats");
    assert_eq!(stats.note_count, 1);
}

/// Test creating notes with embeddings
#[tokio::test]
async fn test_note_with_embedding() {
    let db = init_memory().await.expect("Failed to init db");
    let repo = Repository::new(db);

    // Create a fake embedding (1024 dimensions)
    let embedding: Vec<f32> = (0..1024).map(|i| (i as f32) / 1024.0).collect();

    let note = Note::new("Note with embedding").with_embedding(embedding.clone());

    let created = repo.create_note(note).await.expect("Failed to create note");
    assert_eq!(created.embedding.len(), 1024);
}

/// Test graph edge creation
#[tokio::test]
async fn test_graph_edges() {
    let db = init_memory().await.expect("Failed to init db");
    let repo = Repository::new(db);

    // Create two notes
    let note1 = Note::new("First note about machine learning");
    let note1 = repo.create_note(note1).await.unwrap();
    let id1 = note1.id.unwrap();

    let note2 = Note::new("Second note that supports the first");
    let note2 = repo.create_note(note2).await.unwrap();
    let id2 = note2.id.unwrap();

    // Create an edge
    repo.create_edge(&id2, &id1, EdgeType::Supports, Some(0.9))
        .await
        .expect("Failed to create edge");

    // Verify stats
    let stats = repo.get_stats().await.unwrap();
    assert_eq!(stats.note_count, 2);
    assert_eq!(stats.edge_count, 1);
}

/// Test finding orphan notes
#[tokio::test]
async fn test_find_orphans() {
    let db = init_memory().await.expect("Failed to init db");
    let repo = Repository::new(db);

    // Create notes without connections
    for i in 0..3 {
        let note = Note::new(format!("Orphan note {}", i));
        repo.create_note(note).await.unwrap();
    }

    let orphans = repo.find_orphan_notes().await.unwrap();
    assert_eq!(orphans.len(), 3);
}

/// Test full-text search
#[tokio::test]
async fn test_fulltext_search() {
    let db = init_memory().await.expect("Failed to init db");
    let repo = Repository::new(db);

    // Create searchable notes
    let notes = vec![
        (
            "Machine Learning Basics",
            "Machine learning is a subset of artificial intelligence",
        ),
        (
            "Deep Learning",
            "Neural networks with multiple layers are called deep learning",
        ),
        (
            "Rust Programming",
            "Rust is a systems programming language focused on safety",
        ),
    ];

    for (title, content) in notes {
        let note = Note::new(content).with_title(title);
        repo.create_note(note).await.unwrap();
    }

    // Search for "machine learning"
    let results = repo.fulltext_search("machine learning", 10).await.unwrap();

    // Should find at least the first note
    assert!(!results.is_empty());

    // First result should be the most relevant
    let first = &results[0];
    assert!(first.content.to_lowercase().contains("machine learning"));
}

/// Test vector search (requires fake embeddings)
#[tokio::test]
async fn test_vector_search() {
    let db = init_memory().await.expect("Failed to init db");
    let repo = Repository::new(db);

    // Create notes with similar embeddings
    let base_embedding: Vec<f32> = (0..1024).map(|i| (i as f32) / 1024.0).collect();

    // Note 1: very similar to base
    let mut emb1 = base_embedding.clone();
    emb1[0] += 0.01;
    let note1 = Note::new("Similar note 1").with_embedding(emb1);
    repo.create_note(note1).await.unwrap();

    // Note 2: somewhat similar
    let mut emb2 = base_embedding.clone();
    emb2[0] += 0.1;
    emb2[1] += 0.1;
    let note2 = Note::new("Similar note 2").with_embedding(emb2);
    repo.create_note(note2).await.unwrap();

    // Note 3: very different
    let emb3: Vec<f32> = (0..1024).map(|i| 1.0 - (i as f32) / 1024.0).collect();
    let note3 = Note::new("Different note").with_embedding(emb3);
    repo.create_note(note3).await.unwrap();

    // Search with base embedding
    let results = repo.vector_search(base_embedding, 3).await.unwrap();

    // Should return results ordered by similarity
    assert_eq!(results.len(), 3);
}

/// Test note listing with limit
#[tokio::test]
async fn test_list_with_limit() {
    let db = init_memory().await.expect("Failed to init db");
    let repo = Repository::new(db);

    // Create 10 notes
    for i in 0..10 {
        let note = Note::new(format!("Note number {}", i));
        repo.create_note(note).await.unwrap();
    }

    // List with limit 5
    let notes = repo.list_notes(5).await.unwrap();
    assert_eq!(notes.len(), 5);

    // List all
    let all_notes = repo.list_notes(100).await.unwrap();
    assert_eq!(all_notes.len(), 10);
}

// ==========================================
// TESTS REQUIRING INFERENCE BACKENDS
// Run with: cargo test -- --ignored
// ==========================================

/// Test Librarian agent (requires inference backends)
#[tokio::test]
#[ignore = "Requires local inference backends (TEI/TGI or Ollama)"]
async fn test_librarian_ingest() {
    use graphrag_agents::{LibrarianAgent, TeiClient, TgiClient};

    let db = init_memory().await.expect("Failed to init db");
    let repo = Repository::new(db);
    let tei = TeiClient::default_local();
    let tgi = TgiClient::default_local();

    // Check inference backends are available
    if tei.health().await.is_err() || tgi.health().await.is_err() {
        eprintln!("Skipping test: inference backends not available");
        return;
    }

    let librarian = LibrarianAgent::new(repo.clone(), tei, tgi);

    let note = librarian
        .ingest_text(
            "Rust is a systems programming language that focuses on safety and performance",
            Some("Rust Overview".into()),
            vec!["rust".into(), "programming".into()],
        )
        .await
        .expect("Failed to ingest");

    assert!(note.id.is_some());
    assert!(!note.embedding.is_empty());
    assert_eq!(note.embedding.len(), 1024);
}

/// Test Search agent (requires inference backends)
#[tokio::test]
#[ignore = "Requires local inference backends (TEI/TGI or Ollama)"]
async fn test_search_agent() {
    use graphrag_agents::{LibrarianAgent, SearchAgent, TeiClient, TgiClient};

    let db = init_memory().await.expect("Failed to init db");
    let repo = Repository::new(db);
    let tei = TeiClient::default_local();
    let tgi = TgiClient::default_local();

    if tei.health().await.is_err() || tgi.health().await.is_err() {
        eprintln!("Skipping test: inference backends not available");
        return;
    }

    let librarian = LibrarianAgent::new(repo.clone(), tei.clone(), tgi);
    let search = SearchAgent::new(repo.clone(), tei);

    librarian
        .ingest_text(
            "Machine learning algorithms learn patterns from data",
            Some("ML Basics".into()),
            vec![],
        )
        .await
        .unwrap();
    librarian
        .ingest_text(
            "Neural networks are inspired by biological neurons",
            Some("Neural Networks".into()),
            vec![],
        )
        .await
        .unwrap();
    librarian
        .ingest_text(
            "Rust programming provides memory safety guarantees",
            Some("Rust Safety".into()),
            vec![],
        )
        .await
        .unwrap();

    let results = search
        .search("how do neural networks learn", 5)
        .await
        .unwrap();
    assert!(!results.is_empty());
}

/// Test Gardener agent (requires inference backends)
#[tokio::test]
#[ignore = "Requires local inference backends (TEI/TGI or Ollama)"]
async fn test_gardener_agent() {
    use graphrag_agents::{GardenerAgent, LibrarianAgent, TeiClient, TgiClient};

    let db = init_memory().await.expect("Failed to init db");
    let repo = Repository::new(db);
    let tei = TeiClient::default_local();
    let tgi = TgiClient::default_local();

    if tei.health().await.is_err() || tgi.health().await.is_err() {
        eprintln!("Skipping test: inference backends not available");
        return;
    }

    let librarian = LibrarianAgent::new(repo.clone(), tei, tgi);
    let gardener = GardenerAgent::new(repo.clone()).with_threshold(0.5);

    librarian
        .ingest_text(
            "Machine learning models can be trained on large datasets",
            None,
            vec![],
        )
        .await
        .unwrap();
    librarian
        .ingest_text(
            "Training data is essential for machine learning systems",
            None,
            vec![],
        )
        .await
        .unwrap();

    let orphans = gardener.find_orphans().await.unwrap();
    assert_eq!(orphans.len(), 2);

    let suggestions = gardener.suggest_connections().await.unwrap();
    println!("Generated {} suggestions", suggestions.len());
}

/// End-to-end test (requires inference backends)
#[tokio::test]
#[ignore = "Requires local inference backends (TEI/TGI or Ollama)"]
async fn test_e2e_workflow() {
    use graphrag_agents::{GardenerAgent, LibrarianAgent, SearchAgent, TeiClient, TgiClient};

    let db = init_memory().await.expect("Failed to init db");
    let repo = Repository::new(db);
    let tei = TeiClient::default_local();
    let tgi = TgiClient::default_local();

    if tei.health().await.is_err() || tgi.health().await.is_err() {
        eprintln!("Skipping test: inference backends not available");
        return;
    }

    let librarian = LibrarianAgent::new(repo.clone(), tei.clone(), tgi);
    let search = SearchAgent::new(repo.clone(), tei.clone());
    let gardener = GardenerAgent::new(repo.clone()).with_threshold(0.6);

    let topics = vec![
        (
            "Transformers",
            "Transformer models use self-attention mechanisms",
        ),
        ("BERT", "BERT is a transformer-based model for NLP tasks"),
        (
            "GPT",
            "GPT models are autoregressive transformers for text generation",
        ),
        (
            "Attention",
            "Attention mechanisms allow models to focus on relevant parts of input",
        ),
    ];

    for (title, content) in topics {
        librarian
            .ingest_text(content, Some(title.into()), vec!["ai".into(), "nlp".into()])
            .await
            .unwrap();
    }

    let stats = repo.get_stats().await.unwrap();
    assert_eq!(stats.note_count, 4);

    let results = search
        .search("how does attention work in transformers", 3)
        .await
        .unwrap();
    assert!(!results.is_empty());
    for r in &results {
        println!("  - {}", r.title.as_deref().unwrap_or("(untitled)"));
    }

    let report = gardener.run_maintenance().await.unwrap();
    println!(
        "Gardener: {} orphans, {} suggestions, {} applied",
        report.orphans_found, report.suggestions_generated, report.connections_applied
    );

    let final_stats = repo.get_stats().await.unwrap();
    println!(
        "Final stats: {} notes, {} edges",
        final_stats.note_count, final_stats.edge_count
    );
}
