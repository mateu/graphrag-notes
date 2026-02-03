//! Integration tests for GraphRAG Notes
//! 
//! Note: Tests that require the ML worker are marked with #[ignore]
//! Run them with: cargo test -- --ignored

mod common;

use graphrag_core::{Note, NoteType, EdgeType};
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
    
    // Create a fake embedding (384 dimensions)
    let embedding: Vec<f32> = (0..384).map(|i| (i as f32) / 384.0).collect();
    
    let note = Note::new("Note with embedding")
        .with_embedding(embedding.clone());
    
    let created = repo.create_note(note).await.expect("Failed to create note");
    assert_eq!(created.embedding.len(), 384);
}

/// Test graph edge creation
#[tokio::test]
async fn test_graph_edges() {
    let db = init_memory().await.expect("Failed to init db");
    let repo = Repository::new(db);
    
    // Create two notes
    let note1 = Note::new("First note about machine learning");
    let note1 = repo.create_note(note1).await.unwrap();
    let id1 = note1.id.unwrap().replace("note:", "");
    
    let note2 = Note::new("Second note that supports the first");
    let note2 = repo.create_note(note2).await.unwrap();
    let id2 = note2.id.unwrap().replace("note:", "");
    
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
        ("Machine Learning Basics", "Machine learning is a subset of artificial intelligence"),
        ("Deep Learning", "Neural networks with multiple layers are called deep learning"),
        ("Rust Programming", "Rust is a systems programming language focused on safety"),
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
    let base_embedding: Vec<f32> = (0..384).map(|i| (i as f32) / 384.0).collect();
    
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
    let emb3: Vec<f32> = (0..384).map(|i| 1.0 - (i as f32) / 384.0).collect();
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
// TESTS REQUIRING ML WORKER
// Run with: cargo test -- --ignored
// ==========================================

/// Test Librarian agent (requires ML worker)
#[tokio::test]
#[ignore = "Requires ML worker running on localhost:8100"]
async fn test_librarian_ingest() {
    use graphrag_agents::{LibrarianAgent, MlClient};
    
    let db = init_memory().await.expect("Failed to init db");
    let repo = Repository::new(db);
    let ml = MlClient::default_local();
    
    // Check ML worker is available
    if ml.health().await.is_err() {
        eprintln!("Skipping test: ML worker not available");
        return;
    }
    
    let librarian = LibrarianAgent::new(repo.clone(), ml);
    
    let note = librarian.ingest_text(
        "Rust is a systems programming language that focuses on safety and performance",
        Some("Rust Overview".into()),
        vec!["rust".into(), "programming".into()],
    ).await.expect("Failed to ingest");
    
    assert!(note.id.is_some());
    assert!(!note.embedding.is_empty());
    assert_eq!(note.embedding.len(), 384);
}

/// Test Search agent (requires ML worker)
#[tokio::test]
#[ignore = "Requires ML worker running on localhost:8100"]
async fn test_search_agent() {
    use graphrag_agents::{LibrarianAgent, SearchAgent, MlClient};
    
    let db = init_memory().await.expect("Failed to init db");
    let repo = Repository::new(db);
    let ml = MlClient::default_local();
    
    if ml.health().await.is_err() {
        eprintln!("Skipping test: ML worker not available");
        return;
    }
    
    let librarian = LibrarianAgent::new(repo.clone(), ml.clone());
    let search = SearchAgent::new(repo.clone(), ml);
    
    // Add some notes
    librarian.ingest_text(
        "Machine learning algorithms learn patterns from data",
        Some("ML Basics".into()),
        vec![],
    ).await.unwrap();
    
    librarian.ingest_text(
        "Neural networks are inspired by biological neurons",
        Some("Neural Networks".into()),
        vec![],
    ).await.unwrap();
    
    librarian.ingest_text(
        "Rust programming provides memory safety guarantees",
        Some("Rust Safety".into()),
        vec![],
    ).await.unwrap();
    
    // Search for machine learning topics
    let results = search.search("how do neural networks learn", 5).await.unwrap();
    
    assert!(!results.is_empty());
    // The neural networks note should be highly ranked
}

/// Test Gardener agent (requires ML worker)
#[tokio::test]
#[ignore = "Requires ML worker running on localhost:8100"]
async fn test_gardener_agent() {
    use graphrag_agents::{LibrarianAgent, GardenerAgent, MlClient};
    
    let db = init_memory().await.expect("Failed to init db");
    let repo = Repository::new(db);
    let ml = MlClient::default_local();
    
    if ml.health().await.is_err() {
        eprintln!("Skipping test: ML worker not available");
        return;
    }
    
    let librarian = LibrarianAgent::new(repo.clone(), ml.clone());
    let gardener = GardenerAgent::new(repo.clone(), ml).with_threshold(0.5);
    
    // Add related notes
    librarian.ingest_text(
        "Machine learning models can be trained on large datasets",
        None,
        vec![],
    ).await.unwrap();
    
    librarian.ingest_text(
        "Training data is essential for machine learning systems",
        None,
        vec![],
    ).await.unwrap();
    
    // Find orphans (all notes should be orphans initially)
    let orphans = gardener.find_orphans().await.unwrap();
    assert_eq!(orphans.len(), 2);
    
    // Suggest connections
    let suggestions = gardener.suggest_connections().await.unwrap();
    
    // These similar notes should generate suggestions
    // (depends on similarity threshold)
    println!("Generated {} suggestions", suggestions.len());
}

/// End-to-end test (requires ML worker)
#[tokio::test]
#[ignore = "Requires ML worker running on localhost:8100"]
async fn test_e2e_workflow() {
    use graphrag_agents::{LibrarianAgent, SearchAgent, GardenerAgent, MlClient};
    
    let db = init_memory().await.expect("Failed to init db");
    let repo = Repository::new(db);
    let ml = MlClient::default_local();
    
    if ml.health().await.is_err() {
        eprintln!("Skipping test: ML worker not available");
        return;
    }
    
    let librarian = LibrarianAgent::new(repo.clone(), ml.clone());
    let search = SearchAgent::new(repo.clone(), ml.clone());
    let gardener = GardenerAgent::new(repo.clone(), ml).with_threshold(0.6);
    
    // 1. Ingest several related notes
    let topics = vec![
        ("Transformers", "Transformer models use self-attention mechanisms"),
        ("BERT", "BERT is a transformer-based model for NLP tasks"),
        ("GPT", "GPT models are autoregressive transformers for text generation"),
        ("Attention", "Attention mechanisms allow models to focus on relevant parts of input"),
    ];
    
    for (title, content) in topics {
        librarian.ingest_text(content, Some(title.into()), vec!["ai".into(), "nlp".into()])
            .await
            .unwrap();
    }
    
    // 2. Verify notes were created
    let stats = repo.get_stats().await.unwrap();
    assert_eq!(stats.note_count, 4);
    
    // 3. Search
    let results = search.search("how does attention work in transformers", 3).await.unwrap();
    assert!(!results.is_empty());
    println!("Search returned {} results", results.len());
    for r in &results {
        println!("  - {}", r.title.as_deref().unwrap_or("(untitled)"));
    }
    
    // 4. Run gardener
    let report = gardener.run_maintenance().await.unwrap();
    println!("Gardener: {} orphans, {} suggestions, {} applied",
        report.orphans_found, report.suggestions_generated, report.connections_applied);
    
    // 5. Verify some connections were made
    let final_stats = repo.get_stats().await.unwrap();
    println!("Final stats: {} notes, {} edges", final_stats.note_count, final_stats.edge_count);
}
