//! GraphRAG Notes CLI
//!
//! A command-line interface for the GraphRAG Notes system.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use graphrag_agents::{GardenerAgent, LibrarianAgent, SearchAgent, TeiClient, TgiClient};
use graphrag_core::ChatExport;
use graphrag_db::{init_memory, init_persistent, Repository};
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

/// GraphRAG Notes - An evolving knowledge graph for your notes
#[derive(Parser)]
#[command(name = "graphrag")]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Database path (defaults to ~/.graphrag/data)
    #[arg(short, long)]
    db_path: Option<PathBuf>,
    
    /// Use in-memory database (for testing)
    #[arg(long)]
    memory: bool,
    
    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
    
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Add a new note
    Add {
        /// Note content (reads from stdin if not provided)
        content: Option<String>,
        
        /// Note title
        #[arg(short, long)]
        title: Option<String>,
        
        /// Tags (comma-separated)
        #[arg(short = 'T', long)]
        tags: Option<String>,
    },
    
    /// Import from a file
    Import {
        /// Path to file
        path: PathBuf,
    },

    /// Import chat conversations from Claude Desktop or other chat exports
    ImportChats {
        /// Path to JSON file containing chat export
        path: PathBuf,

        /// Skip entity extraction (faster for testing)
        #[arg(long)]
        skip_extraction: bool,
    },
    
    /// Search notes
    Search {
        /// Search query
        query: String,
        
        /// Maximum results
        #[arg(short, long, default_value = "10")]
        limit: usize,
        
        /// Include graph context
        #[arg(short, long)]
        context: bool,
    },
    
    /// List recent notes
    List {
        /// Maximum results
        #[arg(short, long, default_value = "20")]
        limit: usize,
    },
    
    /// Run the gardener (maintenance)
    Garden {
        /// Only show suggestions, don't apply
        #[arg(long)]
        dry_run: bool,
    },
    
    /// Show database statistics
    Stats,
    
    /// Interactive mode
    Interactive,

    /// Show the embedding dimension from the active embeddings provider
    EmbeddingDim {
        /// Optional text to embed (defaults to "dimension probe")
        text: Option<String>,
    },

    /// Extract entities for notes that are missing entity links
    ExtractEntities {
        /// Maximum notes to process
        #[arg(short, long, default_value = "100")]
        limit: usize,
    },

    /// Show entities linked to a note
    ShowEntities {
        /// Note ID (e.g., note:xxxxxxxx)
        note_id: String,
    },

    /// Show a note by ID
    ShowNote {
        /// Note ID (e.g., note:xxxxxxxx)
        note_id: String,
    },

    /// List note-to-note edges
    ListEdges {
        /// Maximum edges per edge type
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },

    /// Show note-to-note edges for a specific note
    ShowNoteEdges {
        /// Note ID (e.g., note:xxxxxxxx)
        note_id: String,
    },

    /// Delete the local database (fresh start)
    ResetDb {
        /// Database path (defaults to ~/.graphrag/data)
        #[arg(short, long)]
        db_path: Option<PathBuf>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load environment variables from .env if present.
    dotenvy::dotenv().ok();

    let cli = Cli::parse();

    let skip_extraction = matches!(
        cli.command,
        Commands::ImportChats {
            skip_extraction: true,
            ..
        }
    );
    if skip_extraction {
        std::env::set_var("SKIP_ENTITY_EXTRACTION", "true");
    }

    if let Commands::EmbeddingDim { text } = &cli.command {
        let tei = TeiClient::default_local();
        let tei_ok = tei.health().await.unwrap_or(false);
        if !tei_ok {
            eprintln!("Error: embeddings service is not reachable.");
            eprintln!("  TEI (embeddings): {}", tei.base_url());
            anyhow::bail!("Embeddings service unavailable");
        }

        let probe = text.clone().unwrap_or_else(|| "dimension probe".to_string());
        let embedding = tei.embed(&probe, false).await?;
        println!("Embedding dimension: {}", embedding.len());
        return Ok(());
    }
    
    // Setup logging
    let log_level = if cli.verbose { Level::DEBUG } else { Level::INFO };
    let subscriber = FmtSubscriber::builder()
        .with_max_level(log_level)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;
    
    // Initialize database
    if let Commands::ResetDb { db_path } = &cli.command {
        let path = db_path.clone().unwrap_or_else(|| {
            let mut path = dirs::home_dir().expect("Could not find home directory");
            path.push(".graphrag");
            path.push("data");
            path
        });

        if path.exists() {
            std::fs::remove_dir_all(&path)
                .with_context(|| format!("Failed to remove db at {}", path.display()))?;
            println!("✓ Removed database at {}", path.display());
        } else {
            println!("Database not found at {}, nothing to remove", path.display());
        }
        return Ok(());
    }

    let db = if cli.memory {
        info!("Using in-memory database");
        init_memory().await?
    } else {
        let db_path = cli.db_path.unwrap_or_else(|| {
            let mut path = dirs::home_dir().expect("Could not find home directory");
            path.push(".graphrag");
            path.push("data");
            path
        });
        
        // Ensure directory exists
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        info!("Using database at: {}", db_path.display());
        init_persistent(&db_path).await?
    };
    
    let repo = Repository::new(db);
    let tei = TeiClient::default_local();
    let tgi = TgiClient::default_local();

    // Check inference services only when needed
    let needs_tei = matches!(
        cli.command,
        Commands::Add { .. }
            | Commands::Import { .. }
            | Commands::ImportChats { .. }
            | Commands::Search { .. }
            | Commands::Interactive
    );
    let needs_tgi = matches!(
        cli.command,
        Commands::Add { .. }
            | Commands::Import { .. }
            | Commands::ImportChats { .. }
            | Commands::Interactive
            | Commands::ExtractEntities { .. }
    ) && !skip_extraction;

    if needs_tei {
        let tei_ok = tei.health().await.unwrap_or(false);
        if !tei_ok {
            eprintln!("Error: embeddings service is not reachable.");
            eprintln!("  TEI (embeddings): {}", tei.base_url());
            eprintln!("Start it with: docker compose up -d");
            anyhow::bail!("Embeddings service unavailable");
        }
    }

    if needs_tgi {
        let tgi_ok = tgi.health().await.unwrap_or(false);
        if !tgi_ok {
            eprintln!("Error: extraction service is not reachable.");
            eprintln!("  TGI (extraction): {}", tgi.base_url());
            eprintln!("Start it with: docker compose up -d");
            anyhow::bail!("Extraction service unavailable");
        }
    }
    
    // Execute command
    match cli.command {
        Commands::Add { content, title, tags } => {
            cmd_add(repo, tei, tgi, content, title, tags).await?;
        }
        Commands::Import { path } => {
            cmd_import(repo, tei, tgi, path).await?;
        }
        Commands::ImportChats { path, .. } => {
            cmd_import_chats(repo, tei, tgi, path).await?;
        }
        Commands::Search { query, limit, context } => {
            cmd_search(repo, tei, query, limit, context).await?;
        }
        Commands::List { limit } => {
            cmd_list(repo, limit).await?;
        }
        Commands::Garden { dry_run } => {
            cmd_garden(repo, dry_run).await?;
        }
        Commands::Stats => {
            cmd_stats(repo).await?;
        }
        Commands::Interactive => {
            cmd_interactive(repo, tei, tgi).await?;
        }
        Commands::ExtractEntities { limit } => {
            cmd_extract_entities(repo, tgi, limit).await?;
        }
        Commands::ShowEntities { note_id } => {
            cmd_show_entities(repo, note_id).await?;
        }
        Commands::ShowNote { note_id } => {
            cmd_show_note(repo, note_id).await?;
        }
        Commands::ListEdges { limit } => {
            cmd_list_edges(repo, limit).await?;
        }
        Commands::ShowNoteEdges { note_id } => {
            cmd_show_note_edges(repo, note_id).await?;
        }
        Commands::EmbeddingDim { .. } => {
            // Handled before database init.
        }
        Commands::ResetDb { .. } => {
            // Handled before database init.
        }
    }
    
    Ok(())
}

async fn cmd_add(
    repo: Repository,
    tei: TeiClient,
    tgi: TgiClient,
    content: Option<String>,
    title: Option<String>,
    tags: Option<String>,
) -> Result<()> {
    let content = match content {
        Some(c) => c,
        None => {
            // Read from stdin
            eprintln!("Enter note content (Ctrl+D to finish):");
            let stdin = io::stdin();
            let lines: Vec<String> = stdin.lock().lines()
                .filter_map(|l| l.ok())
                .collect();
            lines.join("\n")
        }
    };
    
    if content.trim().is_empty() {
        anyhow::bail!("Note content cannot be empty");
    }
    
    let tags = tags
        .map(|t| t.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_default();
    
    let librarian = LibrarianAgent::new(repo, tei, tgi);
    let note = librarian.ingest_text(content, title, tags).await?;
    
    println!("✓ Created note: {}", note.id.as_ref().map(|id| id.to_string()).unwrap_or_else(|| "(no id)".to_string()));
    
    Ok(())
}

async fn cmd_import(
    repo: Repository,
    tei: TeiClient,
    tgi: TgiClient,
    path: PathBuf,
) -> Result<()> {
    let content = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read file: {}", path.display()))?;

    let librarian = LibrarianAgent::new(repo, tei, tgi);
    let notes = librarian
        .ingest_markdown(path.to_str().unwrap_or("unknown"), content)
        .await?;

    println!("✓ Imported {} notes from {}", notes.len(), path.display());

    Ok(())
}

async fn cmd_import_chats(
    repo: Repository,
    tei: TeiClient,
    tgi: TgiClient,
    path: PathBuf,
) -> Result<()> {
    let content = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read file: {}", path.display()))?;

    // Parse the chat export
    let export = ChatExport::from_json(&content)
        .with_context(|| format!("Failed to parse chat export from: {}", path.display()))?;

    println!(
        "Found {} conversations with {} total messages",
        export.conversation_count(),
        export.total_messages()
    );

    let librarian = LibrarianAgent::new(repo, tei, tgi);
    let result = librarian
        .ingest_chat_export(export, Some(path.display().to_string()))
        .await?;

    println!("\n✓ Import complete:");
    println!(
        "  • Conversations imported: {}",
        result.conversations_imported
    );
    println!("  • Notes created: {}", result.notes_created);

    if result.conversations_failed > 0 {
        println!(
            "  • Conversations failed: {}",
            result.conversations_failed
        );
        for error in &result.errors {
            println!("    - {}", error);
        }
    }

    Ok(())
}

async fn cmd_extract_entities(
    repo: Repository,
    tgi: TgiClient,
    limit: usize,
) -> Result<()> {
    let librarian = LibrarianAgent::new(repo, TeiClient::default_local(), tgi);
    let processed = librarian.extract_entities_for_notes(limit).await?;
    println!("✓ Extracted entities for {} notes", processed);
    Ok(())
}

async fn cmd_show_entities(repo: Repository, note_id: String) -> Result<()> {
    let key = note_id.strip_prefix("note:").unwrap_or(&note_id);
    let note = repo
        .get_note(key)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Note not found: {}", note_id))?;

    let title = note.title.as_deref().unwrap_or("(untitled)");
    println!("Note: {} ({})", title, note_id);

    let entities = repo.get_entities_for_note(key).await?;
    if entities.is_empty() {
        println!("No entities linked to this note.");
        return Ok(());
    }

    println!("Entities ({}):", entities.len());
    for entity in entities {
        let entity_type = serde_json::to_string(&entity.entity_type)
            .unwrap_or_else(|_| "\"other\"".to_string())
            .trim_matches('"')
            .to_string();
        println!(
            "  • {} [{}]",
            entity.name,
            entity_type
        );
    }

    Ok(())
}

async fn cmd_show_note(repo: Repository, note_id: String) -> Result<()> {
    let key = note_id.strip_prefix("note:").unwrap_or(&note_id);
    let note = repo
        .get_note(key)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Note not found: {}", note_id))?;

    println!(
        "Note: {} ({})",
        note.title.as_deref().unwrap_or("(untitled)"),
        note_id
    );
    println!("Type: {}", note.note_type);
    println!("Tags: {}", if note.tags.is_empty() { "(none)".into() } else { note.tags.join(", ") });
    println!();
    println!("{}", note.content);
    Ok(())
}

async fn cmd_list_edges(repo: Repository, limit: usize) -> Result<()> {
    let edges = repo.list_note_edges(limit).await?;
    if edges.is_empty() {
        println!("No note edges found.");
        return Ok(());
    }

    println!("Note edges (up to {} per type):", limit);
    for edge in edges {
        let reason = edge.reason.as_deref().unwrap_or("");
        let confidence = edge.confidence.map(|c| format!("{:.2}", c)).unwrap_or_else(|| "-".into());
        println!(
            "  • {}: {} -> {} (confidence: {}){}",
            edge.edge_type,
            edge.in_id,
            edge.out_id,
            confidence,
            if reason.is_empty() { "".into() } else { format!(" reason: {}", reason) }
        );
    }
    Ok(())
}

async fn cmd_show_note_edges(repo: Repository, note_id: String) -> Result<()> {
    let edges = repo.get_note_edges(&note_id).await?;
    if edges.is_empty() {
        println!("No edges found for {}", note_id);
        return Ok(());
    }

    println!("Edges for {}:", note_id);
    for edge in edges {
        let reason = edge.reason.as_deref().unwrap_or("");
        let confidence = edge.confidence.map(|c| format!("{:.2}", c)).unwrap_or_else(|| "-".into());
        println!(
            "  • {}: {} -> {} (confidence: {}){}",
            edge.edge_type,
            edge.in_id,
            edge.out_id,
            confidence,
            if reason.is_empty() { "".into() } else { format!(" reason: {}", reason) }
        );
    }
    Ok(())
}

async fn cmd_search(
    repo: Repository,
    tei: TeiClient,
    query: String,
    limit: usize,
    context: bool,
) -> Result<()> {
    let search = SearchAgent::new(repo, tei);
    
    if context {
        let results = search.search_with_context(&query, limit).await?;
        
        if results.is_empty() {
            println!("No results found.");
            return Ok(());
        }
        
        println!("Found {} results:\n", results.len());
        
        for (i, result) in results.iter().enumerate() {
            let r = &result.result;
            println!("{}. {}", i + 1, r.title.as_deref().unwrap_or("(untitled)"));
            println!("   ID: {}", r.id);
            println!("   Type: {}", r.note_type);
            
            // Truncate content for display
            let preview: String = r.content.chars().take(200).collect();
            println!("   {}{}", preview, if r.content.len() > 200 { "..." } else { "" });
            
            if let Some(ref related) = result.related {
                let total = related.supporting.len() 
                    + related.contradicting.len() 
                    + related.related.len();
                if total > 0 {
                    println!("   → {} related notes", total);
                }
            }
            
            println!();
        }
    } else {
        let results = search.search(&query, limit).await?;
        
        if results.is_empty() {
            println!("No results found.");
            return Ok(());
        }
        
        println!("Found {} results:\n", results.len());
        
        for (i, r) in results.iter().enumerate() {
            println!("{}. {}", i + 1, r.title.as_deref().unwrap_or("(untitled)"));
            println!("   ID: {}", r.id);
            
            let preview: String = r.content.chars().take(200).collect();
            println!("   {}{}", preview, if r.content.len() > 200 { "..." } else { "" });
            println!();
        }
    }
    
    Ok(())
}

async fn cmd_list(repo: Repository, limit: usize) -> Result<()> {
    let notes = repo.list_notes(limit).await?;
    
    if notes.is_empty() {
        println!("No notes yet. Add one with: graphrag add \"your note\"");
        return Ok(());
    }
    
    println!("Recent notes ({}):\n", notes.len());
    
    for note in notes {
        let title = note.title.as_deref().unwrap_or("(untitled)");
        let id = note.id.to_string();
        let preview: String = note.content.chars().take(80).collect();
        
        println!("• {} [{}]", title, id);
        println!("  {}{}", preview, if note.content.len() > 80 { "..." } else { "" });
        println!();
    }
    
    Ok(())
}

async fn cmd_garden(repo: Repository, dry_run: bool) -> Result<()> {
    let gardener = GardenerAgent::new(repo);
    
    if dry_run {
        println!("Finding orphan notes...\n");
        
        let orphans = gardener.find_orphans().await?;
        
        if orphans.is_empty() {
            println!("No orphan notes found. Your knowledge graph is well connected!");
            return Ok(());
        }
        
        println!("Found {} orphan notes:", orphans.len());
        for orphan in &orphans {
            println!("  • {}", orphan.title.as_deref().unwrap_or("(untitled)"));
        }
        
        println!("\nGenerating suggestions...\n");
        
        let suggestions = gardener.suggest_connections().await?;
        
        if suggestions.is_empty() {
            println!("No connection suggestions found.");
        } else {
            println!("Suggested connections:");
            for s in &suggestions {
                println!(
                    "  {} → {} ({:.0}%: {})",
                    s.from_note.title.as_deref().unwrap_or("(untitled)"),
                    s.to_note.title.as_deref().unwrap_or("(untitled)"),
                    s.similarity * 100.0,
                    s.reason,
                );
            }
        }
    } else {
        println!("Running maintenance...\n");
        
        let report = gardener.run_maintenance().await?;
        
        println!("Maintenance complete:");
        println!("  • Orphans found: {}", report.orphans_found);
        println!("  • Suggestions generated: {}", report.suggestions_generated);
        println!("  • Connections applied: {}", report.connections_applied);
        println!("  • Orphans remaining: {}", report.orphans_remaining);
    }
    
    Ok(())
}

async fn cmd_stats(repo: Repository) -> Result<()> {
    let stats = repo.get_stats().await?;
    
    println!("Database Statistics:");
    println!("  • Notes: {}", stats.note_count);
    println!("  • Entities: {}", stats.entity_count);
    println!("  • Sources: {}", stats.source_count);
    println!("  • Edges: {}", stats.edge_count);
    
    Ok(())
}

async fn cmd_interactive(
    repo: Repository,
    tei: TeiClient,
    tgi: TgiClient,
) -> Result<()> {
    let librarian = LibrarianAgent::new(repo.clone(), tei.clone(), tgi.clone());
    let search = SearchAgent::new(repo.clone(), tei.clone());
    let gardener = GardenerAgent::new(repo.clone());
    
    println!("GraphRAG Notes - Interactive Mode");
    println!("Commands: add, search, list, garden, stats, help, quit");
    println!();
    
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    
    loop {
        print!("graphrag> ");
        stdout.flush()?;
        
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break; // EOF
        }
        
        let parts: Vec<&str> = line.trim().splitn(2, ' ').collect();
        let cmd = parts.first().map(|s| *s).unwrap_or("");
        let arg = parts.get(1).map(|s| *s).unwrap_or("");
        
        match cmd {
            "" => continue,
            
            "add" | "a" => {
                if arg.is_empty() {
                    println!("Usage: add <content>");
                    continue;
                }
                match librarian.ingest_text(arg, None, vec![]).await {
                    Ok(note) => println!("✓ Added: {}", note.id.as_ref().map(|id| id.to_string()).unwrap_or_else(|| "(no id)".to_string())), 
                    Err(e) => println!("Error: {}", e),
                }
            }
            
            "search" | "s" => {
                if arg.is_empty() {
                    println!("Usage: search <query>");
                    continue;
                }
                match search.search(arg, 5).await {
                    Ok(results) => {
                        if results.is_empty() {
                            println!("No results.");
                        } else {
                            for r in results {
                                let preview: String = r.content.chars().take(100).collect();
                                println!("• {} - {}{}", 
                                    r.title.as_deref().unwrap_or("(untitled)"),
                                    preview,
                                    if r.content.len() > 100 { "..." } else { "" }
                                );
                            }
                        }
                    }
                    Err(e) => println!("Error: {}", e),
                }
            }
            
            "list" | "l" => {
                match repo.list_notes(10).await {
                    Ok(notes) => {
                        if notes.is_empty() {
                            println!("No notes yet.");
                        } else {
                            for note in notes {
                                let preview: String = note.content.chars().take(60).collect();
                                println!("• {} - {}{}", 
                                    note.title.as_deref().unwrap_or("(untitled)"),
                                    preview,
                                    if note.content.len() > 60 { "..." } else { "" }
                                );
                            }
                        }
                    }
                    Err(e) => println!("Error: {}", e),
                }
            }
            
            "garden" | "g" => {
                match gardener.run_maintenance().await {
                    Ok(report) => {
                        println!("Maintenance: {} orphans, {} suggestions, {} applied",
                            report.orphans_found,
                            report.suggestions_generated,
                            report.connections_applied,
                        );
                    }
                    Err(e) => println!("Error: {}", e),
                }
            }
            
            "stats" => {
                match repo.get_stats().await {
                    Ok(s) => println!("Notes: {}, Entities: {}, Edges: {}", 
                        s.note_count, s.entity_count, s.edge_count),
                    Err(e) => println!("Error: {}", e),
                }
            }
            
            "help" | "h" | "?" => {
                println!("Commands:");
                println!("  add <content>    - Add a new note");
                println!("  search <query>   - Search notes");
                println!("  list             - List recent notes");
                println!("  garden           - Run maintenance");
                println!("  stats            - Show statistics");
                println!("  quit             - Exit");
            }
            
            "quit" | "q" | "exit" => {
                println!("Goodbye!");
                break;
            }
            
            _ => {
                println!("Unknown command: {}. Type 'help' for available commands.", cmd);
            }
        }
        
        println!();
    }
    
    Ok(())
}
