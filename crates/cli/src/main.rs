//! GraphRAG Notes CLI
//! 
//! A command-line interface for the GraphRAG Notes system.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use graphrag_agents::{LibrarianAgent, SearchAgent, GardenerAgent, MlClient};
use graphrag_db::{Repository, init_persistent, init_memory};
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
    
    /// ML worker URL
    #[arg(long, default_value = "http://localhost:8100")]
    ml_url: String,
    
    /// Use in-memory database (for testing)
    #[arg(long)]
    memory: bool,
    
    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
    /// Ollama URL
    #[arg(long)]
    ollama_url: Option<String>,
    
    /// Ollama Model
    #[arg(long, default_value = "phi4-mini:latest")]
    ollama_model: String,
    
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
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // Setup logging
    let log_level = if cli.verbose { Level::DEBUG } else { Level::INFO };
    let subscriber = FmtSubscriber::builder()
        .with_max_level(log_level)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;
    
    // Initialize database
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
    let mut ml = MlClient::new(&cli.ml_url);
    if let Some(url) = cli.ollama_url {
        ml = ml.with_ollama(url, cli.ollama_model);
    }
    
    // Check ML worker health
    match ml.health().await {
        Ok(true) => info!("ML worker is healthy"),
        Ok(false) => {
            eprintln!("Warning: ML worker returned unhealthy status");
        }
        Err(e) => {
            eprintln!("Warning: Could not connect to ML worker at {}: {}", cli.ml_url, e);
            eprintln!("Some features will not work. Start the ML worker with:");
            eprintln!("  cd python && uv run python -m ml_worker.server");
        }
    }
    
    // Execute command
    match cli.command {
        Commands::Add { content, title, tags } => {
            cmd_add(repo, ml, content, title, tags).await?;
        }
        Commands::Import { path } => {
            cmd_import(repo, ml, path).await?;
        }
        Commands::Search { query, limit, context } => {
            cmd_search(repo, ml, query, limit, context).await?;
        }
        Commands::List { limit } => {
            cmd_list(repo, limit).await?;
        }
        Commands::Garden { dry_run } => {
            cmd_garden(repo, ml, dry_run).await?;
        }
        Commands::Stats => {
            cmd_stats(repo).await?;
        }
        Commands::Interactive => {
            cmd_interactive(repo, ml).await?;
        }
    }
    
    Ok(())
}

async fn cmd_add(
    repo: Repository,
    ml: MlClient,
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
    
    let librarian = LibrarianAgent::new(repo, ml);
    let note = librarian.ingest_text(content, title, tags).await?;
    
    println!("✓ Created note: {}", note.id.as_ref().map(|id| id.to_string()).unwrap_or_else(|| "(no id)".to_string()));
    
    Ok(())
}

async fn cmd_import(
    repo: Repository,
    ml: MlClient,
    path: PathBuf,
) -> Result<()> {
    let content = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read file: {}", path.display()))?;
    
    let librarian = LibrarianAgent::new(repo, ml);
    let notes = librarian.ingest_markdown(
        path.to_str().unwrap_or("unknown"),
        content,
    ).await?;
    
    println!("✓ Imported {} notes from {}", notes.len(), path.display());
    
    Ok(())
}

async fn cmd_search(
    repo: Repository,
    ml: MlClient,
    query: String,
    limit: usize,
    context: bool,
) -> Result<()> {
    let search = SearchAgent::new(repo, ml);
    
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

async fn cmd_garden(repo: Repository, ml: MlClient, dry_run: bool) -> Result<()> {
    let gardener = GardenerAgent::new(repo, ml);
    
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

async fn cmd_interactive(repo: Repository, ml: MlClient) -> Result<()> {
    let librarian = LibrarianAgent::new(repo.clone(), ml.clone());
    let search = SearchAgent::new(repo.clone(), ml.clone());
    let gardener = GardenerAgent::new(repo.clone(), ml);
    
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
