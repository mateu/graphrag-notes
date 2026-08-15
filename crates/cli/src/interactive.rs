//! Interactive REPL implementation.

use crate::app::configured_search_agent;
use anyhow::Result;
use graphrag_agents::{
    GardenerAgent, LibrarianAgent, LibrarianRuntimeConfig, SharedEmbedder, SharedEntityExtractor,
};
use graphrag_config::SearchConfig;
use graphrag_core::record_id_to_string;
use graphrag_db::Repository;
use std::io::{self, BufRead, Write};

#[allow(clippy::too_many_arguments)]
pub(crate) async fn cmd_interactive(
    repo: Repository,
    tei: SharedEmbedder,
    tgi: SharedEntityExtractor,
    librarian_config: LibrarianRuntimeConfig,
    default_search_limit: usize,
    search_config: SearchConfig,
    similarity_threshold: f32,
    auto_apply_threshold: f32,
    auto_apply: bool,
    max_suggestions: usize,
) -> Result<()> {
    let librarian = LibrarianAgent::new(repo.clone(), tei.clone(), tgi.clone())
        .with_runtime_config(librarian_config);
    let search = configured_search_agent(repo.clone(), tei.clone(), &search_config);
    let gardener = GardenerAgent::new(repo.clone())
        .with_threshold(similarity_threshold)
        .with_auto_apply_policy(auto_apply, auto_apply_threshold)
        .with_max_suggestions(max_suggestions);

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
        let cmd = parts.first().copied().unwrap_or("");
        let arg = parts.get(1).copied().unwrap_or("");

        match cmd {
            "" => continue,

            "add" | "a" => {
                if arg.is_empty() {
                    println!("Usage: add <content>");
                    continue;
                }
                match librarian.ingest_text(arg, None, vec![]).await {
                    Ok(note) => println!(
                        "✓ Added: {}",
                        note.id
                            .as_ref()
                            .map(record_id_to_string)
                            .unwrap_or_else(|| "(no id)".to_string())
                    ),
                    Err(e) => println!("Error: {}", e),
                }
            }

            "search" | "s" => {
                if arg.is_empty() {
                    println!("Usage: search <query>");
                    continue;
                }
                match search.search(arg, default_search_limit).await {
                    Ok(results) => {
                        if results.is_empty() {
                            println!("No results.");
                        } else {
                            for r in results {
                                let preview: String = r.content.chars().take(100).collect();
                                println!(
                                    "• {} - {}{}",
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

            "list" | "l" => match repo.list_notes(10).await {
                Ok(notes) => {
                    if notes.is_empty() {
                        println!("No notes yet.");
                    } else {
                        for note in notes {
                            let preview: String = note.content.chars().take(60).collect();
                            println!(
                                "• {} - {}{}",
                                note.title.as_deref().unwrap_or("(untitled)"),
                                preview,
                                if note.content.len() > 60 { "..." } else { "" }
                            );
                        }
                    }
                }
                Err(e) => println!("Error: {}", e),
            },

            "garden" | "g" => match gardener.run_maintenance().await {
                Ok(report) => {
                    println!(
                        "Maintenance: {} orphans, {} suggestions, {} applied",
                        report.orphans_found,
                        report.suggestions_generated,
                        report.connections_applied,
                    );
                }
                Err(e) => println!("Error: {}", e),
            },

            "stats" => match repo.get_stats().await {
                Ok(s) => println!(
                    "Notes: {}, Entities: {}, Edges: {}",
                    s.note_count, s.entity_count, s.edge_count
                ),
                Err(e) => println!("Error: {}", e),
            },

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
                println!(
                    "Unknown command: {}. Type 'help' for available commands.",
                    cmd
                );
            }
        }

        println!();
    }

    Ok(())
}
