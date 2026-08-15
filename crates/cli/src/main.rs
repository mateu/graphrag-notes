//! GraphRAG Notes command-line entry point.

mod app;
mod backup;
mod cli;
mod commands;
mod dispatch;
mod doctor;
mod eval;
mod explain;
mod interactive;
mod output;

#[tokio::main]
async fn main() {
    let exit_code = match app::run().await {
        Ok(()) => output::ExitCode::Success,
        Err(error) => {
            eprintln!("Error: {error:#}");
            app::exit_code_for(&error)
        }
    };
    std::process::exit(exit_code as i32);
}
