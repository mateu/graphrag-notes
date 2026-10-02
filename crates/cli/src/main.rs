//! GraphRAG Notes command-line entry point.

mod app;
mod backup;
mod cli;
mod commands;
mod dispatch;
mod doctor;
mod eval;
mod explain;
mod init;
mod interactive;
mod output;

#[tokio::main]
async fn main() -> std::process::ExitCode {
    let exit_code = match app::run().await {
        Ok(()) => output::ExitCode::Success,
        Err(error) => {
            // Doctor has already rendered its report; preserve its stdout,
            // stderr, and exit status without printing an additional error.
            if error.downcast_ref::<app::DoctorExit>().is_none() {
                eprintln!("Error: {error:#}");
            }
            app::exit_code_for(&error)
        }
    };
    // Return through the runtime so embedded database workers are dropped
    // before the process terminates, including their native RocksDB resources.
    std::process::ExitCode::from(exit_code as u8)
}
