use std::error::Error;

use clap::{Parser, Subcommand};

use hal::{client, server};

const RUST_LOG: &str = "RUST_LOG";

const DEFAULT_HTTP_PORT: u16 = 8000;
const DEFAULT_MODEL: &str = "mistral";

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[derive(Debug)]
struct Cli {
    #[arg(short, long, default_value_t = DEFAULT_HTTP_PORT)]
    port: u16,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Start hal
    Serve {
        #[arg(default_value_t = DEFAULT_MODEL.to_string())]
        model: String,

        #[arg(short, long, default_value_t = DEFAULT_HTTP_PORT)]
        port: u16,
    },
}

fn init_logging() {
    // Hacky way to set default logging level
    if std::env::var(RUST_LOG).is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Some(Commands::Serve { model, port }) => {
            init_logging();
            server::start_server(port, model).await.map_err(|err| {
                log::error!("Server start failed: {}", err.to_string());
                err
            })
        }
        _ => client::run_client(cli.port).await.map_err(|err| {
            log::error!("Client failed: {}", err.to_string());
            err
        }),
    };

    let exit_code = match result {
        Ok(_) => 0,
        _ => 1,
    };

    std::process::exit(exit_code);
}
