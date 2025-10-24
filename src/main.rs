mod orchestra;

use orchestra::orchestrator::Orchestrator;
use orchestra::worker::Worker;
use std::env;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!(
            "Usage: {} <orchestrator|worker> <address> [num_workers]",
            args[0]
        );
        process::exit(1);
    }

    let mode = &args[1];
    let address = &args[2];

    match mode.as_str() {
        "orchestrator" => {
            let waiting_for = if args.len() > 3 {
                args[3].parse::<usize>().unwrap_or(4)
            } else {
                4
            };

            let mut orchestrator = Orchestrator::new(waiting_for);

            if let Err(e) = orchestrator.start(address) {
                eprintln!("Error starting orchestrator: {}", e);
                process::exit(1);
            }
        }

        "worker" => {
            match Worker::new(address) {
                Ok(worker) => {
                    
                    let sys = actix_rt::System::new();
                    if let Err(e) = sys.block_on(async move { worker.start().await }) {
                        eprintln!("Error during worker operation: {}", e);
                        process::exit(1);
                    }
                }
                Err(e) => {
                    eprintln!("Error connecting to Orchestrator: {}", e);
                    process::exit(1);
                }
            }
        }

        _ => {
            eprintln!(
                "Unknown mode: {}. You must use 'orchestrator' or 'worker'.",
                mode
            );
            process::exit(1);
        }
    }
}
