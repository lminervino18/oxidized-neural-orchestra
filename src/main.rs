mod orchestra; // Declare the 'orchestra' module to access its files

use orchestra::orchestrator::Orchestrator; // Use the Orchestrator struct from the orchestrator module
use orchestra::worker::Worker; // Use the Worker struct from the worker module
use std::env;
use std::process;
fn main() {
    // Read the arguments to determine if it's Orchestrator or Worker
    let args: Vec<String> = env::args().collect();

    // Ensure the correct arguments are received
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <orchestrator|worker> <address> [num_workers]",
            args[0]
        );
        process::exit(1);
    }

    let mode = &args[1];
    let address = &args[2]; // Connection address

    match mode.as_str() {
        "orchestrator" => {
            // If orchestrator, receive the number of workers to wait for
            let waiting_for = if args.len() > 3 {
                args[3].parse::<usize>().unwrap_or(4) // Default to 4 workers
            } else {
                4 // Default to 4 workers
            };

            // Create and run the Orchestrator
            let mut orchestrator = Orchestrator::new(waiting_for);

            // Now handle the Result returned by the `start` function
            if let Err(e) = orchestrator.start(address) {
                eprintln!("Error starting orchestrator: {}", e);
                process::exit(1);
            }
        }
        "worker" => {
            // If worker, run the Worker
            match Worker::new(address) {
                Ok(worker) => {
                    if let Err(e) = worker.start() {
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
