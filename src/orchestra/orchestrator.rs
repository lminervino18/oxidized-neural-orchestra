use crate::orchestra::communication::Communication; // Use the Communication struct from the communication module
use crate::orchestra::error::SystemError; // Import error handling
use std::net::TcpListener; // To listen for incoming TCP connections
use std::sync::{Arc, Mutex}; // For thread-safety when sharing TcpStream
use std::thread; // For spawning threads to handle communication

/// Structure representing the Orchestrator.
///
/// The orchestrator coordinates the connection and communication with workers.
/// It waits for workers to connect and then manages communication with them in parallel.
/// Workers are managed using an `Arc<Mutex<Communication>>` to ensure safe and shared access.
pub struct Orchestrator {
    workers: Vec<Arc<Mutex<Communication>>>, // Stores worker connections using the `Communication` struct
    waiting_for: usize,                      // Number of workers the orchestrator is waiting for
}

impl Orchestrator {
    /// Creates a new orchestrator that waits for a specific number of workers to connect.
    ///
    /// # Parameters
    /// - `waiting_for`: The number of workers the orchestrator should wait for.
    ///
    /// # Returns
    /// Returns a new instance of `Orchestrator`.
    pub fn new(waiting_for: usize) -> Self {
        Orchestrator {
            workers: Vec::new(),
            waiting_for,
        }
    }

    /// Starts the orchestrator and waits for workers to connect.
    ///
    /// This method establishes connections with workers and begins listening for them.
    /// When workers connect, the orchestrator manages communication with them.
    ///
    /// # Parameters
    /// - `address`: Address where the orchestrator expects to receive connections (e.g., "127.0.0.1:5000").
    ///
    /// # Returns
    /// Returns a Result with an error in case of failure.
    pub fn start(&mut self, address: &str) -> Result<(), SystemError> {
        println!(
            "Orchestrator started, waiting for {} workers...",
            self.waiting_for
        );

        // Bind to the given address and start listening for incoming connections
        let listener = TcpListener::bind(address).map_err(|e| {
            SystemError::ConnectionError(format!("Failed to bind to address: {}", e))
        })?;

        println!("Orchestrator listening on {}", address);

        // Keep listening indefinitely for worker connections
        while self.workers.len() < self.waiting_for {
            match listener.accept() {
                Ok((stream, _)) => {
                    // For each incoming connection (worker), create a Communication instance
                    let worker = Communication::new_from_stream(stream)?;
                    let worker_arc = Arc::new(Mutex::new(worker)); // Wrap the Communication in Arc<Mutex> for thread safety
                    self.workers.push(worker_arc);

                    println!(
                        "Worker connected! Total workers connected: {}",
                        self.workers.len()
                    );

                    // If the required number of workers is connected, stop listening
                    if self.workers.len() == self.waiting_for {
                        break;
                    }
                }
                Err(e) => {
                    eprintln!("Error accepting worker connection: {}", e);
                }
            }
        }

        // Once all workers are connected, start handling communication with them
        self.handle_workers()?;
        Ok(())
    }

    /// Handles communication with connected workers.
    ///
    /// This method runs in a separate thread for each worker.
    /// It receives data from each worker and sends a response.
    ///
    /// # Returns
    /// Returns a Result with an error in case of failure.
    fn handle_workers(&self) -> Result<(), SystemError> {
        // For each worker, create a thread to handle communication
        for (i, worker) in self.workers.iter().enumerate() {
            let worker = Arc::clone(worker); // Clone the Arc to share ownership with the thread
            thread::spawn(move || {
                println!("Orchestrator handling worker {}...", i + 1);

                let mut worker = worker.lock().unwrap(); // Lock access to the worker for thread safety

                // Receive data from the worker
                match worker.receive() {
                    Ok(data) => {
                        println!(
                            "Received {} bytes from Worker {}: {:?}",
                            data.len(),
                            i + 1,
                            data
                        );

                        // Send a response to the worker
                        if let Err(e) = worker.send(b"Task received. Process completed.\n") {
                            eprintln!("Error sending response to Worker {}: {}", i + 1, e);
                        }
                    }
                    Err(e) => {
                        eprintln!("Error receiving data from Worker {}: {}", i + 1, e);
                    }
                }
            });
        }
        Ok(())
    }
}
