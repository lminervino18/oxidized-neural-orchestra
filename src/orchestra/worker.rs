use crate::orchestra::communication::Communication; // Import Communication struct
use crate::orchestra::error::SystemError; // Import error handling
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// Struct representing the Worker.
///
/// The worker connects to the orchestrator and then listens for incoming data
/// in a loop. It processes the data in parallel using multiple threads:
/// - **Receiver**: Listens for incoming data from the orchestrator.
/// - **Producer**: Processes the data (simulating training of an MLP).
/// - **Sender**: Sends the results back to the orchestrator.
pub struct Worker {
    comm: Arc<Mutex<Communication>>, // Communication struct wrapped in Arc<Mutex> for concurrency
}

impl Worker {
    /// Initializes a new worker and connects it to the orchestrator.
    ///
    /// # Parameters
    /// - `address`: The address of the orchestrator to connect to.
    ///
    /// # Returns
    /// A `Result` that returns a `Worker` on success, or a `SystemError` on failure.
    pub fn new(address: &str) -> Result<Self, SystemError> {
        // Try to establish a connection with the orchestrator in a loop until successful
        loop {
            match Communication::new(address) {
                Ok(comm) => {
                    // Wrap the Communication in an Arc<Mutex> for thread-safe shared ownership
                    return Ok(Worker {
                        comm: Arc::new(Mutex::new(comm)),
                    });
                }
                Err(e) => {
                    eprintln!(
                        "Worker failed to connect to orchestrator: {}. Retrying...",
                        e
                    );
                    std::thread::sleep(std::time::Duration::from_millis(500));
                }
            }
        }
    }

    /// Starts the worker's process: connects to the orchestrator and launches threads.
    ///
    /// It creates three threads:
    /// - A receiver to listen for data from the orchestrator.
    /// - A producer to process the received data (simulating MLP training).
    /// - A sender to send the processed data back to the orchestrator.
    pub fn start(&self) -> Result<(), SystemError> {
        println!("Worker started, connecting to orchestrator...");

        // Create channels to communicate between threads
        let (tx_to_producer, rx_from_receiver): (Sender<String>, Receiver<String>) = channel();
        let (tx_to_sender, rx_from_producer): (Sender<String>, Receiver<String>) = channel();

        // Thread for receiving data from orchestrator
        let receiver_handle = self.start_receiver(tx_to_producer);

        // Thread for processing the data (simulating MLP training)
        let producer_handle = self.start_producer(tx_to_sender, rx_from_receiver);

        // Thread for sending results back to orchestrator
        let sender_handle = self.start_sender(rx_from_producer);

        // Wait for all threads to finish (join threads)
        if let Err(_) = receiver_handle.join().map(|_| ()) {
            return Err(SystemError::WorkerError(format!("Receiver thread failed")));
        }

        if let Err(_) = producer_handle.join().map(|_| ()) {
            return Err(SystemError::WorkerError(format!("Producer thread failed")));
        }

        if let Err(_) = sender_handle.join().map(|_| ()) {
            return Err(SystemError::WorkerError(format!("Sender thread failed")));
        }

        Ok(())
    }

    /// Starts the receiver thread which listens for incoming data from the orchestrator.
    ///
    /// This thread listens to the orchestrator and forwards the data to the producer.
    fn start_receiver(
        &self,
        tx_to_producer: Sender<String>,
    ) -> thread::JoinHandle<Result<(), SystemError>> {
        let comm = Arc::clone(&self.comm);
        thread::spawn(move || {
            loop {
                // Receive data from the orchestrator
                match comm.lock() {
                    Ok(mut comm_locked) => {
                        match comm_locked.receive() {
                            Ok(data) => {
                                let data_str = String::from_utf8_lossy(&data).to_string();
                                println!("Receiver received: {}", data_str);
                                // Send data to the producer thread
                                if let Err(e) = tx_to_producer.send(data_str) {
                                    return Err(SystemError::CommunicationError(format!(
                                        "Failed to send data to producer: {}",
                                        e
                                    )));
                                }
                            }
                            Err(e) => {
                                return Err(SystemError::CommunicationError(format!(
                                    "Failed to receive data: {}",
                                    e
                                )));
                            }
                        }
                    }
                    Err(_) => {
                        return Err(SystemError::WorkerError(
                            "Failed to lock communication".to_string(),
                        ));
                    }
                }
                thread::sleep(Duration::from_secs(1)); // Sleep to simulate time between messages
            }
        })
    }

    /// Starts the producer thread which processes the data received from the receiver.
    ///
    /// This thread simulates training an MLP by processing the received data and generating results.
    fn start_producer(
        &self,
        tx_to_sender: Sender<String>,
        rx_from_receiver: Receiver<String>,
    ) -> thread::JoinHandle<Result<(), SystemError>> {
        thread::spawn(move || {
            loop {
                match rx_from_receiver.recv() {
                    Ok(data) => {
                        println!("Producer received data: {}", data);
                        // Simulate processing the data (training an MLP)
                        let result = format!("Processed data: {}", data);
                        println!("Producer processed: {}", result);

                        // Send result to sender
                        if let Err(e) = tx_to_sender.send(result) {
                            return Err(SystemError::CommunicationError(format!(
                                "Failed to send result to sender: {}",
                                e
                            )));
                        }
                    }
                    Err(e) => {
                        return Err(SystemError::WorkerError(format!(
                            "Error processing data: {}",
                            e
                        )));
                    }
                }
                thread::sleep(Duration::from_secs(1)); // Simulate time taken for processing
            }
        })
    }

    /// Starts the sender thread which sends the processed data back to the orchestrator.
    ///
    /// This thread sends the results produced by the producer back to the orchestrator.
    fn start_sender(
        &self,
        rx_from_producer: Receiver<String>,
    ) -> thread::JoinHandle<Result<(), SystemError>> {
        let comm = Arc::clone(&self.comm);
        thread::spawn(move || {
            loop {
                match rx_from_producer.recv() {
                    Ok(result) => {
                        println!("Sender sending result: {}", result.as_str());
                        // Send the result to the orchestrator
                        match comm.lock() {
                            Ok(mut comm_locked) => {
                                if let Err(e) = comm_locked.send(result.as_bytes()) {
                                    return Err(SystemError::CommunicationError(format!(
                                        "Error sending data to orchestrator: {}",
                                        e
                                    )));
                                }
                            }
                            Err(_) => {
                                return Err(SystemError::WorkerError(
                                    "Failed to lock communication".to_string(),
                                ));
                            }
                        }
                    }
                    Err(e) => {
                        return Err(SystemError::WorkerError(format!(
                            "Error receiving result from producer: {}",
                            e
                        )));
                    }
                }
                thread::sleep(Duration::from_secs(1)); // Simulate time taken to send result
            }
        })
    }
}
