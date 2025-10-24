use crate::orchestra::communication::Communication; // TCP communication
use crate::orchestra::error::SystemError; // Error handling
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use actix::prelude::*; // Actix framework

/// Worker: Receiver -> Producer -> Sender (read/write split)
pub struct Worker {
    comm_read: Arc<Mutex<Communication>>,
    comm_write: Arc<Mutex<Communication>>,
}

/* -------------------- Messages -------------------- */

#[derive(Message)]
#[rtype(result = "()")]
struct IncomingData(pub String); // Receiver -> Producer

#[derive(Message)]
#[rtype(result = "()")]
struct ProcessedData(pub String); // Producer -> Sender

/* -------------------- Actors -------------------- */

/// Blocking thread that listens TCP forever and forwards to Producer.
struct ReceiverActor {
    comm_read: Arc<Mutex<Communication>>,
    producer: Addr<ProducerActor>,
}

impl Actor for ReceiverActor {
    type Context = Context<Self>;

    fn started(&mut self, _ctx: &mut Self::Context) {
        println!("ReceiverActor started, spawning listening thread...");

        let comm_read = Arc::clone(&self.comm_read);
        let producer = self.producer.clone();

        thread::spawn(move || loop {
            match comm_read.lock() {
                Ok(mut comm) => match comm.receive() {
                    Ok(data) => {
                        let s = String::from_utf8_lossy(&data).to_string();
                        println!("[Receiver] Received: {}", s);
                        producer.do_send(IncomingData(s));
                    }
                    Err(e) => {
                        eprintln!(
                            "{}",
                            SystemError::CommunicationError(format!(
                                "[Receiver] Error receiving data: {}",
                                e
                            ))
                        );
                        thread::sleep(Duration::from_millis(200));
                    }
                },
                Err(_) => {
                    eprintln!(
                        "{}",
                        SystemError::WorkerError("[Receiver] Failed to lock communication".into())
                    );
                    thread::sleep(Duration::from_millis(200));
                }
            }
        });
    }
}

struct ProducerActor {
    sender: Addr<SenderActor>,
}

impl Actor for ProducerActor {
    type Context = Context<Self>;
}

impl Handler<IncomingData> for ProducerActor {
    type Result = ();

    fn handle(&mut self, msg: IncomingData, ctx: &mut Self::Context) {
        let data = msg.0;
        println!("[Producer] Received data: {}", data);

        // Simulate work (e.g., ML step)
        ctx.run_later(Duration::from_secs(1), move |act, _| {
            let result = format!("Processed data: {}", data);
            println!("[Producer] Finished processing: {}", result);
            act.sender.do_send(ProcessedData(result));
        });
    }
}

/// Sends processed results back to the Orchestrator using the write handle.
struct SenderActor {
    comm_write: Arc<Mutex<Communication>>,
}

impl Actor for SenderActor {
    type Context = Context<Self>;
}

impl Handler<ProcessedData> for SenderActor {
    type Result = ();

    fn handle(&mut self, msg: ProcessedData, _ctx: &mut Self::Context) {
        let result = msg.0;
        println!("[Sender] Sending result: {}", result);

        match self.comm_write.lock() {
            Ok(mut comm) => {
                if let Err(e) = comm.send(result.as_bytes()) {
                    eprintln!(
                        "{}",
                        SystemError::CommunicationError(format!(
                            "[Sender] Error sending data: {}",
                            e
                        ))
                    );
                }
            }
            Err(_) => {
                eprintln!(
                    "{}",
                    SystemError::WorkerError("[Sender] Failed to lock communication".into())
                );
            }
        }
    }
}

/* -------------------- Worker Implementation -------------------- */

impl Worker {
    /// Connect and create two independent handles (read, write).
    pub fn new(address: &str) -> Result<Self, SystemError> {
        loop {
            match Communication::split(address) {
                Ok((comm_r, comm_w)) => {
                    return Ok(Worker {
                        comm_read: Arc::new(Mutex::new(comm_r)),
                        comm_write: Arc::new(Mutex::new(comm_w)),
                    });
                }
                Err(e) => {
                    eprintln!("Worker failed to connect: {}. Retrying...", e);
                    std::thread::sleep(std::time::Duration::from_millis(500));
                }
            }
        }
    }

    /// Start the 3-actor pipeline and keep the runtime alive (async).
    pub async fn start(&self) -> Result<(), SystemError> {
        println!("Worker started, connecting to orchestrator...");

        let sender_addr = SenderActor {
            comm_write: Arc::clone(&self.comm_write),
        }
        .start();

        let producer_addr = ProducerActor {
            sender: sender_addr.clone(),
        }
        .start();

        let _receiver_addr = ReceiverActor {
            comm_read: Arc::clone(&self.comm_read),
            producer: producer_addr,
        }
        .start();

        // Keep the worker alive forever (or switch to ctrl_c if you prefer graceful shutdown).
        futures::future::pending::<()>().await;

        #[allow(unreachable_code)]
        Ok(())
    }
}
