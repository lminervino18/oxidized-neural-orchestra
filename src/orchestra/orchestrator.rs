use crate::orchestra::communication::Communication;
use crate::orchestra::error::SystemError;
use actix::prelude::*;
use actix::System;
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/* -------------------------------------------------------------------------- */
/*                               Actix messages                               */
/* -------------------------------------------------------------------------- */

#[derive(Message)]
#[rtype(result = "()")]
struct TaskMessage(pub String); // orchestrator -> worker

#[derive(Message)]
#[rtype(result = "()")]
struct WorkerResult {
    pub worker_id: usize,
    pub data: String,
} // worker -> orchestrator

#[derive(Message)]
#[rtype(result = "()")]
struct SetWorkers(pub Vec<Addr<WorkerHandlerActor>>); // inject worker addrs

#[derive(Message)]
#[rtype(result = "()")]
struct Kickoff; // orchestrator self-message

#[derive(Message)]
#[rtype(result = "()")]
struct AllDone; // orchestrator self-message

/* -------------------------------------------------------------------------- */
/*                             Worker handler actor                            */
/* -------------------------------------------------------------------------- */

/// One actor per TCP worker. Uses separate read/write handles to avoid lock contention.
struct WorkerHandlerActor {
    id: usize,
    comm_read: Arc<Mutex<Communication>>,
    comm_write: Arc<Mutex<Communication>>,
    orchestrator: Addr<OrchestratorActor>,
}

impl Actor for WorkerHandlerActor {
    type Context = Context<Self>;

    fn started(&mut self, _ctx: &mut Self::Context) {
        println!("WorkerHandlerActor {} started.", self.id);

        // Blocking reader loop on a dedicated OS thread (simple and reliable).
        let comm_read = Arc::clone(&self.comm_read);
        let orchestrator = self.orchestrator.clone();
        let worker_id = self.id;

        thread::spawn(move || loop {
            match comm_read.lock() {
                Ok(mut guard) => match guard.receive() {
                    Ok(bytes) => {
                        let msg = String::from_utf8_lossy(&bytes).to_string();
                        orchestrator.do_send(WorkerResult { worker_id, data: msg });
                    }
                    Err(e) => {
                        eprintln!("Worker {} receive error: {}", worker_id, e);
                        thread::sleep(Duration::from_secs(1));
                    }
                },
                Err(_) => {
                    eprintln!("Worker {}: communication lock failed", worker_id);
                    thread::sleep(Duration::from_millis(500));
                }
            }
        });
    }
}

impl Handler<TaskMessage> for WorkerHandlerActor {
    type Result = ();

    fn handle(&mut self, msg: TaskMessage, _ctx: &mut Self::Context) {
        let data = msg.0;
        println!("Worker {} sending task: {}", self.id, data);

        if let Ok(mut guard) = self.comm_write.lock() {
            if let Err(e) = guard.send(data.as_bytes()) {
                eprintln!("Failed to send task to Worker {}: {}", self.id, e);
            }
        } else {
            eprintln!("Worker {}: communication lock failed", self.id);
        }
    }
}

/* -------------------------------------------------------------------------- */
/*                            Orchestrator main actor                          */
/* -------------------------------------------------------------------------- */

pub struct OrchestratorActor {
    workers: Vec<Addr<WorkerHandlerActor>>,
    next_task_id: usize,

    // Simple round coordination
    expected: usize,
    received: usize,
    buffer: Vec<(usize, String)>,
}

impl OrchestratorActor {
    fn send_batch(&mut self) {
        self.received = 0;
        self.buffer.clear();
        self.expected = self.workers.len();

        println!(
            "Orchestrator: sending Task {} to {} workers",
            self.next_task_id, self.expected
        );

        for (i, w) in self.workers.iter().enumerate() {
            let payload = format!("task {} for worker {}", self.next_task_id, i + 1);
            w.do_send(TaskMessage(payload));
        }
    }
}

impl Actor for OrchestratorActor {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        println!("OrchestratorActor started.");
        // Keep actor alive (no-op heartbeat).
        ctx.run_interval(Duration::from_secs(3600), |_, _| {});
    }
}

impl Handler<SetWorkers> for OrchestratorActor {
    type Result = ();

    fn handle(&mut self, msg: SetWorkers, ctx: &mut Self::Context) {
        self.workers = msg.0;
        println!("Orchestrator: registered {} workers", self.workers.len());

        // Give workers a brief moment to enter `started()` before first batch.
        ctx.run_later(Duration::from_millis(100), |_, ctx| {
            ctx.address().do_send(Kickoff);
        });
    }
}

impl Handler<Kickoff> for OrchestratorActor {
    type Result = ();
    fn handle(&mut self, _msg: Kickoff, _ctx: &mut Self::Context) {
        self.send_batch();
    }
}

impl Handler<WorkerResult> for OrchestratorActor {
    type Result = ();

    fn handle(&mut self, msg: WorkerResult, ctx: &mut Self::Context) {
        self.buffer.push((msg.worker_id, msg.data));
        self.received += 1;

        println!("Orchestrator: received {}/{}", self.received, self.expected);

        if self.received >= self.expected {
            ctx.address().do_send(AllDone);
        }
    }
}

impl Handler<AllDone> for OrchestratorActor {
    type Result = ();

    fn handle(&mut self, _msg: AllDone, ctx: &mut Self::Context) {
        // Simulate processing then launch next round.
        ctx.run_later(Duration::from_millis(300), |act, ctx| {
            println!("Orchestrator: processing {} responses...", act.buffer.len());
            for (wid, data) in &act.buffer {
                println!("  - from worker {} -> {}", wid, data);
            }
            act.next_task_id += 1;
            ctx.address().do_send(Kickoff);
        });
    }
}

/* -------------------------------------------------------------------------- */
/*                        Public orchestrator interface                        */
/* -------------------------------------------------------------------------- */

pub struct Orchestrator {
    waiting_for: usize,
}

impl Orchestrator {
    pub fn new(waiting_for: usize) -> Self {
        Orchestrator { waiting_for }
    }

    /// Sync entrypoint: accepts N workers, creates actors, runs forever.
    pub fn start(&mut self, address: &str) -> Result<(), SystemError> {
        println!(
            "Orchestrator started, waiting for {} workers...",
            self.waiting_for
        );

        // Accept raw TCP connections (blocking) and split them into (read, write).
        let listener = TcpListener::bind(address).map_err(|e| {
            SystemError::ConnectionError(format!("Failed to bind to {}: {}", address, e))
        })?;
        println!("Listening on {}", address);

        let mut accepted: Vec<(Arc<Mutex<Communication>>, Arc<Mutex<Communication>>)> = Vec::new();

        while accepted.len() < self.waiting_for {
            match listener.accept() {
                Ok((stream, peer)) => {
                    println!("Worker connected from {}", peer);
                    let (comm_read, comm_write) = Communication::split_from_stream(stream)?;
                    accepted.push((
                        Arc::new(Mutex::new(comm_read)),
                        Arc::new(Mutex::new(comm_write)),
                    ));
                    println!("Workers connected: {}/{}", accepted.len(), self.waiting_for);
                }
                Err(e) => eprintln!("Accept error: {}", e),
            }
        }

        println!("All workers connected. Launching actor system...");

        // Create an Actix system and keep it alive indefinitely.
        System::new().block_on(async move {
            // Start orchestrator (workers will be injected).
            let orchestrator = OrchestratorActor {
                workers: Vec::new(),
                next_task_id: 0,
                expected: 0,
                received: 0,
                buffer: Vec::new(),
            }
            .start();

            // Create WorkerHandlerActor per connection with separate read/write comms.
            let mut worker_addrs = Vec::with_capacity(accepted.len());
            for (i, (comm_read, comm_write)) in accepted.into_iter().enumerate() {
                let addr = WorkerHandlerActor {
                    id: i + 1,
                    comm_read,
                    comm_write,
                    orchestrator: orchestrator.clone(),
                }
                .start();
                worker_addrs.push(addr);
            }

            // Inject workers and start loop (Kickoff → wait → AllDone → Kickoff → ...).
            orchestrator.do_send(SetWorkers(worker_addrs));

            println!("Actix orchestrator system is running indefinitely.");
            futures::future::pending::<()>().await;
        });

        Ok(())
    }
}
