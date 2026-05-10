use std::io;

use comms::specs::server::ServerSpec;

/// The result of running a worker instance.
#[derive(Debug)]
pub enum Run {
    Done,
    Switch {
        server_sizes: Vec<usize>,
        server_ordering: Vec<usize>,
    },
    Upgrade {
        spec: ServerSpec,
        worker_addrs: Vec<String>,
    },
}

/// The main worker trait.
#[async_trait::async_trait]
pub trait Worker: Send {
    /// Executes the training of the model.
    ///
    /// # Returns
    /// An io error if occurred.
    async fn run(&mut self) -> io::Result<Run>;
}
