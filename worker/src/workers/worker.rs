use std::io;

use comms::specs::server::ServerSpec;
use machine_learning::training::Trainer;

/// The result of running a worker instance.
pub enum Run {
    Done,
    Switch {
        server_addrs: Vec<String>,
        server_sizes: Vec<usize>,
        server_ordering: Vec<usize>,
    },
    Upgrade {
        spec: ServerSpec,
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

    /// Drops self and returns it's inner trainer.
    ///
    /// # Returns
    /// A boxed `Trainer` instance.
    fn into_trainer(self: Box<Self>) -> Box<dyn Trainer>;
}
