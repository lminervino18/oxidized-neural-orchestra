use std::io;

use comms::{OrchHandle, TransportLayer};

/// The main worker trait.
pub trait Worker<T>
where
    T: TransportLayer,
{
    /// Executes the training of the model.
    ///
    /// # Args
    /// * `orch_handle` - The handle to communicate to the orchestrator.
    ///
    /// # Returns
    /// An io error if occurred.
    async fn run(self, orch_handle: OrchHandle<T>) -> io::Result<()>;
}
