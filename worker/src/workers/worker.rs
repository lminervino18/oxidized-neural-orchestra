use std::io;

/// The main worker trait.
#[async_trait::async_trait]
pub trait Worker: Send {
    /// Executes the training of the model.
    ///
    /// # Returns
    /// An io error if occurred.
    async fn run(&mut self) -> io::Result<()>;
}
