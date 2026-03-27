use machine_learning::training::Trainer;

/// The local worker state.
pub struct Worker {
    trainer: Box<dyn Trainer>,
}

impl Worker {
    /// Creates a new `Worker`.
    ///
    /// # Args
    /// * `trainer` - The local trainer.
    ///
    /// # Returns
    /// A new `Worker`.
    pub fn new(trainer: Box<dyn Trainer>) -> Self {
        Self { trainer }
    }

    /// Consumes the worker and returns its trainer.
    ///
    /// # Returns
    /// The owned trainer.
    pub(crate) fn into_trainer(self) -> Box<dyn Trainer> {
        self.trainer
    }
}
