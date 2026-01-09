use crate::{server::ParameterServer, training::Trainer};

pub struct TrainingSession<T: Trainer> {
    id: usize,
    pserver: ParameterServer<T>,
}

impl<T: Trainer> TrainingSession<T> {
    pub fn new(id: usize, pserver: ParameterServer<T>) -> Self {
        Self { id, pserver }
    }
}

impl<T: Trainer> Session for TrainingSession<T> {}

pub trait Session {}
