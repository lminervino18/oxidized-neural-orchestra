pub enum TrainerSpec {
    BarrierSync { barrier_size: usize },
    NonBlocking,
}
