use std::io;

use comms::{TransportLayer, WorkerHandle};

pub struct WorkerRingManager<T>
where
    T: TransportLayer,
{
    prev: WorkerHandle<T>,
    next: WorkerHandle<T>,
}

impl<T> WorkerRingManager<T>
where
    T: TransportLayer,
{
    pub async fn gather(&mut self) -> io::Result<&[f32]> {
        todo!()
    }

    pub async fn scatter(&mut self) -> io::Result<()> {
        todo!()
    }

    pub async fn disconnect(&mut self) -> io::Result<()> {
        todo!()
    }
}
