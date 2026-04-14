use crate::transport::TransportLayer;

pub struct OrchHandle<T: TransportLayer> {
    transport: T,
}

impl<T> OrchHandle<T>
where
    T: TransportLayer,
{
    pub fn new(transport: T) -> Self {
        Self { transport }
    }
}
