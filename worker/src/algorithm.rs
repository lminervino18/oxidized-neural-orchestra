use std::{io, net::SocketAddr};

use comms::specs::worker::AlgorithmSpec;
use tokio::net::TcpStream;

/// Connects a worker to the algorithm data plane.
pub struct AlgorithmConnector;

impl AlgorithmConnector {
    /// Connects to the training server defined by the algorithm spec.
    ///
    /// # Args
    /// * `spec` - Algorithm selection and configuration.
    ///
    /// # Returns
    /// A receiver/sender pair for data-plane messages.
    ///
    /// # Errors
    /// Returns `io::Error` when a connection fails.
    pub async fn connect(spec: &AlgorithmSpec) -> io::Result<(
        comms::OnoReceiver<tokio::net::tcp::OwnedReadHalf>,
        comms::OnoSender<tokio::net::tcp::OwnedWriteHalf>,
    )> {
        let addr = parameter_server_addr(spec);
        let stream = TcpStream::connect(addr).await?;
        let (rx, tx) = stream.into_split();
        Ok(comms::channel(rx, tx))
    }
}

fn parameter_server_addr(spec: &AlgorithmSpec) -> SocketAddr {
    match spec {
        AlgorithmSpec::ParameterServer { server_ip } => *server_ip,
    }
}
