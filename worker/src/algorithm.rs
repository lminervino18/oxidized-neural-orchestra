use std::{io, net::SocketAddr};

use comms::specs::worker::AlgorithmSpec;
use tokio::net::TcpStream;

/// Connects a worker to the algorithm data plane.
pub struct AlgorithmConnector;

impl AlgorithmConnector {
    /// Connects to the training server(s) defined by the algorithm spec.
    ///
    /// # Args
    /// * `spec` - Algorithm selection and configuration.
    ///
    /// # Returns
    /// A receiver/sender pair for data-plane messages.
    ///
    /// # Errors
    /// Returns `io::Error` when no endpoints are available or a connection fails.
    pub async fn connect(spec: &AlgorithmSpec) -> io::Result<(
        comms::OnoReceiver<tokio::net::tcp::OwnedReadHalf>,
        comms::OnoSender<tokio::net::tcp::OwnedWriteHalf>,
    )> {
        match spec {
            AlgorithmSpec::ParameterServer { server_ips } => {
                let addr = pick_first(server_ips)?;
                let stream = TcpStream::connect(addr).await?;
                let (rx, tx) = stream.into_split();
                Ok(comms::channel(rx, tx))
            }
            AlgorithmSpec::AllReduce { .. } => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "all_reduce is not supported yet",
            )),
            AlgorithmSpec::StrategySwitch => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "strategy_switch is not supported yet",
            )),
        }
    }
}

fn pick_first(addrs: &[SocketAddr]) -> io::Result<SocketAddr> {
    addrs
        .first()
        .copied()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "no server endpoint provided"))
}
