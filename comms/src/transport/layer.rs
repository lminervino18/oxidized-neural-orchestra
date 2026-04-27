use std::io;

use crate::protocol::Msg;

/// The trait that the different transport layers should implement
/// following a decorator pattern to easily add capabilities to the
/// transport.
#[allow(unused)]
#[trait_variant::make(TransportLayer: Send)]
pub trait TransportLayerTemplate {
    /// Receives a message from the inner layer.
    ///
    /// # Returns
    /// The deserialized message or an io error if occurred.
    async fn recv(&mut self) -> io::Result<Msg<'_>>;

    /// Sends a message through the inner layer.
    ///
    /// # Args
    /// * `msg` - The message to be sent.
    ///
    /// # Returns
    /// An io error if occurred.
    async fn send<'a>(&mut self, msg: &Msg<'a>) -> io::Result<()>;
}
