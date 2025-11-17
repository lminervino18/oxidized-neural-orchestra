mod protocol;
mod serialization;
mod sink;
mod source;

use std::{io, net::TcpStream};

use serialization::{Deserialize, Serialize};
use sink::Sink;
use source::Source;

/// Creates a `Sink` and `Source` network channel.
///
/// Given a connection stream will create and return both ends of the communication.
pub fn channel<T>(id: usize, stream: TcpStream) -> io::Result<(Sink<T>, Source<T>)>
where
    T: Serialize<Vec<u8>> + Deserialize + Send + 'static,
{
    let sink = Sink::new(id, stream.try_clone()?);
    let source = Source::new(id, stream)?;
    Ok((sink, source))
}
