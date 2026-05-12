use std::io::{self, Error, ErrorKind};

use crate::{
    protocol::{Command, Msg, Payload},
    transport::TransportLayer,
};

/// Receives chunks of the dataset and writes them into a storage.
///
/// # Args
/// * `xs` - The storage for writing the sample chunks.
/// * `ys` - The storage for writing the label chunks.
/// * `transport` - The transport layer of the communication.
///
/// # Errors
/// Returns an `io::Error` if the connection or writting to the storage fail.
pub async fn recv_dataset<T>(
    xs: &mut Vec<f32>,
    ys: &mut Vec<f32>,
    transport: &mut T,
) -> io::Result<()>
where
    T: TransportLayer,
{
    recv_chunks_into(xs, transport).await?;
    recv_chunks_into(ys, transport).await?;
    Ok(())
}

/// Receives a dataset chunk through the transport layer and writes it's data into
/// the given writer.
///
/// # Args
/// * `writer` - The sink for the dataset bytes.
/// * `transport` - The transport layer of the communication.
///
/// # Returns
/// An io error if occurred.
async fn recv_chunks_into<T>(acc: &mut Vec<f32>, transport: &mut T) -> io::Result<()>
where
    T: TransportLayer,
{
    let size = match transport.recv().await? {
        Msg::Control(Command::ShareDatasetSize { size }) => size,
        msg => {
            let text = format!("expected ShareDatasetSize, got: {msg:?}");
            return Err(io::Error::other(text));
        }
    };

    let additional = size.saturating_sub(acc.capacity());
    acc.reserve(additional);

    loop {
        match transport.recv().await? {
            Msg::Data(Payload::Datachunk(chunk)) => {
                let bytes = bytemuck::cast_slice(chunk);
                acc.extend_from_slice(bytes);
            }
            Msg::Control(Command::Eof) => break,
            msg => {
                let text = format!("expected Datachunk, got: {msg:?}");
                return Err(Error::new(ErrorKind::InvalidData, text));
            }
        }
    }

    Ok(())
}
