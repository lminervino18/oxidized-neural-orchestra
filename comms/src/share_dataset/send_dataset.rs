use std::io;

use tokio::io::AsyncRead;

use crate::{
    protocol::{Msg, Payload},
    transport::TransportLayer,
    utils,
};

/// Sends chunks of the dataset with `chunk` size.
///
/// # Args
/// * `xs` - The sample's source.
/// * `ys` - The label's source.
/// * `chunk_size` - The size of each chunk.
/// * `transport` - The transport layer of the communication.
///
/// # Errors
/// Returns an `io::Error` if the connection or reading from the storage fail.
pub async fn send_dataset<R, T>(
    xs: &mut R,
    ys: &mut R,
    chunk_size: usize,
    transport: &mut T,
) -> io::Result<()>
where
    R: AsyncRead + Unpin,
    T: TransportLayer,
{
    let mut buf = vec![0; chunk_size];
    send_chunks_from(xs, &mut buf, transport).await?;
    send_chunks_from(ys, &mut buf, transport).await?;
    Ok(())
}

/// Reads through the reader and sends the datachunks to the worker.
///
/// # Arsg
/// * `reader` - The byte source.
/// * `acc` - The accumulating buffer.
/// * `transport` - The transport layer of the communication.
///
/// # Returns
/// An io error if occurred.
async fn send_chunks_from<R, T>(reader: &mut R, acc: &mut [u8], transport: &mut T) -> io::Result<()>
where
    R: AsyncRead + Unpin,
    T: TransportLayer,
{
    while utils::read_all(reader, acc).await? > 0 {
        let nums = bytemuck::cast_slice(&acc);
        let msg = Msg::Data(Payload::Datachunk(nums));
        transport.send(&msg).await?;
    }

    Ok(())
}
