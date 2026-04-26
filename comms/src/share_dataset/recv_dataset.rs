use std::io::{self, Cursor, Error, ErrorKind};

use tokio::io::{AsyncWrite, AsyncWriteExt};

use crate::{
    protocol::{Msg, Payload},
    transport::TransportLayer,
};

/// Helper function for wrapping the dataset's source buffer in a writeable bytes cursor.
///
/// # Args
/// * `dataset_raw`: The `f32` dataset buffer.
///
/// # Returns a `Cursor<&mut [u8]>` wrapping the buffer.
pub fn get_dataset_cursor(dataset_raw: &mut [f32]) -> Cursor<&mut [u8]> {
    let dataset_bytes: &mut [u8] = bytemuck::cast_slice_mut(dataset_raw);
    Cursor::new(dataset_bytes)
}

/// Receives chunks of the dataset and writes them into a storage.
///
/// # Args
/// * `x_storage` - The storage for writing the sample chunks.
/// * `y_storage` - The storage for writing the label chunks.
/// * `x_size` - The total size of the dataset samples in bytes.
/// * `y_size` - The total size of the dataset labels in bytes.
/// * `transport` - The transport layer of the communication.
///
/// # Errors
/// Returns an `io::Error` if the connection or writting to the storage fail.
pub async fn recv_dataset<W, T>(
    x_storage: &mut W,
    y_storage: &mut W,
    x_size: usize,
    y_size: usize,
    transport: &mut T,
) -> io::Result<()>
where
    W: AsyncWrite + Unpin,
    T: TransportLayer,
{
    recv_chunks_into(x_storage, x_size, transport).await?;
    recv_chunks_into(y_storage, y_size, transport).await?;
    Ok(())
}

/// Receives a dataset chunk through the transport layer and writes it's data into
/// the given writer.
///
/// # Args
/// * `writer` - The sink for the dataset bytes.
/// * `size` - The amount of bytes to read.
/// * `transport` - The transport layer of the communication.
///
/// # Returns
/// An io error if occurred.
async fn recv_chunks_into<W, T>(
    writer: &mut W,
    size_bytes: usize,
    transport: &mut T,
) -> io::Result<()>
where
    W: AsyncWrite + Unpin,
    T: TransportLayer,
{
    let mut received = 0;

    while received < size_bytes {
        let msg: Msg = transport.recv().await?;

        let Msg::Data(Payload::Datachunk(chunk)) = msg else {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("expected Datachunk, got: {msg:?}"),
            ));
        };

        let bytes = bytemuck::cast_slice(chunk);
        writer.write_all(bytes).await?;
        received += bytes.len();
    }

    writer.flush().await
}
