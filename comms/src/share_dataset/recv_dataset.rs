use std::io::{Cursor, Error, ErrorKind, Result};
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt, BufWriter};

use crate::{
    OnoReceiver,
    msg::{Msg, Payload},
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
/// * `storage` - The storage for writing the chunks.
/// * `size` - The total size of the dataset in bytes.
/// * `rx` - An `OnoReceiver` for receiving the chunks.
///
/// # Errors
/// Returns an `io::Error` if the connection or writting to the storage fail.
pub async fn recv_dataset<W, R>(
    x_storage: &mut W,
    y_storage: &mut W,
    x_size: usize,
    y_size: usize,
    rx: &mut OnoReceiver<R>,
) -> Result<()>
where
    W: AsyncWrite + Unpin,
    R: AsyncRead + Unpin,
{
    // TODO: ver si conviene configurar la capacity de writer
    let mut x_writer = BufWriter::new(x_storage);
    let mut y_writer = BufWriter::new(y_storage);

    recv_dataset_into(&mut x_writer, rx, x_size).await?;
    recv_dataset_into(&mut y_writer, rx, y_size).await?;

    Ok(())
}

async fn recv_dataset_into<W, R>(
    writer: &mut BufWriter<&mut W>,
    rx: &mut OnoReceiver<R>,
    size: usize,
) -> Result<()>
where
    W: AsyncWrite + Unpin,
    R: AsyncRead + Unpin,
{
    let mut received = 0;
    while received < size {
        let msg: Msg = rx.recv().await?;

        let Msg::Data(Payload::Datachunk(chunk)) = msg else {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("expected Datachunk, got: {msg:?}"),
            ));
        };

        let bytes = bytemuck::cast_slice(chunk);

        received += bytes.len();
        writer.write_all(bytes).await?;
    }

    writer.flush().await?;

    Ok(())
}
