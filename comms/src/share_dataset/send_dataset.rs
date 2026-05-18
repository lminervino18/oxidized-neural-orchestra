use std::io::{self, Cursor};

use tokio::io::AsyncRead;

use crate::{
    protocol::{Command, Msg, Payload},
    transport::TransportLayer,
    utils,
};

/// Helper function for wrapping the dataset's source buffer in a readable bytes cursor.
///
/// # Args
/// * `dataset_raw`: The `f32` dataset buffer.
///
/// # Returns
/// A `Cursor<&[u8]>` wrapping the buffer.
pub fn get_dataset_cursor(dataset_raw: &[f32]) -> Cursor<&[u8]> {
    let dataset_bytes = bytemuck::cast_slice(dataset_raw);
    Cursor::new(dataset_bytes)
}

/// Sends chunks of the dataset with `chunk_size` size.
///
/// # Args
/// * `xs` - The sample's source.
/// * `ys` - The label's source.
/// * `xs_size_hint` - The minimum amount of units that xs has.
/// * `ys_size_hint` - The minimum amount of units that ys has.
/// * `chunk_size` - The size of each chunk in bytes.
/// * `transport` - The transport layer of the communication.
///
/// # Errors
/// Returns an `io::Error` if the connection or reading from the storage fail.
pub async fn send_dataset<R, T>(
    xs: &mut R,
    ys: &mut R,
    xs_size_hint: usize,
    ys_size_hint: usize,
    chunk_size: usize,
    transport: &mut T,
) -> io::Result<()>
where
    R: AsyncRead + Unpin,
    T: TransportLayer,
{
    const UNIT_SIZE: usize = size_of::<f32>();
    let mut buf: Vec<u32> = vec![0; chunk_size / UNIT_SIZE];
    let buf_u8 = bytemuck::cast_slice_mut(&mut buf);

    send_chunks_from(xs, xs_size_hint, buf_u8, transport).await?;
    send_chunks_from(ys, ys_size_hint, buf_u8, transport).await?;

    Ok(())
}

/// Reads through the reader and sends the datachunks to the worker.
///
/// # Arsg
/// * `reader` - The byte source.
/// * `size_hint` - The minimum amount of bytes that reader has left to read.
/// * `acc` - The accumulating buffer.
/// * `transport` - The transport layer of the communication.
///
/// # Returns
/// An io error if occurred.
async fn send_chunks_from<R, T>(
    reader: &mut R,
    size_hint: usize,
    acc: &mut [u8],
    transport: &mut T,
) -> io::Result<()>
where
    R: AsyncRead + Unpin,
    T: TransportLayer,
{
    let msg = Msg::Control(Command::ShareDatasetSize { size: size_hint });
    transport.send(&msg).await?;

    while let n = utils::read_all(reader, acc).await?
        && n > 0
    {
        let nums = bytemuck::cast_slice(&acc[..n]);
        let msg = Msg::Data(Payload::Datachunk(nums));
        transport.send(&msg).await?;
    }

    let msg = Msg::Control(Command::Eof);
    transport.send(&msg).await
}
