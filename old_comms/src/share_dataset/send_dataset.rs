use std::io::Result;

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, BufReader};

use crate::{
    OnoSender,
    msg::{Msg, Payload},
};

/// Sends chunks of the dataset with `chunk` size.
///
/// # Args
/// * `x_storage` - The sample's source.
/// * `y_storage` - The label's source.
/// * `chunk_size` - The size of each chunk.
/// * `tx` - An `OnoSender` for sending the chunks.
///
/// # Errors
/// Returns an `io::Error` if the connection or reading from the storage fail.
pub async fn send_dataset<R, W>(
    x_storage: &mut R,
    y_storage: &mut R,
    chunk_size: usize,
    tx: &mut OnoSender<W>,
) -> Result<()>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let mut buf = vec![0; chunk_size];
    let mut x_reader = BufReader::new(x_storage);
    let mut y_reader = BufReader::new(y_storage);

    send_chunks_from(&mut x_reader, &mut buf, tx).await?;
    send_chunks_from(&mut y_reader, &mut buf, tx).await?;

    Ok(())
}

async fn send_chunks_from<R, W>(
    reader: &mut BufReader<&mut R>,
    buf: &mut [u8],
    tx: &mut OnoSender<W>,
) -> Result<()>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    loop {
        let read = reader.read(buf).await?;

        if read > 0 {
            let nums = bytemuck::cast_slice(&buf[..read]);
            let msg = Msg::Data(Payload::Datachunk(nums));

            tx.send(&msg).await?;
        }

        if read == 0 {
            break;
        }
    }

    Ok(())
}
