use std::io::Result;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, BufReader};

use crate::{
    OnoSender,
    msg::{Msg, Payload},
};

/// Sends chunks of the dataset with `chunk` size.
///
/// # Args
/// * `dataset` - The dataset's source.
/// * `chunk` - The size of each chunk.
/// * `tx` - An `OnoSender` for sending the chunks.
///
/// # Errors
/// Returns an `io::Error` if the connection or reading from the storage fail.
pub async fn send_dataset<R, W>(
    storage: &mut R,
    chunk_size: usize,
    tx: &mut OnoSender<W>,
) -> Result<()>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let mut buf = vec![0; chunk_size];
    let mut reader = BufReader::new(storage);

    loop {
        let read = reader.read(&mut buf).await?;

        if read > 0 {
            let nums = bytemuck::cast_slice(&buf[..read]);
            let msg = Msg::Data(Payload::Datachunk(nums));
            tx.send(msg).await?;
        }

        if read == 0 {
            break;
        }
    }

    Ok(())
}
