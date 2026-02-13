use std::io;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

trait SendDataset {
    async fn send_dataset<R, W>(sender: &mut W, disk: &mut R, chunk_size: usize) -> io::Result<()>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        let mut buf = Vec::with_capacity(chunk_size);

        loop {
            let read = disk.read(&mut buf).await?;
            sender.write_all(&buf[..read]).await?;

            if read < chunk_size {
                break;
            }
        }

        Ok(())
    }
}
