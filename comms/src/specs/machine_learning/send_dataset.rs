use super::dataset::ChunkSpec;
use crate::{
    OnoSender,
    msg::{Chunk, Msg},
};
use std::io;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite};

async fn send_chunks<W, R>(
    sender: &mut OnoSender<W>,
    disk: &mut R,
    chunk_size: usize,
    mut offset: usize,
    mut buf: Vec<u8>,
) -> io::Result<()>
where
    W: AsyncWrite + Unpin,
    R: AsyncRead + Unpin,
{
    let mut last = false;
    while !last {
        let read = disk.read(&mut buf).await?;
        if read < chunk_size {
            last = true;
        }

        let msg = Msg::Dataset(Chunk::Chunk(ChunkSpec {
            offset,
            last,
            data: &buf,
        }));

        sender.send(&msg).await?;

        offset += read;
    }

    Ok(())
}

trait SendDataset {
    async fn send_dataset<R, W>(
        sender: &mut OnoSender<W>,
        disk: &mut R,

        chunk_size: usize,
        x_size: usize,
        y_size: usize,
    ) -> io::Result<()>
    where
        W: AsyncWrite + Unpin,
        R: AsyncRead + Unpin,
    {
        let mut buf = Vec::with_capacity(chunk_size);
        let read = disk.read(&mut buf).await?;

        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "read 0 bytes from disk",
            ));
        }

        let last = read < chunk_size;
        let first = ChunkSpec {
            offset: 0,
            last,
            data: &buf,
        };

        let msg = Msg::Dataset(Chunk::Header(super::DatasetSpec {
            size: 0, // TODO
            x_size,
            y_size,
            first,
        }));

        sender.send(&msg).await?;

        if !last {
            send_chunks(sender, disk, chunk_size, read, buf).await?;
        }

        Ok(())
    }
}
