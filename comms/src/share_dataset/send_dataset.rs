use std::io::Result;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, BufReader};

use crate::{
    OnoSender,
    msg::{Msg, Payload},
};

pub async fn send_dataset<R, W>(
    dataset: &mut R,
    chunk: usize,
    sender: &mut OnoSender<W>,
) -> Result<()>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let mut buf = vec![0u8; chunk];
    let mut reader = BufReader::new(dataset);

    loop {
        let read = reader.read(&mut buf).await?;

        if read > 0 {
            let nums = bytemuck::cast_slice(&buf);
            let msg = Msg::Data(Payload::Datachunk(nums));

            sender.send(&msg).await?;
        }

        if read == 0 {
            break;
        }
    }

    Ok(())
}
