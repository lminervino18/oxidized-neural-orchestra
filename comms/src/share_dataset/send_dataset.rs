use std::io::{Error, ErrorKind, Result};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite};

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

    loop {
        let read = dataset.read(&mut buf).await?;

        if read > 0 {
            let msg = Msg::Data(Payload::Datachunk(&buf[..read]));
            sender.send(&msg).await?;
        }

        if read == 0 {
            break;
        }
    }

    Ok(())
}
