use std::io::{Error, ErrorKind, Result};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite};

use crate::{
    OnoSender,
    msg::{Msg, Payload},
    specs::machine_learning::DatasetSpec,
};

pub async fn send_dataset<R, W>(
    dataset: &mut R,
    spec: DatasetSpec,
    sender: &mut OnoSender<W>,
) -> Result<()>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let mut buf = vec![0u8; spec.chunk];

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
