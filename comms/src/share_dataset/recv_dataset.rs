use std::io::{Error, ErrorKind, Result};
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt, BufWriter};

use crate::{
    OnoReceiver,
    msg::{Msg, Payload},
    specs::machine_learning::DatasetSpec,
};

pub async fn recv_dataset<W, R>(
    storage: &mut W,
    spec: DatasetSpec,
    receiver: &mut OnoReceiver<R>,
) -> Result<()>
where
    W: AsyncWrite + Unpin,
    R: AsyncRead + Unpin,
{
    let mut buf = Vec::<u32>::new();
    // TODO: ver si conviene configurar la capacity de writer
    let mut writer = BufWriter::new(storage);

    let mut received = 0;

    while (received as u64) < spec.size {
        let msg: Msg = receiver.recv_into(&mut buf).await?;

        let chunk = match msg {
            Msg::Data(Payload::Datachunk(chunk)) => chunk,
            _ => {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!("expected Datachunk, got: {msg:?}"),
                ));
            }
        };

        received += chunk.len();
        writer.write_all(chunk).await?;
    }

    writer.flush().await?;

    Ok(())
}
