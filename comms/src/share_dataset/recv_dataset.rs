use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt};

use crate::{
    OnoReceiver,
    msg::{Msg, Payload},
    specs::machine_learning::DatasetSpec,
};

pub async fn recv_dataset<W, R>(
    storage: &mut W,
    spec: DatasetSpec,
    receiver: &mut OnoReceiver<R>,
) -> std::io::Result<()>
where
    W: AsyncWrite + Unpin,
    R: AsyncRead + Unpin,
{
    let mut received = 0;

    while received < spec.size {
        let msg: Msg = receiver.recv().await?;

        let chunk = match msg {
            Msg::Data(Payload::Datachunk(chunk)) => chunk,
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("expected Datachunk, got: {msg:?}"),
                ));
            }
        };

        let byte_chunk = bytemuck::cast_slice(chunk);

        storage.write_all(byte_chunk).await?;
        received += byte_chunk.len();
    }

    Ok(())
}
