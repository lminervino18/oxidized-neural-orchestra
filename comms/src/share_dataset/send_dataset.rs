use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite};

use crate::{
    OnoSender,
    msg::{Msg, Payload},
};

async fn send_dataset<R: AsyncRead + Unpin, W: AsyncWrite + Unpin>(
    dataset: &mut R,
    chunk: usize,
    sender: &mut OnoSender<W>,
) -> std::io::Result<()> {
    let mut buf = vec![0u8; chunk];

    loop {
        let read = dataset.read(&mut buf).await?;
        if read > 0 {
            let msg = Msg::Data(Payload::Datachunk(&buf[..read]));
            sender.send(&msg).await?;
        }

        if read < chunk {
            break;
        }
    }

    Ok(())
}
