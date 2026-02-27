use std::io::{Error, ErrorKind, Result};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite};

use crate::{
    OnoSender,
    msg::{Msg, Payload},
};

fn invalid_data() -> Result<()> {
    Err(Error::new(ErrorKind::InvalidData, "invalid data"))
}

pub async fn send_dataset<R, W>(
    dataset: &mut R,
    chunk: usize,
    sender: &mut OnoSender<W>,
) -> Result<()>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    if chunk % 4 != 0 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "chunk size should be a multiple a 4",
        ));
    }

    let mut buf = vec![0f32; chunk / 4];

    loop {
        let bytes_buf = bytemuck::cast_slice_mut(&mut buf);
        let read = dataset.read(bytes_buf).await?;

        if read > 0 {
            let floats = read / 4;
            let msg = Msg::Data(Payload::Datachunk(&buf[..floats]));
            sender.send(&msg).await?;
        }

        if read < chunk {
            break;
        }
    }

    Ok(())
}
