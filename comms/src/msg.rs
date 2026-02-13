use std::io;

use crate::{
    Deserialize, Serialize,
};

use super::protocol::*;

/// The payload data for the `Data` variant of the `Msg` enum.
#[derive(Debug)]
pub enum Payload<'a> {
    Grad(&'a [f32]),
    Params(&'a mut [f32]),
}

/// The command for the `Control` variant of the `Msg` enum.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Command {
    CreateServer(ServerSpec),
    CreateWorker(WorkerSpec),

    /// Reports a sequence of losses computed by the worker over its partial dataset
    /// after completing an epoch (i.e., one full pass over that partial dataset).
    ReportLoss {
        worker_id: usize,
        losses: Vec<f32>,
    },

    Disconnect,
}

/// The errors for the `Err` variant of the `Msg` enum.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Detail {
    BufferSizeMismatch { expected: usize, got: usize },
    Fatal(String),
}

/// The application layer message for the entire system.
#[derive(Debug)]
pub enum Msg<'a> {
    Control(Command),
    Data(Payload<'a>),
    Err(Detail),
}

impl Msg<'_> {
    fn buf_is_too_small<T>(size: usize) -> io::Result<T> {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("The given buffer is too small {size}, must at least be {HEADER_SIZE} bytes"),
        ))
    }

    fn invalid_header<T>(bytes: Header) -> io::Result<T> {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Received invalid header kind bytes {bytes:?}"),
        ))
    }
}

impl<'a> Serialize<'a> for Msg<'a> {
    fn serialize(&'a self, buf: &mut Vec<u8>) -> Option<&'a [u8]> {
        match self {
            Msg::Err(detail) => {
                buf.extend_from_slice(&ERR);

                // SAFETY: Serialize impl for `Detail` is derived and not implemented
                //         by hand. Nor has a non string-key map inside.
                serde_json::to_writer(buf, &detail).unwrap();
                None
            }
            Msg::Control(cmd) => {
                buf.extend_from_slice(&CONTROL);

                // SAFETY: Serialize impl for `Command` is derived and not implemented
                //         by hand. Nor has a non string-key map inside.
                serde_json::to_writer(buf, &cmd).unwrap();
                None
            }
            Msg::Data(payload) => {
                let (header, nums): (_, &[_]) = match payload {
                    Payload::Grad(grad) => (&GRAD, grad),
                    Payload::Params(params) => (&PARAMS, params),
                    _ => todo!(),
                };

                buf.extend_from_slice(header);
                Some(bytemuck::cast_slice(nums))
            }
        }
    }
}

impl<'a> Deserialize<'a> for Msg<'a> {
    fn deserialize(buf: &'a mut [u8]) -> io::Result<Self> {
        if buf.len() < HEADER_SIZE {
            return Self::buf_is_too_small(buf.len());
        }

        let (header, rest) = buf.split_at_mut(HEADER_SIZE);

        // SAFETY: We splitted the buffer to be of size `HEADER_SIZE` just above.
        let header: Header = header.try_into().unwrap();

        match header {
            ERR => {
                let detail = serde_json::from_slice(rest)?;
                Ok(Self::Err(detail))
            }
            CONTROL => {
                let cmd = serde_json::from_slice(rest)?;
                Ok(Self::Control(cmd))
            }
            GRAD | PARAMS => {
                let nums = bytemuck::cast_slice_mut(rest);

                let payload = match header {
                    GRAD => Payload::Grad(nums),
                    PARAMS => Payload::Params(nums),
                    _ => unreachable!(),
                };

                Ok(Self::Data(payload))
            }
            bytes => Self::invalid_header(bytes),
        }
    }
}
