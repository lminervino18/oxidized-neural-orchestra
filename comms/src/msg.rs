use std::{borrow::Cow, io};

use super::{
    Align1, Deserialize,
    specs::{server::ServerSpec, worker::WorkerSpec},
};

pub type Header = u32;
const HEADER_SIZE: usize = size_of::<Header>();

/// The payload data for the `Data` variant of the `Msg` enum.
#[derive(Debug)]
pub enum Payload<'a> {
    Grad(&'a mut [f32]),
    Params(&'a mut [f32]),
    Datachunk(&'a [f32]),
}

/// The command for the `Control` variant of the `Msg` enum.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Command<'a> {
    CreateServer(ServerSpec),
    CreateWorker(WorkerSpec),
    ReportLoss { losses: Cow<'a, [f32]> },
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
    Control(Command<'a>),
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

    fn invalid_kind_byte<T>(byte: u8) -> io::Result<T> {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Received an invalid kind byte {byte}"),
        ))
    }
}

impl<'a> Deserialize<'a> for Msg<'a> {
    fn deserialize<B: Align1>(data: &'a mut [B]) -> io::Result<Self> {
        let bytes = bytemuck::cast_slice_mut(data);

        if bytes.len() < HEADER_SIZE {
            return Self::buf_is_too_small(bytes.len());
        }

        let (kind_buf, rest) = bytes.split_at_mut(HEADER_SIZE);

        // SAFETY: We splitted the buffer to be of size `HEADER_SIZE` just above.
        let kind = Header::from_be_bytes(kind_buf.try_into().unwrap()) as u8;

        match kind {
            0 => Ok(Self::Err(serde_json::from_slice(rest)?)),
            1 => Ok(Self::Control(serde_json::from_slice(rest)?)),
            2 => {
                todo!("Implement sparse deserialization")
            }
            3..6 => {
                let nums = bytemuck::cast_slice_mut(rest);

                let payload = match kind {
                    3 => Payload::Grad(nums),
                    4 => Payload::Params(nums),
                    5 => Payload::Datachunk(nums),
                    _ => unreachable!(),
                };

                Ok(Self::Data(payload))
            }
            byte => Self::invalid_kind_byte(byte),
        }
    }
}
