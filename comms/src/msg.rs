use std::{borrow::Cow, io};

use crate::{
    Align1, Deserialize, Serialize,
    specs::{server::ServerSpec, worker::WorkerSpec},
};

type Header = u32;
const HEADER_SIZE: usize = size_of::<Header>();

/// The payload data for the `Data` variant of the `Msg` enum.
///
/// `Grad` uses `Cow<'a, [f32]>` to support both the sending path
/// (`Borrowed` — the caller owns the f32 buffer) and the receiving path
/// (`Owned` — converted from the wire's f16 representation).
/// `Params` and `Datachunk` remain as raw `f32` slices.
#[derive(Debug)]
pub enum Payload<'a> {
    Grad(Cow<'a, [f32]>),
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

impl<'a> Serialize<'a> for Msg<'a> {
    fn serialize(&'a self, buf: &mut Vec<u8>) -> Option<&'a [u8]> {
        match self {
            Msg::Err(detail) => {
                let header = (0 as Header).to_be_bytes();
                buf.extend_from_slice(&header);
                serde_json::to_writer(buf, &detail).unwrap();
                None
            }
            Msg::Control(cmd) => {
                let header = (1 as Header).to_be_bytes();
                buf.extend_from_slice(&header);
                serde_json::to_writer(buf, &cmd).unwrap();
                None
            }
            Msg::Data(payload) => match payload {
                Payload::Grad(grad) => {
                    let header = (2 as Header).to_be_bytes();
                    buf.extend_from_slice(&header);
                    for &val in grad.as_ref() {
                        buf.extend_from_slice(&half::f16::from_f32(val).to_le_bytes());
                    }
                    None
                }
                Payload::Params(params) => {
                    let header = (3 as Header).to_be_bytes();
                    buf.extend_from_slice(&header);
                    Some(bytemuck::cast_slice(params))
                }
                Payload::Datachunk(chunk) => {
                    let header = (4 as Header).to_be_bytes();
                    buf.extend_from_slice(&header);
                    Some(bytemuck::cast_slice(chunk))
                }
            },
        }
    }
}

impl<'a> Deserialize<'a> for Msg<'a> {
    fn deserialize<B: Align1>(buf: &'a mut [B]) -> io::Result<Self> {
        let bytes = bytemuck::cast_slice_mut(buf);

        if bytes.len() < HEADER_SIZE {
            return Self::buf_is_too_small(bytes.len());
        }

        let (kind_buf, rest) = bytes.split_at_mut(HEADER_SIZE);
        let kind = Header::from_be_bytes(kind_buf.try_into().unwrap()) as u8;

        match kind {
            0 => Ok(Self::Err(serde_json::from_slice(rest)?)),
            1 => Ok(Self::Control(serde_json::from_slice(rest)?)),
            2 => {
                if rest.len() % 2 != 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "Grad payload has odd byte count {}, expected even (f16 pairs)",
                            rest.len()
                        ),
                    ));
                }
                let grad: Vec<f32> = rest
                    .chunks_exact(2)
                    .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
                    .collect();
                Ok(Self::Data(Payload::Grad(Cow::Owned(grad))))
            }
            3 => Ok(Self::Data(Payload::Params(bytemuck::cast_slice_mut(rest)))),
            4 => Ok(Self::Data(Payload::Datachunk(bytemuck::cast_slice_mut(rest)))),
            byte => Self::invalid_kind_byte(byte),
        }
    }
}