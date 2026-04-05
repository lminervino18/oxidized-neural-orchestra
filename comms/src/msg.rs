use std::{borrow::Cow, io};

use crate::{
    Align1, Deserialize, Serialize,
    specs::{server::ServerSpec, worker::WorkerSpec},
};

pub type Header = u32;
pub const HEADER_SIZE: usize = size_of::<Header>();

#[derive(Debug)]
pub enum Payload<'a> {
    Grad(&'a [f32]),
    Params(&'a mut [f32]),
    Datachunk(&'a [f32]),
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Command<'a> {
    CreateServer(ServerSpec),
    CreateWorker(WorkerSpec),
    ReportLoss { losses: Cow<'a, [f32]> },
    Disconnect,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Detail {
    BufferSizeMismatch { expected: usize, got: usize },
    Fatal(String),
}

#[derive(Debug)]
pub enum Msg<'a> {
    Control(Command<'a>),
    Data(Payload<'a>),
    Err(Detail),
}

impl Msg<'_> {
    pub(super) fn buf_is_too_small<T>(size: usize) -> io::Result<T> {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("The given buffer is too small {size}, must at least be {HEADER_SIZE} bytes"),
        ))
    }

    pub(super) fn invalid_kind_byte<T>(byte: u8) -> io::Result<T> {
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

                    for &val in grad.iter() {
                        buf.extend_from_slice(bytemuck::bytes_of(&half::f16::from_f32(val)));
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
            2 => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "received a Grad message via recv_into; use recv_grad_or_msg instead",
            )),
            3 => Ok(Self::Data(Payload::Params(bytemuck::cast_slice_mut(rest)))),
            4 => Ok(Self::Data(Payload::Datachunk(bytemuck::cast_slice(rest)))),
            byte => Self::invalid_kind_byte(byte),
        }
    }
}
