use std::{borrow::Cow, io};

use crate::{Deserialize, Serialize, specs::server::ServerSpec};

type Header = u32;
const HEADER_SIZE: usize = size_of::<Header>();

/// The payload data for the `Data` variant of the `Msg` enum.
#[derive(Debug)]
pub enum Payload<'a> {
    Gradient(&'a [f32]),
    Weights(&'a mut [f32]),
}

/// The command for the `Control` variant of the `Msg` enum.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Command {
    Connect,
    CreateServer(ServerSpec),
    Disconnect,
}

/// The application layer message for the entire system.
#[derive(Debug)]
pub enum Msg<'a> {
    Control(Command),
    Data(Payload<'a>),
    Err(Cow<'a, str>),
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
            Msg::Err(e) => {
                let header = (0 as Header).to_be_bytes();
                buf.extend_from_slice(&header);
                Some(e.as_bytes())
            }
            Msg::Control(cmd) => {
                let header = (1 as Header).to_be_bytes();
                buf.extend_from_slice(&header);

                // SAFETY: Serialize impl for `Command` is derived and not implemented
                //         by hand. Nor has a non string-key map inside.
                serde_json::to_writer(buf, &cmd).unwrap();
                None
            }
            Msg::Data(payload) => {
                let (kind, nums) = match payload {
                    Payload::Gradient(grad) => (2, grad.as_ref()),
                    Payload::Weights(weights) => (3, weights.as_ref()),
                };

                let header = (kind as Header).to_be_bytes();
                buf.extend_from_slice(&header);
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

        let (kind_buf, rest) = buf.split_at_mut(HEADER_SIZE);

        // SAFETY: We splitted the buffer to be of size `HEADER_SIZE` just above.
        let kind = Header::from_be_bytes(kind_buf.try_into().unwrap()) as u8;

        match kind {
            0 => {
                let string = str::from_utf8(rest)
                    .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;

                Ok(Self::Err(Cow::Borrowed(string)))
            }
            1 => {
                let cmd = serde_json::from_slice(rest)?;
                Ok(Self::Control(cmd))
            }
            2..4 => {
                let nums = bytemuck::cast_slice_mut(rest);

                let payload = match kind {
                    2 => Payload::Gradient(nums),
                    3 => Payload::Weights(nums),
                    _ => unreachable!(),
                };

                Ok(Self::Data(payload))
            }
            byte => Self::invalid_kind_byte(byte),
        }
    }
}
