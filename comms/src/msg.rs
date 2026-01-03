use std::io;

use crate::{Deserialize, Serialize};

type Header = u32;
const HEADER_SIZE: usize = size_of::<Header>();

#[derive(Debug)]
pub enum Payload<'a> {
    Gradient(&'a [f32]),
    Weights(&'a [f32]),
}

#[derive(Debug)]
pub enum Msg<'a> {
    Data(Payload<'a>),
}

impl<'a> Serialize<'a> for Msg<'a> {
    fn serialize(&'a self, buf: &mut Vec<u8>) -> Option<&'a [u8]> {
        let (kind, nums) = match self {
            Msg::Data(Payload::Gradient(grad)) => (0, grad),
            Msg::Data(Payload::Weights(weights)) => (1, weights),
        };

        let header = (kind as Header).to_be_bytes();
        buf.extend_from_slice(&header);

        Some(bytemuck::cast_slice(nums))
    }
}

impl<'a> Deserialize<'a> for Msg<'a> {
    fn deserialize(buf: &'a [u8]) -> io::Result<Self> {
        if buf.len() < HEADER_SIZE {
            return Self::buf_is_too_small(buf.len());
        }

        let (kind_buf, rest) = buf.split_at(HEADER_SIZE);

        // SAFETY: We splitted the buffer to be of size `HEADER_SIZE` just above.
        let kind = Header::from_be_bytes(kind_buf.try_into().unwrap()) as u8;
        let nums = bytemuck::cast_slice(rest);

        let payload = match kind {
            0 => Payload::Gradient(nums),
            1 => Payload::Weights(nums),
            byte => return Self::invalid_kind_byte(byte),
        };

        Ok(Self::Data(payload))
    }
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
