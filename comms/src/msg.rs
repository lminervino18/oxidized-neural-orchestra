use std::{borrow::Cow, io};

use crate::{Deserialize, Serialize};

#[derive(Debug)]
pub enum Payload<'a> {
    Gradient(Cow<'a, [f32]>),
    Weights(Cow<'a, [f32]>),
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

        buf.push(kind);
        Some(bytemuck::cast_slice(nums))
    }
}

impl<'a> Deserialize<'a> for Msg<'a> {
    fn deserialize(buf: &'a [u8]) -> io::Result<Self> {
        let kind = match buf[0] {
            0 => Payload::Gradient,
            1 => Payload::Weights,
            x => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Got an invalid message kind {x}"),
                ));
            }
        };

        let nums = bytemuck::cast_slice(&buf[1..]);
        Ok(Msg::Data(kind(Cow::Borrowed(nums))))
    }
}
