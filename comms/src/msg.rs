use std::{
    borrow::Cow,
    io::{self, Read, Write},
};

use crate::{Deserialize, Serialize};

pub enum Payload<'a> {
    Gradient(Cow<'a, [f32]>),
    Weights(Cow<'a, [f32]>),
}

pub enum ControlMsg {}

pub enum Msg<'a> {
    Data(Payload<'a>),
    Control(ControlMsg),
}

impl Deserialize for Msg<'_> {
    fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        todo!()
    }
}

impl Serialize for Msg<'_> {
    fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        match self {
            Msg::Data(Payload::Gradient(grad)) => {
                writer.write_all(&[0])?;

                let len = grad.len() as u64;
                writer.write_all(&len.to_be_bytes())?;

                let bytes = bytemuck::cast_slice(grad);
                writer.write_all(&bytes)?;
            }
            Msg::Data(Payload::Weights(weights)) => {
                writer.write(&[1])?;
            }
            Msg::Control(control_msg) => todo!(),
        }

        Ok(())
    }
}
