use std::{
    borrow::Cow,
    io::{self, Read, Write},
};

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

impl Deserialize for Msg<'_> {
    fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut kind_buf = [0; 1];
        reader.read_exact(&mut kind_buf)?;
        let kind = u8::from_be_bytes(kind_buf);

        let mut len_buf = [0; 8];
        reader.read_exact(&mut len_buf)?;
        let len = u64::from_be_bytes(len_buf) as usize;

        let mut nums = vec![0f32; len];
        let data_buf = bytemuck::cast_slice_mut(&mut nums);
        reader.read_exact(data_buf)?;

        let payload = match kind {
            0 => Payload::Gradient(Cow::Owned(nums)),
            1 => Payload::Weights(Cow::Owned(nums)),
            x => panic!("wrong message kind {x}"),
        };

        Ok(Msg::Data(payload))
    }
}

impl Serialize for Msg<'_> {
    fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        let (kind, nums) = match self {
            Msg::Data(Payload::Gradient(grad)) => (0, grad),
            Msg::Data(Payload::Weights(weights)) => (1, weights),
        };

        writer.write_all(&[kind])?;

        let len = nums.len() as u64;
        writer.write_all(&len.to_be_bytes())?;

        let bytes = bytemuck::cast_slice(nums);
        writer.write_all(bytes)?;

        Ok(())
    }
}
