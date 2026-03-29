use std::io;

use super::{
    msg::{HEADER_SIZE, Header, Msg, Payload},
    sparse,
};

pub struct Deserializer;

impl Deserializer {
    pub fn new() -> Self {
        Self
    }

    /// Deserializes the given bytes and creates a new Self.
    ///
    /// # Args
    /// * `data` - A byte array.
    /// * `nums` - A number array, it's used to write an incoming sparse gradient.
    ///
    /// # Returns
    /// A result object that returns `Self` on success or `io::Error` on failure.
    pub fn deserialize<'a>(
        &mut self,
        data: &'a mut [u8],
        nums: Option<&'a mut [f32]>,
    ) -> io::Result<Msg<'a>> {
        if data.len() < HEADER_SIZE {
            return Msg::buf_is_too_small(data.len());
        }

        let (kind_buf, rest) = data.split_at_mut(HEADER_SIZE);

        // SAFETY: We splitted the buffer to be of size `HEADER_SIZE` just above.
        let kind = Header::from_be_bytes(kind_buf.try_into().unwrap()) as u8;

        match kind {
            0 => Ok(Msg::Err(serde_json::from_slice(rest)?)),
            1 => Ok(Msg::Control(serde_json::from_slice(rest)?)),
            2 => {
                let Some(grad) = nums else {
                    return Err(io::Error::other(
                        "didn't provide a number array, failed to lift sparse grad",
                    ));
                };

                sparse::grad_lift_into(grad, rest).map_err(io::Error::other)?;
                Ok(Msg::Data(Payload::Grad(grad)))
            }
            3..6 => {
                let nums = bytemuck::cast_slice_mut(rest);

                let payload = match kind {
                    3 => Payload::Grad(nums),
                    4 => Payload::Params(nums),
                    5 => Payload::Datachunk(nums),
                    _ => unreachable!(),
                };

                Ok(Msg::Data(payload))
            }
            byte => Msg::invalid_kind_byte(byte),
        }
    }
}
