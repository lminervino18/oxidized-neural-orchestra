use std::io;

use half::f16;

use super::{
    msg::{Header, Msg, Payload, HEADER_SIZE},
    sparse,
};

/// The message deserializer, it handles the deserialization of compressed and uncompressed messages.
#[derive(Clone)]
pub struct Deserializer {
    nums: Vec<f32>,
}

impl Deserializer {
    /// Creates a new `Deserializer`.
    ///
    /// # Returns
    /// A new `Deserializer` instance.
    pub fn new() -> Self {
        Self { nums: Vec::new() }
    }

    /// Creates a new `Deserializer` with a certain buffer size.
    ///
    /// # Args
    /// * `size` - The size of the internal number buffer.
    ///
    /// # Returns
    /// A new `Deserializer` instance.
    pub fn new_with_size(size: usize) -> Self {
        Self {
            nums: vec![0.0; size],
        }
    }

    /// Deserializes the given bytes and creates a new Self.
    ///
    /// # Args
    /// * `data` - A byte array.
    ///
    /// # Returns
    /// A result object that returns `Self` on success or `io::Error` on failure.
    pub fn deserialize<'a>(&'a mut self, data: &'a mut [u8]) -> io::Result<Msg<'a>> {
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
                // Sparse gradient: decompress f16 → f32
                self.nums.fill(0.0);
                sparse::grad_lift_into(&mut self.nums, rest).map_err(io::Error::other)?;
                Ok(Msg::Data(Payload::Grad(&mut self.nums)))
            }
            3 => {
                // Dense gradient: decode f16 → f32
                let f16_slice: &[f16] = bytemuck::cast_slice(rest);
                self.nums.resize(f16_slice.len(), 0.0);

                for (i, &f16_val) in f16_slice.iter().enumerate() {
                    self.nums[i] = f16_val.to_f32();
                }

                Ok(Msg::Data(Payload::Grad(&mut self.nums)))
            }
            4 => {
                // Params: raw f32
                let nums = bytemuck::cast_slice_mut(rest);
                Ok(Msg::Data(Payload::Params(nums)))
            }
            5 => {
                // Datachunk: raw f32
                let nums = bytemuck::cast_slice_mut(rest);
                Ok(Msg::Data(Payload::Datachunk(nums)))
            }
            byte => Msg::invalid_kind_byte(byte),
        }
    }
}
