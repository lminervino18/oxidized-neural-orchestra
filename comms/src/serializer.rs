use half::f16;
use rand::{rngs::StdRng, Rng, SeedableRng};

use super::{
    msg::{Header, Msg, Payload},
    sparse::{self, Float01},
};

/// The message serializer, it handles the different capabilities for message serialization.
#[derive(Clone)]
pub struct Serializer(Inner<StdRng>);

impl Default for Serializer {
    /// Creates a new `Serializer` with the base configuration.
    ///
    /// # Returns
    /// A new `Serializer` instance.
    fn default() -> Self {
        Self(Inner::default())
    }
}

impl Serializer {
    /// Creates a new base `Serializer`.
    ///
    /// # Returns
    /// A new `Serializer` instance.
    pub fn new() -> Self {
        Self(Inner::Base)
    }

    /// Creates a new `Serializer` capable of sparse gradient serialization.
    ///
    /// # Args
    /// * `r` - The ratio of compression for calculating the threshold value.
    /// * `seed` - A seed to initialize a random number generator.
    ///
    /// # Returns
    /// A new `Serializer` instance.
    pub fn new_sparse_capable(r: Float01, seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        };

        let inner = Inner::SparseCapable {
            ser_buf: Vec::new(),
            r,
            rng,
        };

        Self(inner)
    }

    /// Serializes a given message and writes it's bytes in `out`.
    ///
    /// The writing presedence is first to write the contents of `out` and then the optional return bytes.
    ///
    /// # Args
    /// * `msg` - The message to serialize.
    /// * `out` - The output buffer.
    ///
    /// # Returns
    /// An optional zero copy slice and the threshold used to filter the
    /// gradient values if the payload was smaller than sending the dense
    /// gradient.
    pub fn serialize<'a>(
        &'a mut self,
        msg: &'a Msg<'a>,
        out: &mut Vec<u8>,
    ) -> (Option<&'a [u8]>, Option<f32>) {
        match &mut self.0 {
            Inner::Base => {
                let zero_copy_data = Inner::<StdRng>::serialize_base(&msg, out);
                (zero_copy_data, None)
            }
            Inner::SparseCapable { ser_buf, r, rng } => {
                Inner::serialize_sparse_capable(&msg, out, ser_buf, *r, rng)
            }
        }
    }
}

/// An internal structure meant to obfuscate the inner implementation of the message serializer.
#[derive(Default, Clone)]
enum Inner<R: Rng> {
    #[default]
    Base,
    SparseCapable {
        ser_buf: Vec<u8>,
        r: Float01,
        rng: R,
    },
}

impl<R: Rng> Inner<R> {
    /// Serializes the given message without any extra capabilities.
    ///
    /// # Args
    /// * `msg` - The message to serialize.
    /// * `out` - The output buffer.
    ///
    /// # Returns
    /// An optional zero copy slice.
    fn serialize_base<'a>(msg: &'a Msg<'a>, out: &mut Vec<u8>) -> Option<&'a [u8]> {
        match msg {
            Msg::Err(detail) => {
                let header = (0 as Header).to_be_bytes();
                out.extend_from_slice(&header);

                // SAFETY: Serialize impl for `Detail` is derived and not implemented
                //         by hand. Nor has a non string-key map inside.
                serde_json::to_writer(out, &detail).unwrap();
                None
            }
            Msg::Control(cmd) => {
                let header = (1 as Header).to_be_bytes();
                out.extend_from_slice(&header);

                // SAFETY: Serialize impl for `Command` is derived and not implemented
                //         by hand. Nor has a non string-key map inside.
                serde_json::to_writer(out, &cmd).unwrap();
                None
            }
            Msg::Data(payload) => match payload {
                Payload::Grad(grad) => {
                    // Encode dense gradients as f16 to halve network traffic.
                    let header = (3 as Header).to_be_bytes();
                    out.extend_from_slice(&header);

                    for &g in grad.iter() {
                        let g16 = f16::from_f32(g);
                        out.extend_from_slice(&g16.to_le_bytes());
                    }

                    None
                }
                Payload::Params(params) => {
                    let header = (4 as Header).to_be_bytes();
                    out.extend_from_slice(&header);
                    Some(bytemuck::cast_slice(params))
                }
                Payload::Datachunk(chunk) => {
                    let header = (5 as Header).to_be_bytes();
                    out.extend_from_slice(&header);
                    Some(bytemuck::cast_slice(chunk))
                }
            },
        }
    }

    /// Serializes the given message with the sparse matrix capability.
    ///
    /// # Args
    /// * `msg` - The message to serialize.
    /// * `out` - The output buffer.
    /// * `ser_buf` - A helper buffer to write the compressed bytes.
    /// * `r` - The ratio of compression for calculating the threshold value.
    /// * `rng` - A random number generator.
    ///
    /// # Returns
    /// An optional zero copy slice and the threshold used to filter the
    /// gradient values if the payload was smaller than sending the dense
    /// gradient.
    fn serialize_sparse_capable<'a>(
        msg: &'a Msg<'a>,
        out: &mut Vec<u8>,
        ser_buf: &'a mut Vec<u8>,
        r: Float01,
        rng: &mut R,
    ) -> (Option<&'a [u8]>, Option<f32>) {
        let Msg::Data(Payload::Grad(grad)) = msg else {
            let zero_copy_data = Self::serialize_base(msg, out);
            return (zero_copy_data, None);
        };

        let threshold = sparse::calculate_threshold(grad, r, rng);

        ser_buf.clear();
        sparse::grad_drop_into(ser_buf, grad, threshold);

        if ser_buf.len() < grad.len() * size_of::<f16>() {
            // Sparse is smaller than dense f16 → send sparse
            let header = (2 as Header).to_be_bytes();
            out.extend_from_slice(&header);
            (Some(ser_buf), Some(threshold))
        } else {
            // Dense f16 is smaller or equal → send dense as f16
            let header = (3 as Header).to_be_bytes();
            out.extend_from_slice(&header);

            for &g in grad.iter() {
                let g16 = f16::from_f32(g);
                out.extend_from_slice(&g16.to_le_bytes());
            }

            (None, None)
        }
    }
}
