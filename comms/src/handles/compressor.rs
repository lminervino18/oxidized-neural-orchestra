use half::f16;
use rand::Rng;

use crate::{floats::Float01, sparse};

/// A compressed gradient.
pub enum CompressedGrad<'a> {
    Dense { grad: &'a [f16] },
    Sparse { sparse: &'a [u8], threshold: f32 },
}

/// Compresses a given residual gradient into a slice of f16 or a sparse binary
/// formatted slice.
pub struct Compressor<R>
where
    R: Rng,
{
    sparse_capability: Option<SparseCapability<R>>,
    compression_buf: Vec<f16>,
}

/// The necessary metadata for enabling sparse gradient compression.
struct SparseCapability<R>
where
    R: Rng,
{
    r: Float01,
    rng: R,
    ser_buf: Vec<u8>,
}

impl<R> Compressor<R>
where
    R: Rng,
{
    /// Creates a new `Compressor` with dense capabilities.
    ///
    /// # Returns
    /// A new `Compressor` instance.
    pub fn new() -> Self {
        Self {
            sparse_capability: None,
            compression_buf: Vec::new(),
        }
    }

    /// Enables the sparse gradient capability for this compressor.
    ///
    /// # Args
    /// * `r` - The ratio of compression for calculating the threshold value.
    /// * `rng` - A random number generator.
    pub fn enable_sparse_compression(&mut self, r: Float01, rng: R) {
        let sparse_capability = SparseCapability {
            r,
            rng,
            ser_buf: Vec::new(),
        };

        self.sparse_capability = Some(sparse_capability);
    }

    /// Compresses the given residual gradient.
    ///
    /// # Args
    /// * `residual` - The gradient to compress.
    ///
    /// # Returns
    /// The compressed gradient, either as a dense or sparse buffer.
    pub fn compress(&mut self, residual: &[f32]) -> CompressedGrad<'_> {
        match self.sparse_capability.as_mut() {
            Some(cap) => {
                cap.ser_buf.clear();

                let threshold = sparse::calculate_threshold(residual, cap.r, &mut cap.rng);
                sparse::grad_drop_into(&mut cap.ser_buf, residual, threshold);

                if cap.ser_buf.len() <= residual.len() * size_of::<f16>() {
                    CompressedGrad::Sparse {
                        sparse: &cap.ser_buf,
                        threshold,
                    }
                } else {
                    compress_dense_grad(&mut self.compression_buf, residual);
                    CompressedGrad::Dense {
                        grad: &self.compression_buf,
                    }
                }
            }
            None => {
                compress_dense_grad(&mut self.compression_buf, residual);
                CompressedGrad::Dense {
                    grad: &self.compression_buf,
                }
            }
        }
    }
}

/// Compresses the given gradient buffer into the inner `compression_buf`.
///
/// # Args
/// * `compression_buf`: The buffer where to write the compressed residual gradient.
/// * `residual` - The gradient to compress.
fn compress_dense_grad(compression_buf: &mut Vec<f16>, residual: &[f32]) {
    if let Some(additional) = residual.len().checked_sub(compression_buf.capacity()) {
        compression_buf.reserve(additional);
    }

    // SAFETY: The new uninitialized bytes will be overwritten right
    //         after with the compressed 16 bit gradient values.
    unsafe { compression_buf.set_len(residual.len()) };

    for (g, r) in compression_buf.iter_mut().zip(residual) {
        *g = f16::from_f32(*r);
    }
}
