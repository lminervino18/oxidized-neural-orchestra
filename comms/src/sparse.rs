use half::f16;
use rand::{Rng, seq::IndexedRandom};
use serde::{Deserialize, Serialize};

type Idx = u64;
type ChunkLen = u32;

const IDX_SIZE: usize = size_of::<Idx>();
const CHUNK_LEN_SIZE: usize = size_of::<ChunkLen>();
const SAMPLE_SIZE_MAX: usize = 1 << 14;

const MIN_POSITIVE_F16: f32 = f16::MIN_POSITIVE.to_f32_const();

/// A float with a value between `0.0` and `1.0`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Float01 {
    value: f32,
}

impl Float01 {
    /// Creates a new `Float01`.
    ///
    /// # Args
    /// * `value` - The value to store.
    ///
    /// # Returns
    /// A new `Float01` instance if the given value is between `0.0` and `1.0`.
    pub fn new(value: f32) -> Option<Self> {
        (0.0 <= value && value <= 1.0).then_some(Self { value })
    }
}

/// Calculates the gradient's threshold approximately by sampling it's values.
///
/// # Args
/// * `residual` - The residual gradient to use to calculate the threshold.
/// * `r` - The ratio of compression for calculating the threshold value.
/// * `rng` - A random number generator to stochastically sample the gradient's values.
///
/// # Returns
/// The threshold to use with `grad_drop`.
pub fn calculate_threshold<R: Rng>(residual: &[f32], r: Float01, rng: &mut R) -> f32 {
    if residual.is_empty() {
        return 0.0;
    }

    let sample_size = SAMPLE_SIZE_MAX.min(residual.len());
    let mut sample: Vec<_> = residual
        .choose_multiple(rng, sample_size)
        .map(|x| x.abs())
        .collect();

    let k = (sample_size as f32 * (1.0 - r.value)) as usize;
    let k = k.clamp(0, sample_size - 1);
    sample.select_nth_unstable_by(k, |a, b| a.total_cmp(b));

    sample[k].max(MIN_POSITIVE_F16)
}

/// Serializes the given gradient into `buf` dropping any values with a magnitude lower than `threshold`.
///
/// # Args
/// * `buf` - The buffer to use to serialize the residual gradient.
/// * `residual` - The residual gradient to serialize.
/// * `threshold` - The minimum value the gradient's values have to reach to be sent.
pub fn grad_drop_into(buf: &mut Vec<u8>, residual: &[f32], threshold: f32) {
    buf.reserve(residual.len() / 10);

    let mut i = 0;
    while i < residual.len() {
        if residual[i].abs() >= threshold {
            let start = i;

            while i < residual.len() && residual[i].abs() >= threshold {
                i += 1;
            }

            let chunk_len = i - start;
            buf.extend_from_slice(&(start as Idx).to_le_bytes());
            buf.extend_from_slice(&(chunk_len as ChunkLen).to_le_bytes());

            for &g in &residual[start..i] {
                let g_short = f16::from_f32(g);
                buf.extend_from_slice(&g_short.to_le_bytes());
            }
        } else {
            i += 1;
        }
    }
}

/// Deserializes a sparse gradient into `grad`.
///
/// # Args
/// * `grad` - The gradient to apply the new values.
/// * `buf` - The buffer containing the serialized sparse gradient.
///
/// # Returns
/// An error if there are missing or invalid values in the input buffer.
pub fn grad_lift_into(grad: &mut [f32], buf: &[u8]) -> Result<(), &'static str> {
    let mut i = 0;

    while i < buf.len() {
        let idx_bytes: [_; IDX_SIZE] = buf
            .get(i..i + IDX_SIZE)
            .ok_or("Missing index bytes at grad lift")?
            .try_into()
            .unwrap();

        let idx = Idx::from_le_bytes(idx_bytes) as usize;
        i += IDX_SIZE;

        let chunk_len_bytes: [_; CHUNK_LEN_SIZE] = buf
            .get(i..i + CHUNK_LEN_SIZE)
            .ok_or("Missing chunk length bytes at grad lift")?
            .try_into()
            .unwrap();

        let chunk_len = ChunkLen::from_le_bytes(chunk_len_bytes) as usize;
        i += CHUNK_LEN_SIZE;

        if idx > grad.len() || grad.len() - idx < chunk_len {
            return Err("Gradient chunk exceeds target vector bounds");
        }

        for j in idx..idx + chunk_len {
            let b = buf
                .get(i..i + size_of::<f16>())
                .ok_or("Truncated float data")?;

            grad[j] = f16::from_le_bytes([b[0], b[1]]).to_f32();
            i += size_of::<f16>();
        }
    }

    Ok(())
}

#[test]
fn test_grad_drop() {
    let grad = vec![1.0, -1.0, 0.0, 2.0];
    let mut buf = Vec::new();
    let threshold = 1.0;

    grad_drop_into(&mut buf, &grad, threshold);

    let expected = vec![
        0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 60, 0, 188, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        64,
    ];

    assert_eq!(buf, expected);
}

#[test]
fn test_grad_lift() {
    let buf = vec![
        0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 60, 0, 188, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        64,
    ];

    let expected = vec![1.0, -1.0, 0.0, 2.0];
    let mut grad = vec![0.0; expected.len()];
    grad_lift_into(&mut grad, &buf).unwrap();

    assert_eq!(grad, expected);
}

#[test]
fn test_drop_and_lift_consistency() {
    let residual = vec![1.0, -1.0, 0.0, 2.0];
    let expected = residual.clone();

    let mut buf = Vec::new();
    let threshold = 1.0;

    grad_drop_into(&mut buf, &residual, threshold);

    let mut grad = vec![0.0; residual.len()];
    grad_lift_into(&mut grad, &buf).unwrap();

    assert_eq!(grad, expected);
}
