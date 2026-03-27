use tokio::io;

type Idx = u64;
type ChunkLen = u32;

const IDX_SIZE: usize = size_of::<Idx>();
const CHUNK_LEN_SIZE: usize = size_of::<ChunkLen>();

fn get_threshold(grad: &[f32], r: f32) -> f32 {
    1.0
}

pub fn grad_drop(grad: &[f32], r: f32) -> Vec<u8> {
    let treshold = get_threshold(grad, r);
    let mut kept = vec![];

    let mut iter = grad.iter().enumerate().peekable();

    while let Some((i, g)) = iter.next() {
        if g.abs() >= treshold {
            kept.extend_from_slice(&(i as Idx).to_be_bytes());
            let size_idx = kept.len();
            kept.extend_from_slice(&(0 as ChunkLen).to_be_bytes());
            kept.extend_from_slice(&g.to_be_bytes());

            let mut chunk_len: ChunkLen = 1;
            while let Some((_, g)) = iter.peek()
                && g.abs() >= treshold
            {
                kept.extend_from_slice(&g.to_be_bytes());
                chunk_len += 1;
                iter.next();
            }

            kept[size_idx..size_idx + CHUNK_LEN_SIZE].copy_from_slice(&chunk_len.to_be_bytes());
        }
    }

    kept
}

/// # Panics
///
pub fn grad_lift(grad: &mut [f32], kept: &[u8]) -> io::Result<()> {
    let mut i = 0;

    while i < kept.len() {
        let idx_bytes = kept[i..i + IDX_SIZE].try_into().unwrap();

        let idx = Idx::from_be_bytes(idx_bytes) as usize;
        i += IDX_SIZE;

        let chunk_len_bytes = kept[i..i + CHUNK_LEN_SIZE].try_into().unwrap();
        let chunk_len = ChunkLen::from_be_bytes(chunk_len_bytes) as usize;
        i += CHUNK_LEN_SIZE;

        for j in 0..chunk_len {
            let num_bytes = kept[i..i + size_of::<f32>()].try_into().unwrap();
            let num: f32 = f32::from_be_bytes(num_bytes);
            i += size_of::<f32>();

            grad[idx + j] = num;
        }
    }

    Ok(())
}

#[test]
fn test_grad_drop() {
    let grad = vec![1.0, 0.0, 1.0, 0.0];

    let dropped = grad_drop(&grad, 1.0);
    println!("{:?}", dropped);
}

#[test]
fn test_grad_lift() {
    let kept = vec![
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 63, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 63,
        128, 0, 0,
    ];

    let expected = vec![1.0, 0.0, 1.0, 0.0];
    let mut got = vec![0.0; expected.len()];

    grad_lift(&mut got, &kept).unwrap();

    assert_eq!(got, expected);
}

#[test]
fn test_drop_and_lift_consistency() {
    let expected = vec![1.0, 0.0, 1.0, 0.0];
    let mut got = vec![0.0; expected.len()];

    let kept = grad_drop(&expected, 1.0);
    grad_lift(&mut got, &kept).unwrap();

    assert_eq!(got, expected);
}
