#[cfg(test)]
use {
    super::{dense::Dense, layer::Layer, sigmoid::Sigmoid},
    rand::Rng,
};

#[test]
fn xd1() {
    let mut a = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    let (a12, a3) = a.split_at_mut(2);
    let (a34, a5) = a3.split_at_mut(2);
    let (a56, a7) = a5.split_at_mut(2);
    a34[0] = 58;
    a56[1] = 100;
    assert_eq!(a12[0], 1);
    assert_eq!(a12[1], 2);
    assert_eq!(a34[0], 58);
    assert_eq!(a56[0], 5);
    assert_eq!(a56[1], 100);
    assert_eq!(a7[2], 9);
    // fiuf
}

#[test]
fn xd2() {
    let a = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    let mut res_chunk = &a[..];
    let chunk_size = 2;
    let mut a_chunks = vec![];
    for _ in 0..a.len() / chunk_size {
        let chunk;
        (chunk, res_chunk) = res_chunk.split_at(chunk_size);
        a_chunks.push(chunk);
    }

    assert_eq!(a_chunks[0], &[1, 2]);
    assert_eq!(a_chunks[1], &[3, 4]);
    assert_eq!(a_chunks[2], &[5, 6]);
    assert_eq!(a_chunks[3], &[7, 8]);
    // assert_eq!(a_chunks[4], &[9]); no entra al loop
}

#[test]
fn xd3() {
    let mut a = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    let len = a.len();
    let mut res_chunk = &mut a[..];
    let chunk_size = 2;
    let mut a_chunks = vec![];
    for _ in 0..len / chunk_size {
        let chunk;
        (chunk, res_chunk) = res_chunk.split_at_mut(chunk_size);
        a_chunks.push(chunk);
    }

    a_chunks[0][0] = 17;
    a_chunks[3][1] = 684;

    assert_eq!(a_chunks[0], &[17, 2]);
    assert_eq!(a_chunks[1], &[3, 4]);
    assert_eq!(a_chunks[2], &[5, 6]);
    assert_eq!(a_chunks[3], &[7, 684]);
    // fiuffffffffffff
}

// esto va a tener que ir sí o sí adentro de un model factory al que le pases las layers,
// probablemente haciendo que los chunks sean opcionales para que se puedan crear afuera sin todo
// este boilerplate. Le pasás las layers y la memoria de los weights y biases al factory y listo!!!
// :)
#[test]
fn mlp_init() {
    fn get_array_mut_chunks<'a>(sizes: &[usize], array: &'a mut [f32]) -> Vec<&'a mut [f32]> {
        let mut res_chunk = array;
        let mut chunks: Vec<&mut [f32]> = vec![];
        for s in sizes {
            let c;
            (c, res_chunk) = res_chunk.split_at_mut(*s);
            chunks.push(c);
        }

        chunks
    }

    let dims = [2, 3, 1];

    let w_dims: Vec<usize> = (0..dims.len() - 2).map(|i| dims[i] * dims[i + 1]).collect();
    let mut weights: Vec<f32> = (0..w_dims.iter().sum())
        .map(|_| rand::rng().random())
        .collect();

    let b_dims = dims;
    let mut biases: Vec<f32> = (0..dims.iter().sum())
        .map(|_| rand::rng().random())
        .collect();

    let mut w_chunks = get_array_mut_chunks(&w_dims, &mut weights);
    let mut b_chunks = get_array_mut_chunks(&b_dims, &mut biases);

    let _net = vec![
        Layer::Dense(
            Dense::new((dims[0], dims[1]), w_chunks.remove(0), b_chunks.remove(0)).unwrap(),
        ),
        Layer::Sigmoid(Sigmoid::new(dims[1])),
        Layer::Dense(
            Dense::new((dims[1], dims[2]), w_chunks.remove(0), b_chunks.remove(0)).unwrap(),
        ),
        Layer::Sigmoid(Sigmoid::new(dims[2])),
    ];
}
