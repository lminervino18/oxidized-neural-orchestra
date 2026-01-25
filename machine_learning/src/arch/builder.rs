// fn something() {
//     fn get_array_mut_chunks<'a>(sizes: &[usize], array: &'a mut [f32]) -> Vec<&'a mut [f32]> {
//         let mut res_chunk = array;
//         let mut chunks: Vec<&mut [f32]> = vec![];
//         for s in sizes {
//             let c;
//             (c, res_chunk) = res_chunk.split_at_mut(*s);
//             chunks.push(c);
//         }
//
//         chunks
//     }
//
//     let dims = [2, 3, 1];
//
//     let w_dims: Vec<usize> = (0..dims.len() - 2).map(|i| dims[i] * dims[i + 1]).collect();
//     let mut weights: Vec<f32> = (0..w_dims.iter().sum())
//         .map(|_| rand::rng().random())
//         .collect();
//
//     let b_dims = dims;
//     let mut biases: Vec<f32> = (0..dims.iter().sum())
//         .map(|_| rand::rng().random())
//         .collect();
//
//     let mut w_chunks = get_array_mut_chunks(&w_dims, &mut weights);
//     let mut b_chunks = get_array_mut_chunks(&b_dims, &mut biases);
//
//     let _net = vec![
//         Layer::Dense(
//             Dense::new((dims[0], dims[1]), w_chunks.remove(0), b_chunks.remove(0)).unwrap(),
//         ),
//         Layer::Sigmoid(Sigmoid::new(dims[1])),
//         Layer::Dense(
//             Dense::new((dims[1], dims[2]), w_chunks.remove(0), b_chunks.remove(0)).unwrap(),
//         ),
//         Layer::Sigmoid(Sigmoid::new(dims[2])),
//     ];
// }
