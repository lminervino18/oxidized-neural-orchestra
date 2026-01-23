use super::r#trait::Layer;

struct NetFactory;

impl NetFactory {
    pub fn create<'a>(
        mut layers: Vec<Layer>,
        dims: &[usize],
        raw_weights: &'a mut [f32],
        raw_biases: &'a mut [f32],
    ) -> Vec<Layer<'a>> {
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

        let w_chunks = get_array_mut_chunks(dims, raw_weights);
        let b_chunks = get_array_mut_chunks(dims, raw_biases);
    }
}
