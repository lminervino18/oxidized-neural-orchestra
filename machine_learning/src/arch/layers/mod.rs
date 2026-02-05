mod dense;
mod layer;

pub(super) use dense::Dense;
pub use layer::Layer;

use std::mem;

use ndarray::Array2;

trait InplaceReshape {
    fn reshape_inplace(&mut self, shape: (usize, usize)) -> Self;
}

impl<T: Clone + Default> InplaceReshape for Array2<T> {
    fn reshape_inplace(&mut self, shape: (usize, usize)) -> Self {
        let arr = mem::take(self);

        let (mut v, Some(0)) = arr.into_raw_vec_and_offset() else {
            // TODO: ver de arreglar esto
            panic!("wtf, no es 0 el offset");
        };

        let size = shape.0 * shape.1;
        if size > v.len() {
            v.resize(size, T::default());
        }

        Array2::from_shape_vec(shape, v).unwrap()
    }
}
