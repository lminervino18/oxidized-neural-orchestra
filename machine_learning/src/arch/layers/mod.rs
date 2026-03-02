mod dense;
mod layer;

use std::mem;

use ndarray::Array2;

pub(super) use dense::Dense;
pub use layer::Layer;

/// A trait whose sole purpose is to give the ndarray::ArrayX
/// types a way to resize their inner memory regions inplace.
///
/// Only owned `ndarray`s should implement this trait.
trait InplaceReshape {
    /// Resizes the inner memory region for `Self` and returns a new instance.
    ///
    /// # Arguments
    /// * `shape` - The shape of the given array.
    ///
    /// # Returns
    /// A new `Self` instance with a new inner memory size.
    fn reshape_inplace(&mut self, shape: (usize, usize)) -> Self;
}

impl<T: Clone + Default> InplaceReshape for Array2<T> {
    fn reshape_inplace(&mut self, shape: (usize, usize)) -> Self {
        let arr = mem::take(self);

        let (mut v, Some(0)) = arr.into_raw_vec_and_offset() else {
            // SAFETY: This implementation assumes 'self' is a standard,
            // uniquely owned, contiguous array.
            //
            // into_raw_vec_and_offset returns None if the array's storage is
            // shared (cow) or not a standard Vec. It returns Some(offset)
            // where offset > 0 if the array is a slice/view into a larger
            // allocation.
            //
            // Since our architecture ensures layers own their parameters and
            // we do not pass slices into this internal utility, the offset
            // is guaranteed to be 0 and the storage uniquely owned.
            unreachable!("By how we're utilizing the arrays, we should never reach this point");
        };

        let size = shape.0 * shape.1;
        if size > v.len() {
            v.resize(size, T::default());
        }

        Array2::from_shape_vec(shape, v).unwrap()
    }
}
