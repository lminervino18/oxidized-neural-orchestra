mod conv2d;
mod dense;
mod layer;
mod reshape;
mod sigmoid;

use std::mem;

use ndarray::{Array, Dimension, IntoDimension};

pub(super) use conv2d::Conv2d;
pub(super) use dense::Dense;
pub use layer::Layer;
pub(super) use reshape::Reshape;
pub(super) use sigmoid::Sigmoid;

/// A trait whose sole purpose is to give the ndarray::ArrayX
/// types a way to resize their inner memory regions inplace.
///
/// Only owned `ndarray`s should implement this trait.
pub trait InplaceReshape<D: Dimension> {
    /// Resizes the inner memory region for `Self` and returns a new instance.
    ///
    /// # Arguments
    /// * `shape` - The shape of the given array.
    ///
    /// # Returns
    /// A new `Self` instance with a new inner memory size.
    fn reshape_inplace<I>(&mut self, shape: I)
    where
        I: IntoDimension<Dim = D>;
}

impl<T: Clone + Default, D: Dimension> InplaceReshape<D> for Array<T, D> {
    fn reshape_inplace<I>(&mut self, shape: I)
    where
        I: IntoDimension<Dim = D>,
    {
        let dim = shape.into_dimension();
        if self.raw_dim() == dim {
            return;
        }

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
            unreachable!("Owned array had non-zero offset during inplace reshaping");
        };

        let size = dim.size();
        if size > v.len() {
            v.resize(size, T::default());
        }

        // SAFETY: v was previously resized to
        //         accomodate the new dimension.
        *self = Array::from_shape_vec(dim, v).unwrap();
    }
}
