use ndarray::{Array1, ArrayView1, ShapeError};

pub struct Input<'a> {
    data: ArrayView1<'a, f32>,
}

impl<'a> Input<'a> {
    pub fn new(dim: usize, data_raw: &'a mut [f32]) -> Result<Self, ShapeError> {
        let data = ArrayView1::from_shape(dim, data_raw)?;
        Ok(Self { data })
    }

    pub fn forward(&self, x: ArrayView1<f32>) -> Array1<f32> {
        // q lástima q para el forward tendría hay que clonar, si no sí me gustaba esta capa :(
        x.to_owned()
    }
}
