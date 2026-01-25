use ndarray::Array2;
use std::f32;

#[derive(Debug, Default)]
pub struct Sigmoid {
    dim: usize,
    amp: f32,
    a: Option<Array2<f32>>,
}

impl Sigmoid {
    pub fn new(dim: usize, amp: f32) -> Self {
        Self {
            dim,
            amp,
            ..Default::default()
        }
    }

    fn sigmoid(&self, z: f32) -> f32 {
        self.amp / (1. + f32::consts::E.powf(-z))
    }

    pub fn forward(&mut self, z: Array2<f32>) -> Array2<f32> {
        let a = z.mapv_into(|z| self.sigmoid(z));

        // TODO: eventualmente devolver un ArrayView2
        self.a = Some(a.clone());
        a
    }

    pub fn backward(&mut self, mut d: Array2<f32>) -> Array2<f32> {
        // TODO: sacar el unwrap
        let a = self.a.take().unwrap();

        d.zip_mut_with(&a, |d, &a| {
            *d *= (a * (self.amp - a)) / self.amp;
        });

        d
    }
}
