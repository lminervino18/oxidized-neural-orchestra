use ndarray::Array2;
use std::f32;

#[derive(Debug, Default)]
pub struct Sigmoid {
    dim: usize,
    amp: f32,
    a_out: Option<Array2<f32>>,
}

impl Sigmoid {
    pub fn new(dim: usize, amp: f32) -> Self {
        Self {
            dim,
            amp,
            a_out: None,
        }
    }

    pub fn f(&self, z: f32) -> f32 {
        self.amp / (1. + f32::consts::E.powf(-z))
    }

    pub fn df(&self, z: f32) -> f32 {
        let amp = self.amp;
        let s = self.f(z);

        (s * (amp - s)) / amp
    }

    pub fn forward(&mut self, z: Array2<f32>) -> Array2<f32> {
        let a_out = z.mapv_into(|z| self.f(z));

        // TODO: eventualmente devolver un ArrayView2
        self.a_out = Some(a_out.clone());
        a_out
    }

    pub fn backward(&mut self, mut d: Array2<f32>) -> Array2<f32> {
        // TODO: sacar el unwrap
        let a_out = self.a_out.take().unwrap();

        d.zip_mut_with(&a_out, |d, &a| {
            *d *= (a * (self.amp - a)) / self.amp;
        });

        d
    }
}
