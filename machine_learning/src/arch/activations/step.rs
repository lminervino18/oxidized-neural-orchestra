use std::f32;

#[derive(Clone, Debug, Default)]
pub struct Step {
    top: f32,
    bottom: f32,
    tresh: f32,
}

impl Step {
    pub fn new(top: f32, bottom: f32, tresh: f32) -> Self {
        Self { top, bottom, tresh }
    }

    pub fn f(&self, z: f32) -> f32 {
        let s = 1. / (1. + f32::consts::E.powf(-z));

        if s >= self.tresh {
            self.top
        } else {
            self.bottom
        }
    }

    pub fn df(&self, _z: f32) -> f32 {
        1.
    }
}
