use super::inline_src::InlineSrc;
use rand::Rng;
use std::ops::Range;

pub enum DatasetSrc {
    Inline(InlineSrc),
    // Stream(StreamSrc<R>),
}

impl DatasetSrc {
    pub fn inline(data: Vec<f32>) -> Self {
        DatasetSrc::Inline(InlineSrc::new(data))
    }
}

impl DatasetSrc {
    pub fn len(&self) -> usize {
        match self {
            DatasetSrc::Inline(src) => src.len(),
            // DatasetSrc::Stream(_src) => todo!(),
        }
    }

    pub fn shuffle<Rn: Rng>(&mut self, rows: usize, row_size: usize, rng: &mut Rn) {
        match self {
            DatasetSrc::Inline(src) => src.shuffle(rows, row_size, rng),
            // DatasetSrc::Stream(_src) => todo!(),
        }
    }

    pub fn raw_batch(&self, range: Range<usize>) -> &[f32] {
        match self {
            DatasetSrc::Inline(src) => src.raw_batch(range),
            // DatasetSrc::Stream(_src) => todo!(),
        }
    }
}
