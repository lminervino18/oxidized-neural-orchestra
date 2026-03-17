use crate::{Result, optimization::Optimizer};

pub trait ForwardParamIter {
    fn next(&mut self, n: usize) -> Option<&mut [f32]>;
}

pub trait BackwardParamIter {
    fn next(&mut self, n: usize) -> Option<(&mut [f32], &mut [f32])>;
}

pub trait ParamProvider {
    type Front<'pm>: ForwardParamIter + 'pm
    where
        Self: 'pm;

    type Back<'pm>: BackwardParamIter + 'pm
    where
        Self: 'pm;

    fn front(&mut self) -> Self::Front<'_>;
    fn back(&mut self) -> Self::Back<'_>;

    fn optimize<O: Optimizer + Send>(&mut self, optimizers: &mut [O]) -> Result<()>;
    fn zero_grad(&mut self);
    fn acc_grad(&mut self);
}
