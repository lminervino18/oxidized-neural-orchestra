use std::sync::{
    Arc,
    atomic::{AtomicU8, AtomicUsize, Ordering},
};

use rayon::prelude::*;
use tokio::sync::Notify;

use crate::{optimization::Optimizer, parameter_server::PSClient};

#[derive(Debug)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

#[derive(Debug, Default)]
struct GradientTable {
    grads: Vec<Box<[f32]>>,
    notify: Arc<Notify>,
    ptrs: Vec<SendPtr>,
    writers: Arc<AtomicUsize>,
}

#[derive(Debug)]
pub struct PSServer<O>
where
    O: Optimizer,
{
    active_idx: Arc<AtomicU8>,
    buf: Box<[f32]>,
    tables: [GradientTable; 2],
    optimizer: O,
    weights: Box<[f32]>,
}

impl<O> PSServer<O>
where
    O: Optimizer,
{
    pub fn new(params: usize, optimizer: O) -> Self {
        Self {
            active_idx: Default::default(),
            tables: Default::default(),
            buf: vec![0.; params].into_boxed_slice(),
            weights: vec![0.; params].into_boxed_slice(),
            optimizer,
        }
    }

    pub fn client_handle(&mut self) -> PSClient {
        let params = self.weights.len();
        let worker_id = self.tables[0].grads.len();

        for grad_table in self.tables.iter_mut() {
            let grad = vec![0.; params].into_boxed_slice();
            grad_table.grads.push(grad);

            let ptr_to_grad = grad_table.grads[worker_id].as_mut_ptr();
            grad_table.ptrs.push(SendPtr(ptr_to_grad));
        }

        let active_idx = Arc::clone(&self.active_idx);
        let grads = self.tables.each_ref().map(|table| table.ptrs[worker_id].0);
        let writers = self.tables.each_ref().map(|table| table.writers.clone());
        let notifies = self.tables.each_ref().map(|table| table.notify.clone());

        PSClient {
            active_idx,
            grads,
            notifies,
            params,
            writers,
        }
    }

    pub async fn update_weights(&mut self) {
        let frozen_idx = self.active_idx.fetch_xor(1, Ordering::Release) as usize;
        let GradientTable {
            writers,
            notify,
            ptrs,
            ..
        } = &self.tables[frozen_idx];

        while writers.load(Ordering::Acquire) > 0 {
            notify.notified().await;
        }

        self.buf.par_iter_mut().enumerate().for_each(|(i, g)| {
            let mut sum = 0.;

            for SendPtr(ptr) in ptrs {
                unsafe {
                    let val_ptr = ptr.add(i);
                    sum += *val_ptr;
                    *val_ptr = 0.;
                }
            }

            *g = sum;
        });

        self.optimizer.update_weights(&mut self.weights, &self.buf);
    }

    pub fn get_weights(&self) -> &[f32] {
        &self.weights
    }
}
