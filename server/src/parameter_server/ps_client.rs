use std::{
    slice,
    sync::{
        Arc,
        atomic::{AtomicU8, AtomicUsize, Ordering},
    },
};

use tokio::sync::Notify;

#[derive(Debug)]
pub struct PSClient {
    pub(super) active_idx: Arc<AtomicU8>,
    pub(super) grads: [*mut f32; 2],
    pub(super) notifies: [Arc<Notify>; 2],
    pub(super) params: usize,
    pub(super) writers: [Arc<AtomicUsize>; 2],
}

impl PSClient {
    fn get_active_idx(&self) -> usize {
        loop {
            let idx = self.active_idx.load(Ordering::Acquire) as usize;
            self.writers[idx].fetch_add(1, Ordering::AcqRel);

            if self.active_idx.load(Ordering::Acquire) as usize == idx {
                break idx;
            }

            if self.writers[idx].fetch_sub(1, Ordering::Release) == 1 {
                self.notifies[idx].notify_one();
            }
        }
    }

    pub fn accumulate(&self, gradient: &[f32]) {
        let idx = self.get_active_idx();
        let grad = unsafe { slice::from_raw_parts_mut(self.grads[idx], self.params) };

        for (acc, &delta) in grad.iter_mut().zip(gradient) {
            *acc += delta;
        }

        if self.writers[idx].fetch_sub(1, Ordering::Release) == 1 {
            self.notifies[idx].notify_one();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parameter_server::PSServer, tests::TestOptimizer};

    fn get_grad(pc: &PSClient, idx: usize) -> &[f32] {
        unsafe { slice::from_raw_parts_mut(pc.grads[idx], pc.params) }
    }

    #[test]
    fn writes_the_buffer_pointed_by_the_active_idx() {
        let mut ps = PSServer::new(3, TestOptimizer {});
        let pc = ps.client_handle();

        let gradient = [1., 2., 3.];
        pc.accumulate(&gradient);

        assert_eq!(get_grad(&pc, 0), gradient);
    }

    #[tokio::test]
    async fn after_updating_weights_grad_is_cleared_and_switched() {
        let mut ps = PSServer::new(3, TestOptimizer {});
        let pc = ps.client_handle();

        let gradient = [1., 2., 3.];
        pc.accumulate(&gradient);
        assert_eq!(get_grad(&pc, 0), gradient);
        assert_eq!(get_grad(&pc, 1), [0.].repeat(3));

        ps.update_weights().await;
        assert_eq!(get_grad(&pc, 0), [0.].repeat(3));
        assert_eq!(get_grad(&pc, 1), [0.].repeat(3));

        pc.accumulate(&gradient);
        assert_eq!(get_grad(&pc, 0), [0.].repeat(3));
        assert_eq!(get_grad(&pc, 1), gradient);
    }
}
