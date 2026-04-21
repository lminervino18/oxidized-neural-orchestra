use std::io;

use tokio::io::{AsyncRead, AsyncWrite};

struct PeerMetadata {
    id: usize,
    addr: String,
}

/// The communication manager between the worker process and the all-reduce peers.
pub struct AllReduceMiddleware<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    prev: PeerMetadata,
    next: PeerMetadata,
    _marker: std::marker::PhantomData<(R, W)>,
}

impl<R, W> AllReduceMiddleware<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    /// Creates a new `AllReduceMiddleware`.
    ///
    /// # Returns
    /// A new `AllReduceMiddleware` instance.
    pub fn new(local_addr: &str, worker_addrs: Vec<String>) -> io::Result<Self> {
        let local_idx = find_local_worker_idx(local_addr, &worker_addrs)?;
        let (prev, next) = build_ring_neighbors(local_idx, &worker_addrs)?;

        Ok(Self {
            prev,
            next,
            _marker: std::marker::PhantomData,
        })
    }

    pub fn prev_id(&self) -> usize {
        self.prev.id
    }

    pub fn prev_addr(&self) -> &str {
        &self.prev.addr
    }

    pub fn next_id(&self) -> usize {
        self.next.id
    }

    pub fn next_addr(&self) -> &str {
        &self.next.addr
    }
}

fn find_local_worker_idx(local_addr: &str, worker_addrs: &[String]) -> io::Result<usize> {
    if worker_addrs.len() < 2 {
        return Err(io::Error::other("all-reduce requires at least two workers"));
    }

    worker_addrs
        .iter()
        .position(|addr| addr == local_addr)
        .ok_or_else(|| {
            io::Error::other(format!(
                "local worker address {local_addr:?} is not part of the ring"
            ))
        })
}

fn build_ring_neighbors(
    local_idx: usize,
    worker_addrs: &[String],
) -> io::Result<(PeerMetadata, PeerMetadata)> {
    if local_idx >= worker_addrs.len() {
        return Err(io::Error::other(format!(
            "worker index {local_idx} is not part of the ring"
        )));
    }

    let ring_len = worker_addrs.len();
    let prev_id = if local_idx == 0 {
        ring_len - 1
    } else {
        local_idx - 1
    };
    let next_id = (local_idx + 1) % ring_len;

    Ok((
        PeerMetadata {
            id: prev_id,
            addr: worker_addrs[prev_id].clone(),
        },
        PeerMetadata {
            id: next_id,
            addr: worker_addrs[next_id].clone(),
        },
    ))
}
