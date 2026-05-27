//! Very small replay buffer utility. This is intentionally minimal to avoid
//! adding external dependencies; it provides FIFO push and a `recent(n)` view for
//! simple learners.

use std::collections::VecDeque;

#[derive(Clone)]
pub struct ReplayBuffer<T> {
    data: VecDeque<T>,
    capacity: usize,
}

impl<T> ReplayBuffer<T> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, item: T) {
        if self.capacity == 0 {
            return;
        }
        if self.data.len() >= self.capacity {
            // drop the oldest
            self.data.pop_front();
        }
        self.data.push_back(item);
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Return the most recent `n` items (or fewer if not available).
    pub fn recent(&self, n: usize) -> Vec<&T> {
        let len = self.data.len();
        if n >= len {
            self.data.iter().collect()
        } else {
            self.data.iter().skip(len - n).collect()
        }
    }

    /// Sample `k` items randomly from the buffer using a simple xorshift RNG.
    pub fn sample(&self, k: usize) -> Vec<&T> {
        let len = self.data.len();
        if len == 0 || k == 0 {
            return vec![];
        }
        let mut out = Vec::with_capacity(k);
        let mut rng = SimpleRng::new();
        for _ in 0..k {
            let idx = (rng.next_u64() % len as u64) as usize;
            out.push(&self.data[idx]);
        }
        out
    }
}

// Simple xorshift-based RNG for sampling (no external deps)
struct SimpleRng(u64);
impl SimpleRng {
    fn new() -> Self {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        SimpleRng(t ^ 0x9E3779B97F4A7C15)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_recent() {
        let mut buf = ReplayBuffer::with_capacity(3);
        buf.push(1);
        buf.push(2);
        buf.push(3);
        assert_eq!(buf.len(), 3);
        buf.push(4);
        assert_eq!(buf.len(), 3);
        let recent = buf.recent(2);
        assert_eq!(*recent[0], 3);
        assert_eq!(*recent[1], 4);
    }

    #[test]
    fn test_sample() {
        let mut buf = ReplayBuffer::with_capacity(5);
        buf.push(10);
        buf.push(20);
        buf.push(30);
        let s = buf.sample(3);
        assert_eq!(s.len(), 3);
        for item in s {
            assert!(*item == 10 || *item == 20 || *item == 30);
        }
    }
}
