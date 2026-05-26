//! Very small replay buffer utility. This is intentionally minimal to avoid
//! adding external dependencies; it provides FIFO push and a `recent(n)` view for
//! simple learners.

#[derive(Clone)]
pub struct ReplayBuffer<T> {
    data: Vec<T>,
    capacity: usize,
}

impl<T> ReplayBuffer<T> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, item: T) {
        if self.capacity == 0 {
            return;
        }
        if self.data.len() >= self.capacity {
            // drop the oldest
            self.data.remove(0);
        }
        self.data.push(item);
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
}
