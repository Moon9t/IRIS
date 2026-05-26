//! A minimal prioritized replay implementation without external RNG deps.
//!
//! Stores `(priority, item)` pairs and samples proportionally using prefix-sums
//! and a small xorshift RNG implemented locally to avoid adding `rand` as a
//! dependency.

pub struct PrioritizedReplay<T> {
    data: Vec<(f32, T)>,
    capacity: usize,
}

impl<T> PrioritizedReplay<T> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, priority: f32, item: T) {
        if self.capacity == 0 {
            return;
        }
        if self.data.len() >= self.capacity {
            // drop lowest priority item (naive)
            if let Some((idx, _)) = self.data.iter().enumerate().min_by(|a, b| {
                a.1 .0
                    .partial_cmp(&b.1 .0)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                self.data.remove(idx);
            }
        }
        self.data.push((priority.max(1e-6), item));
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Sample `k` items by weighted priority using prefix sums and a simple RNG.
    pub fn sample(&self, k: usize) -> Vec<(usize, &T)> {
        if self.data.is_empty() {
            return vec![];
        }
        let mut out = Vec::with_capacity(k);
        let mut prefs: Vec<f32> = Vec::with_capacity(self.data.len());
        let mut acc = 0.0f32;
        for (p, _) in &self.data {
            acc += *p;
            prefs.push(acc);
        }
        let total = acc.max(1e-12);

        let mut rng = SimpleRng::new();
        for _ in 0..k {
            let r = rng.next_f32() * total;
            let idx = match prefs
                .binary_search_by(|v| v.partial_cmp(&r).unwrap_or(std::cmp::Ordering::Equal))
            {
                Ok(i) => i,
                Err(i) => i,
            };
            let chosen = if idx >= self.data.len() {
                self.data.len() - 1
            } else {
                idx
            };
            out.push((chosen, &self.data[chosen].1));
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
    fn next_f32(&mut self) -> f32 {
        let v = self.next_u64();
        ((v & 0xFFFFFFFF) as f32) / (u32::MAX as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_sample() {
        let mut buf = PrioritizedReplay::with_capacity(4);
        buf.push(0.1, 1);
        buf.push(10.0, 2);
        buf.push(5.0, 3);
        assert_eq!(buf.len(), 3);
        let s = buf.sample(2);
        assert_eq!(s.len(), 2);
    }
}
