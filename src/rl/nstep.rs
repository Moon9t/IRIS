//! N-step buffer for accumulating multi-step returns.

pub struct NStepBuffer<Obs, Act> {
    buf: Vec<(Obs, Act, f32)>,
    n: usize,
}

impl<Obs, Act> NStepBuffer<Obs, Act> {
    pub fn new(n: usize) -> Self {
        Self { buf: Vec::new(), n }
    }

    pub fn push(&mut self, obs: Obs, act: Act, reward: f32) {
        self.buf.push((obs, act, reward));
    }

    /// If we have >= n entries, consume and return the n-step tuple:
    /// (obs_0, action_0, discounted_return, next_obs_n, done)
    pub fn maybe_pop(&mut self, discount: f32) -> Option<(Obs, Act, f32)>
    where
        Obs: Clone,
        Act: Clone,
    {
        if self.n > 0 && self.buf.len() >= self.n {
            let mut ret = 0.0f32;
            for (i, (_, _, r)) in self.buf.iter().take(self.n).enumerate() {
                ret += r * discount.powi(i as i32);
            }
            let (obs0, act0, _) = self.buf.remove(0);
            // drop n-1 front elements to slide window
            for _ in 0..(self.n - 1) {
                self.buf.remove(0);
            }
            Some((obs0, act0, ret))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nstep_accumulates() {
        let mut b = NStepBuffer::new(3);
        b.push(1i32, 0i32, 1.0);
        b.push(2, 0, 2.0);
        b.push(3, 0, 3.0);
        if let Some((_o, a, ret)) = b.maybe_pop(0.99) {
            assert_eq!(a, 0);
            assert!(ret > 5.9);
        } else {
            panic!("expected n-step")
        }
    }
}
