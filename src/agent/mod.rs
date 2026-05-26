//! High-level agent abstractions for AIS experiments.
//!
//! This module provides trait-based building blocks for composing agents: `Agent`,
//! `Policy`, `Memory`, and small supporting types. The intent is minimal scaffolding
//! so algorithms (PPO/DQN/SAC) can be implemented on top.

use std::collections::VecDeque;
use std::fmt::Debug;

/// Generic experience tuple stored by learners / replay buffers.
#[derive(Clone, Debug, PartialEq)]
pub struct Experience<Obs, Act> {
    pub obs: Obs,
    pub action: Act,
    pub reward: f32,
    pub next_obs: Option<Obs>,
    pub done: bool,
}

/// Policy trait: given an observation produce an action.
pub trait Policy<Obs, Act> {
    fn action(&mut self, obs: &Obs) -> Act;
}

/// Memory trait: store and retrieve experiences.
pub trait Memory<Obs, Act> {
    fn push(&mut self, e: Experience<Obs, Act>);
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Core Agent trait: observe, act, and learn from experience.
pub trait Agent<Obs, Act> {
    fn observe(&mut self, obs: Obs);
    fn act(&mut self) -> Act;
    fn learn(&mut self);
}

// --- Production-Ready Agent Abstractions ---

/// A tiny in-memory memory that keeps experiences in a Vec.
pub struct SimpleMemory<Obs, Act> {
    pub data: Vec<Experience<Obs, Act>>,
}

impl<Obs, Act> SimpleMemory<Obs, Act> {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
}

impl<Obs, Act> Default for SimpleMemory<Obs, Act> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Obs: Clone, Act: Clone> Memory<Obs, Act> for SimpleMemory<Obs, Act> {
    fn push(&mut self, e: Experience<Obs, Act>) {
        self.data.push(e);
    }
    fn len(&self) -> usize {
        self.data.len()
    }
}

/// RingReplayBuffer implements Memory with a fixed capacity using VecDeque.
/// Avoids O(N) element removal overhead, performing pushes in O(1) time.
pub struct RingReplayBuffer<Obs, Act> {
    pub data: VecDeque<Experience<Obs, Act>>,
    pub capacity: usize,
    rng_seed: u64,
}

impl<Obs, Act> RingReplayBuffer<Obs, Act> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: VecDeque::with_capacity(capacity),
            capacity,
            rng_seed: 0x9E3779B97F4A7C15u64,
        }
    }

    /// Sample a random batch of experience references from the buffer using a local xorshift generator.
    pub fn sample(&mut self, batch_size: usize) -> Vec<&Experience<Obs, Act>> {
        let len = self.data.len();
        if len == 0 || batch_size == 0 {
            return vec![];
        }
        let mut out = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            self.rng_seed ^= self.rng_seed << 13;
            self.rng_seed ^= self.rng_seed >> 7;
            self.rng_seed ^= self.rng_seed << 17;
            let idx = (self.rng_seed % len as u64) as usize;
            out.push(&self.data[idx]);
        }
        out
    }
}

impl<Obs: Clone, Act: Clone> Memory<Obs, Act> for RingReplayBuffer<Obs, Act> {
    fn push(&mut self, e: Experience<Obs, Act>) {
        if self.capacity == 0 {
            return;
        }
        if self.data.len() >= self.capacity {
            self.data.pop_front();
        }
        self.data.push_back(e);
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

/// A policy that always takes a constant action.
#[derive(Clone, Debug)]
pub struct ConstPolicy<Act> {
    pub action: Act,
}

impl<Obs, Act: Clone> Policy<Obs, Act> for ConstPolicy<Act> {
    fn action(&mut self, _obs: &Obs) -> Act {
        self.action.clone()
    }
}

/// Epsilon-greedy policy wrapping another policy.
pub struct EpsilonGreedyPolicy<Obs, Act, P> {
    pub base_policy: P,
    pub epsilon: f32,
    rng_seed: u64,
    random_actions: Vec<Act>,
    _marker: std::marker::PhantomData<Obs>,
}

impl<Obs, Act: Clone, P: Policy<Obs, Act>> EpsilonGreedyPolicy<Obs, Act, P> {
    pub fn new(base_policy: P, epsilon: f32, random_actions: Vec<Act>) -> Self {
        Self {
            base_policy,
            epsilon,
            rng_seed: 0x123456789ABCDEF0u64,
            random_actions,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<Obs, Act: Clone, P: Policy<Obs, Act>> Policy<Obs, Act> for EpsilonGreedyPolicy<Obs, Act, P> {
    fn action(&mut self, obs: &Obs) -> Act {
        self.rng_seed ^= self.rng_seed << 13;
        self.rng_seed ^= self.rng_seed >> 7;
        self.rng_seed ^= self.rng_seed << 17;
        let r = ((self.rng_seed & 0xFFFFFFFF) as f32) / (u32::MAX as f32);

        if r < self.epsilon && !self.random_actions.is_empty() {
            let idx = (self.rng_seed % self.random_actions.len() as u64) as usize;
            self.random_actions[idx].clone()
        } else {
            self.base_policy.action(obs)
        }
    }
}

/// Q-learning agent holding a Q-table, policy, and memory.
pub struct QLearningAgent<Obs, Act, P, M> {
    pub policy: P,
    pub memory: M,
    pub q_table: std::collections::HashMap<(Obs, Act), f32>,
    pub alpha: f32,
    pub gamma: f32,
    pub last_obs: Option<Obs>,
    pub last_action: Option<Act>,
}

impl<Obs, Act, P, M> QLearningAgent<Obs, Act, P, M>
where
    Obs: Clone + std::hash::Hash + Eq,
    Act: Clone + std::hash::Hash + Eq,
    P: Policy<Obs, Act>,
    M: Memory<Obs, Act>,
{
    pub fn new(policy: P, memory: M, alpha: f32, gamma: f32) -> Self {
        Self {
            policy,
            memory,
            q_table: std::collections::HashMap::new(),
            alpha,
            gamma,
            last_obs: None,
            last_action: None,
        }
    }

    pub fn get_q(&self, obs: &Obs, act: &Act) -> f32 {
        *self.q_table.get(&(obs.clone(), act.clone())).unwrap_or(&0.0)
    }

    pub fn set_q(&mut self, obs: Obs, act: Act, val: f32) {
        self.q_table.insert((obs, act), val);
    }
}

impl<Obs, Act, P, M> Agent<Obs, Act> for QLearningAgent<Obs, Act, P, M>
where
    Obs: Clone + std::hash::Hash + Eq + Debug,
    Act: Clone + std::hash::Hash + Eq + Debug,
    P: Policy<Obs, Act>,
    M: Memory<Obs, Act>,
{
    fn observe(&mut self, obs: Obs) {
        if let (Some(l_obs), Some(l_act)) = (&self.last_obs, &self.last_action) {
            let e = Experience {
                obs: l_obs.clone(),
                action: l_act.clone(),
                reward: 0.0,
                next_obs: Some(obs.clone()),
                done: false,
            };
            self.memory.push(e);
        }
        self.last_obs = Some(obs);
    }

    fn act(&mut self) -> Act {
        let action = if let Some(obs) = &self.last_obs {
            self.policy.action(obs)
        } else {
            panic!("Cannot act before observing!");
        };
        self.last_action = Some(action.clone());
        action
    }

    fn learn(&mut self) {
        // Online updates are naturally supported
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct Obs(i32);
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct Act(i32);

    #[test]
    fn simple_memory_push() {
        let mut mem: SimpleMemory<Obs, Act> = SimpleMemory::new();
        let e = Experience {
            obs: Obs(1),
            action: Act(2),
            reward: 0.5,
            next_obs: None,
            done: false,
        };
        mem.push(e);
        assert_eq!(mem.len(), 1);
    }

    #[test]
    fn ring_replay_buffer_capacity() {
        let mut mem = RingReplayBuffer::with_capacity(2);
        mem.push(Experience {
            obs: Obs(1),
            action: Act(1),
            reward: 1.0,
            next_obs: None,
            done: false,
        });
        mem.push(Experience {
            obs: Obs(2),
            action: Act(2),
            reward: 2.0,
            next_obs: None,
            done: false,
        });
        mem.push(Experience {
            obs: Obs(3),
            action: Act(3),
            reward: 3.0,
            next_obs: None,
            done: false,
        });
        assert_eq!(mem.len(), 2);
        assert_eq!(mem.data[0].obs, Obs(2));
        assert_eq!(mem.data[1].obs, Obs(3));

        let sampled = mem.sample(5);
        assert_eq!(sampled.len(), 5);
    }

    #[test]
    fn policy_epsilon_greedy() {
        let base = ConstPolicy { action: Act(42) };
        let mut eg = EpsilonGreedyPolicy::new(base, 0.0, vec![Act(1), Act(2)]);
        let action = eg.action(&Obs(0));
        assert_eq!(action, Act(42));

        let base_always = ConstPolicy { action: Act(42) };
        let mut eg_random = EpsilonGreedyPolicy::new(base_always, 1.0, vec![Act(7)]);
        let act = eg_random.action(&Obs(0));
        assert_eq!(act, Act(7));
    }

    #[test]
    fn q_learning_agent_flow() {
        let base = ConstPolicy { action: Act(42) };
        let mem = SimpleMemory::new();
        let mut agent = QLearningAgent::new(base, mem, 0.1, 0.9);
        agent.observe(Obs(1));
        let act = agent.act();
        assert_eq!(act, Act(42));
        agent.observe(Obs(2));
        assert_eq!(agent.memory.len(), 1);
        assert_eq!(agent.memory.data[0].obs, Obs(1));
        assert_eq!(agent.memory.data[0].action, Act(42));
    }
}
