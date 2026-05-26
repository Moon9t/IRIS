//! High-level agent abstractions for AIS experiments.
//!
//! This module provides trait-based building blocks for composing agents: `Agent`,
//! `Policy`, `Memory`, and small supporting types. The intent is minimal scaffolding
//! so algorithms (PPO/DQN/SAC) can be implemented on top.

use std::fmt::Debug;

/// Generic experience tuple stored by learners / replay buffers.
#[derive(Clone, Debug)]
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
}

/// Core Agent trait: observe, act, and learn from experience.
pub trait Agent<Obs, Act> {
    fn observe(&mut self, obs: Obs);
    fn act(&mut self) -> Act;
    fn learn(&mut self);
}

// --- Simple example implementations for tests ---

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

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    struct Obs(i32);
    #[derive(Clone, Debug, PartialEq)]
    struct Act(i32);

    struct ConstPolicy;
    impl Policy<Obs, Act> for ConstPolicy {
        fn action(&mut self, _obs: &Obs) -> Act {
            Act(42)
        }
    }

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
    fn policy_returns_action() {
        let mut p = ConstPolicy;
        let a = p.action(&Obs(0));
        assert_eq!(a.0, 42);
    }
}
