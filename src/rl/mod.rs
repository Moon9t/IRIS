//! Minimal Reinforcement Learning environment and utilities.

pub mod nstep;
pub mod prioritized;
pub mod replay;

/// Gym-like environment trait.
pub trait Env {
    type Obs;
    type Act;

    fn reset(&mut self) -> Self::Obs;
    fn step(&mut self, action: Self::Act) -> (Self::Obs, f32, bool, Option<String>);
}
