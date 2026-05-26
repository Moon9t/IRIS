use iris::agent::{Experience, Policy};
use iris::rl::{nstep::NStepBuffer, replay::ReplayBuffer, Env};

// Simple random policy (placeholder for a trainable policy)
struct RandomPolicy;
impl Policy<i32, i32> for RandomPolicy {
    fn action(&mut self, _obs: &i32) -> i32 {
        0
    }
}

// Reuse CounterEnv from tests as a toy environment
struct ToyEnv {
    count: i32,
    max: i32,
}
impl Env for ToyEnv {
    type Obs = i32;
    type Act = i32;
    fn reset(&mut self) -> Self::Obs {
        self.count = 0;
        self.count
    }
    fn step(&mut self, _action: Self::Act) -> (Self::Obs, f32, bool, Option<String>) {
        self.count += 1;
        let done = self.count >= self.max;
        (self.count, 1.0, done, None)
    }
}

fn main() {
    println!("PPO scaffold: collect rollouts and store in replay buffer.");
    let mut env = ToyEnv { count: 0, max: 10 };
    let mut policy = RandomPolicy;
    let mut replay: ReplayBuffer<Experience<i32, i32>> = ReplayBuffer::with_capacity(1024);
    let mut nbuf = NStepBuffer::new(3);

    for episode in 0..5 {
        let mut obs = env.reset();
        loop {
            let act = policy.action(&obs);
            let (next_obs, reward, done, _) = env.step(act);
            let exp = Experience {
                obs,
                action: act,
                reward,
                next_obs: Some(next_obs),
                done,
            };
            replay.push(exp);
            nbuf.push(obs, act, reward);
            if let Some((_o, a, ret)) = nbuf.maybe_pop(0.99) {
                // In a real PPO we'd compute advantages and add to batch here
                let _ = (_o, a, ret);
            }
            obs = next_obs;
            if done {
                break;
            }
        }
        println!("Episode {} done, buffer size {}", episode, replay.len());
    }
}
