# AIS / ML / ROS2 Assessment and Autonomy Roadmap

**Date**: 2026-08-05
**Method**: function-level inventory of `src/stdlib/{ais,ml,ros2}.iris`, compared
against the capability set a production autonomous system requires.

---

## Scorecard

| Module | Functions | Rating | One-line verdict |
|---|---|---|---|
| `std.ais` | 61 | **7 / 10** | Genuinely differentiated. The strongest reason to choose IRIS. |
| `std.ml` | 87 | **6 / 10** | Competent classical ML + inference bridges. Missing robotics estimation. |
| `std.ros2` | 29 | **2 / 10** | Publish-only telemetry. Cannot express a real robot node. |

The gap between the first and the third is the central finding: IRIS has a
research-grade autonomy library sitting on a middleware binding that cannot carry
it onto a robot.

---

## 1. `std.ais` — 7/10, the real differentiator

This is not a generic ML library with an "agent" wrapper. It implements concepts
that are current research practice in autonomous systems:

| Area | Functions |
|---|---|
| Homeostatic regulation | `homeostat_new/add_var/update/is_safe`, `homeovar_new` |
| Active inference | `active_inference_step`, `belief_update`, `genmodel_new` |
| Continual learning | `ewc_new`, `ewc_penalty` (elastic weight consolidation) |
| Intrinsic motivation | `novelty_score`, `curiosity_reward` |
| Decision policies | `argmax`, `epsilon_greedy`, `softmax_sample`, `boltzmann_sample`, `ucb_select` |
| Credit assignment | `discount_rewards`, `gae` |
| Neuroevolution | `genome_random/mutate/crossover`, `population_new` |
| Safety | `constitution_new/add/check` |
| Multi-agent | `mas_new/add_role/assign/consensus` |
| Metacognition | `metacog_new`, `meta_record`, `meta_performance_trend`, `meta_update_efficacy` |
| Autonomic loop | `mapek_agent_new`, `mapek_step`, `mapek_run`, `agent_loop` |
| Self-governance | `autonomic_rule_new`, `software_governor_new/add_rule`, `eval_rule` |

**Why this scores well:** active inference, EWC, intrinsic motivation and MAPE-K
are not commodity library content. Having them as *first-class stdlib* with a
language whose GC is pause-free and whose effects are statically tracked is a
defensible position no mainstream robotics stack occupies.

**Why not higher:**

- `mapek_step` / `mapek_run` were previously disabled because function-typed
  record fields could not be called. That defect is recorded as fixed
  (`CallClosure pass_env`), but the MAPE-K path is one of the 104 non-asserting
  tests — **verify before claiming it**.
- A safety "constitution" that is a list of predicate checks is a good primitive
  but is not a safety *case*. No runtime monitor synthesis, no formal guarantee.
- No uncertainty propagation into the decision policies; `std.uncertainty` exists
  separately (22 functions) but is not wired into `ais`.

---

## 2. `std.ml` — 6/10, solid but not robotics-shaped

**Present and competent:** activations with derivatives (sigmoid, relu,
leaky_relu, tanh, softmax), losses (MSE, BCE, cross-entropy, MAE), optimisers
(SGD, momentum, Adam), metrics (accuracy, R², precision, recall, F1, confusion
matrix), classical models (linear/logistic regression, k-NN), data prep
(standardise, train/test split, shuffle), and external inference via ONNX /
PyTorch / TensorFlow — now resolved by `dlopen` rather than linked.

**The robotics-specific gap.** None of the estimation and planning machinery an
autonomous system actually runs is present:

| Missing | Needed for |
|---|---|
| Kalman / EKF / UKF | sensor fusion, IMU+odometry state estimation |
| Particle filter | localisation under non-Gaussian noise |
| SLAM (or a binding) | mapping in unknown environments |
| A* / D* / RRT* | global path planning |
| DWA / TEB / MPC | local planning and obstacle avoidance |
| PID / LQR | low-level actuator control |
| Occupancy grid / costmap | navigation representation |
| Quaternion / SE(3) maths | pose composition, orientation |

`std.ais` gives you the *cognitive* layer; `std.ml` gives you the *learning*
layer. The **estimation and control layer between them is absent**, and that is
the layer a robot spends most of its cycles in.

---

## 3. `std.ros2` — 2/10, the blocking weakness

What exists: context/node/publisher/subscription lifecycle, and publishers for
`float64`, `int64`, `string`, `Vector3`, `Twist`, `Pose`, plus constructors for
those geometry types.

What that adds up to is **outbound telemetry**. The decisive limitation:

```iris
pub def wait_for_message(sub: ROS2Sub, timeout_ms: i64) -> bool
```

Subscription exists, but this returns a **`bool`** — you can detect that a
message arrived and cannot read it. There is no path to consume inbound data,
which means no perception, no closed loop, no reactive behaviour.

Measured against what ROS 2 nodes are built from — topics, services, actions,
parameters and tf2 as the core communication mechanisms
([MathWorks](https://www.mathworks.com/help/ros/gs/robot-operating-system-ros2-basic-concepts.html),
[Thelliez](https://thomasthelliez.com/blog/ros-2-architecture-patterns-that-scale/)):

| Capability | Status | Consequence if absent |
|---|---|---|
| Subscription **payload** | ❌ | No perception at all |
| Executors (single/multi-threaded, static) | ❌ | No callback dispatch; can't react to events |
| Services (request/response) | ❌ | No configuration, no queries |
| Actions (long-running goals with feedback) | ❌ | No navigation goals, no manipulation |
| Parameters | ❌ | No runtime tuning |
| **tf2** transforms | ❌ | Cannot relate sensor frames to robot frames — fatal for any multi-sensor robot |
| **QoS** policies (reliability, durability, deadline, liveliness) | ❌ | No control over lossy/real-time links |
| Lifecycle nodes (Unconfigured/Inactive/Active/Finalized) | ❌ | No managed startup, no supervisory control |
| Sensor messages (LaserScan, PointCloud2, Image, Odometry, JointState, IMU) | ❌ | Cannot speak to real drivers |
| Clock / simulated time | ❌ | No deterministic replay, no sim |

QoS in particular is not optional: its policies govern history, depth,
reliability, durability, deadline, lifespan and liveliness, and mismatches
silently break communication
([QoS dependency analysis](https://arxiv.org/pdf/2509.03381)).

**Verdict:** you cannot currently write a ROS 2 node in IRIS that does useful
work beyond publishing numbers.

---

## 4. What IRIS genuinely brings to autonomy

Worth stating plainly, because these are real and rare in combination:

1. **Deterministic, pause-free memory.** Reference counting with cycle collection
   rather than a tracing GC. This is a real advantage over Python and JVM stacks
   for control loops, where a GC pause is a missed deadline.
2. **Native compilation** with an LLVM pipeline, plus CUDA/NVPTX and SIMD
   backends — the same language for the control loop and the perception kernel.
3. **A static effect system.** Uniquely applicable to safety: `with pure { … }`
   and `effect io, net` can *prove at compile time* that a control path performs
   no I/O or allocation. No mainstream robotics language offers this.
4. **Borrow checking** for memory safety without a tracing collector.
5. **Capability sandbox** for constraining what a deployed agent may touch.
6. **Native tensors + tape autodiff**, so learning is not a foreign-library bolt-on.
7. **`std.ais`** — the cognitive layer, already written.

The pitch that follows from this is *not* "a better ROS 2 client". It is:
**a language in which the autonomy layer is statically verifiable and
deadline-predictable**, which is precisely what safety-critical autonomy lacks.

---

## 5. Roadmap to top-class autonomous software

Ordered by what unblocks the most. Phases 1–2 are prerequisites for any real
robot demonstration.

### Phase 1 — Make ROS 2 usable (highest priority)

| Task | Why first |
|---|---|
| `subscription_take(sub) -> option<Msg>` returning **payload** | Without it there is no perception. Everything else is blocked. |
| Executor + callback dispatch (start single-threaded) | The event loop every node is built around |
| Standard message types: `LaserScan`, `PointCloud2`, `Image`, `Odometry`, `IMU`, `JointState` | Required to talk to real drivers |
| QoS profiles (at minimum: reliable/best-effort, history depth) | Sensor topics need best-effort; commands need reliable |
| tf2: transform buffer, lookup, broadcast | Cannot fuse multi-sensor data without it |

*Deliverable:* a node that subscribes to `/scan`, transforms into `base_link`,
and publishes `/cmd_vel`. That single program proves the stack.

### Phase 2 — The estimation and control layer

| Task | Notes |
|---|---|
| `std.filter`: Kalman, EKF, UKF, complementary filter | The workhorse of state estimation |
| `std.spatial`: quaternions, SE(3), rotations, pose composition | Prerequisite for tf2 and for any 3D reasoning |
| `std.control`: PID, LQR, rate limiting, anti-windup | The regulator layer; IRIS's determinism is a genuine selling point here |
| `std.planning`: A*, RRT*, occupancy grid, costmap | Global planning |

### Phase 3 — Services, actions, lifecycle

Services and actions unlock navigation goals and manipulation. Lifecycle nodes
(Unconfigured → Inactive → Active → Finalized) give managed startup and
supervisory control, which is what makes a fleet deployable
([lifecycle overview](https://eureka.patsnap.com/report-ros-2-node-lifecycle-managed-nodes-health-and-supervisory-control)).

### Phase 4 — Play to the unique strengths

This is where IRIS stops catching up and starts leading:

- **Effect-typed safety contracts.** Annotate control functions with an effect row
  and have the compiler *prove* a real-time path performs no allocation and no
  I/O. Pair with `constitution_check` for runtime monitors that the type system
  has already constrained. Nobody else can do this.
- **Deadline-aware scheduling.** With no GC pauses, publish measured worst-case
  latency for a control loop. That number is the argument.
- **Verified autonomy demo.** `std.ais` homeostasis + safety constitution +
  effect-restricted control path, on real hardware over ROS 2 — a robot whose
  safety envelope is partly compile-time enforced.

### Phase 5 — Simulation and evidence

Gazebo or Isaac integration for hardware-in-the-loop; deterministic replay using
simulated clock. For funding, a reproducible sim demo de-risks the hardware claim.

---

## 6. Honest summary for a proposal

**Strength:** `std.ais` (61 functions of research-grade autonomy: active
inference, EWC, homeostasis, intrinsic motivation, MAPE-K, safety constitution)
combined with pause-free deterministic memory, native compilation, and a static
effect system. That combination is genuinely novel.

**Weakness:** the ROS 2 binding is publish-only — subscriptions cannot read
payloads, and there is no tf2, QoS, executor, service, action or lifecycle
support. The estimation/control layer (Kalman, SE(3), PID, planning) is also
absent. **Today, IRIS cannot drive a real robot.**

**The credible claim** is therefore about the *autonomy and safety layer*, not
about robotics middleware — with Phase 1 as the funded work that connects them.
Claiming robot-readiness now would not survive a technical reviewer opening
`ros2.iris` and noticing that `wait_for_message` returns a bool.

**Sources:**
- [ROS 2 basic concepts — MathWorks](https://www.mathworks.com/help/ros/gs/robot-operating-system-ros2-basic-concepts.html)
- [ROS 2 architecture patterns that scale — Thelliez](https://thomasthelliez.com/blog/ros-2-architecture-patterns-that-scale/)
- [ROS 2 node lifecycle and supervisory control](https://eureka.patsnap.com/report-ros-2-node-lifecycle-managed-nodes-health-and-supervisory-control)
- [Dependency chain analysis of ROS 2 DDS QoS policies (arXiv 2509.03381)](https://arxiv.org/pdf/2509.03381)
- [Impact of ROS 2 node composition in robotic systems (arXiv 2305.09933)](https://arxiv.org/pdf/2305.09933)
