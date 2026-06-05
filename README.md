# WheelFuturaPendulum — Isaac Lab Project

**A momentum-wheel inverted pendulum** RL control environment built on [Isaac Lab](https://github.com/isaac-sim/IsaacLab). The system consists of a two-link pendulum with a reaction wheel at the tip — only the wheel is actuated, making this an **underactuated swing-up and balance control** problem.

## Dependencies

- **[Isaac Lab](https://github.com/isaac-sim/IsaacLab)** — robot simulation framework (includes RSL-RL via `isaaclab_rl`)
- Python 3.10+

## Installation

```bash
# Install this package (Isaac Lab must already be installed separately)
python -m pip install -e source/WheelFuturaPendulum

# Verify the environment is registered
python -c "import gymnasium as gym; gym.make('Isaac-WheelFuturaPendulum-v0')"
```

## Task: `Isaac-WheelFuturaPendulum-v0`

### System Overview

```
base_link (fixed)
  └── joint1 (passive)
        └── link1
              └── joint2 (passive)
                    └── link2
                          └── wheel_joint (actuated)
                                └── wheel_link
```

- **Action** (1-D): torque on the reaction wheel, scaled to `[-10, 10]` N·m
- **Observation** (4-D): `[joint1_pos, joint2_pos, joint1_vel, joint2_vel]`
  - Relative joint positions and velocities
- **Reward**:
  - `+1` for staying alive (every step)
  - `-2` on termination
  - `-5 × ‖joint_pos − target‖²` tracking error for both `joint1` (target 0) and `joint2` (target −π)
  - `-0.005 × ‖joint_vel‖²` velocity penalty on pendulum and wheel joints
- **Termination**: time-out only (the pendulum needs full range to swing up)
- **Episode length**: 5 seconds at 60 Hz policy rate → 300 steps

### Key Configuration

| Parameter | Value | Note |
|-----------|-------|------|
| `action_scale` | 10.0 | Wheel torque scaling |
| `sim.dt` | 1/120 s | Physics step |
| `decimation` | 2 | Policy every 2 physics steps |
| `episode_length_s` | 5.0 | Episode duration |

## Training

```bash
# With rendering (default: 4096 parallel envs)
python scripts/rsl_rl/train.py --task Isaac-WheelFuturaPendulum-v0 --max_iterations 150

# Headless (faster)
python scripts/rsl_rl/train.py --task Isaac-WheelFuturaPendulum-v0 --headless --max_iterations 150
```

Training logs are saved to `logs/rsl_rl/`. Monitor with TensorBoard:

```bash
tensorboard --logdir logs/rsl_rl
```

### PPO Configuration (default)

| Parameter | Value |
|-----------|-------|
| `num_steps_per_env` | 16 |
| `actor_hidden_dims` | [32, 32] |
| `actor_obs_normalization` | False |
| `gamma` | 0.99 |
| `entropy_coef` | 0.005 |
| `learning_rate` | 1.0e-3 (adaptive schedule) |

## Testing the Trained Policy

```bash
python scripts/rsl_rl/play.py --task Isaac-WheelFuturaPendulum-v0
```

## Project Structure

```
WheelFuturaPendulum/
├── source/WheelFuturaPendulum/          # Python package
│   ├── WheelFuturaPendulum/
│   │   ├── assets/                      # Robot articulation config
│   │   └── tasks/
│   │       └── manager_based/wheelfuturapendulum/
│   │           ├── __init__.py           # Gym registration
│   │           ├── WheelFuturaPendulum_env_cfg.py  # MDP config
│   │           ├── agents/
│   │           │   └── rsl_rl_ppo_cfg.py           # PPO hyper-parameters
│   │           └── mdp/                 # Custom reward functions
│   └── asset/
│       └── robots/wheel_futura_pendulum.py  # USD articulation config
├── scripts/rsl_rl/                      # Training & play scripts
├── usd/                                 # USD & URDF robot models
└── logs/rsl_rl/                         # Training logs
```

## TODO (后续补充)

- [ ] Train a converged policy (current max 150 iterations)
- [ ] Adjust reward weights for better swing-up performance
- [ ] More detailed documentation
