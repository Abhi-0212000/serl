# Controller Types and SERL Policy Outputs Guide

## TL;DR - Verified from Code

**SERL Policy Output:** `np.ndarray` with shape `(action_dim,)` - **ONE action vector per step**

**No chunking, no sequences, no joint angles in default SERL - just delta end-effector commands**

---

## VERIFIED: What Policies Actually Output

### Code Proof #1: SAC/DrQ sample_actions()
```python
# File: serl_launcher/serl_launcher/agents/continuous/sac.py:302
def sample_actions(self, observations, seed=None, argmax=False):
    dist = self.forward_policy(observations, rng=seed, train=False)
    if argmax:
        return dist.mode()  # shape: (action_dim,)
    else:
        return dist.sample(seed=seed)  # shape: (action_dim,)
```
**Returns:** `jnp.ndarray` with shape matching `env.action_space.shape`

### Code Proof #2: Policy Network
```python
# File: serl_launcher/serl_launcher/networks/actor_critic_nets.py:178
class Policy(nn.Module):
    action_dim: int
    def __call__(self, observations):
        means = nn.Dense(self.action_dim)(outputs)  # Output layer
        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=stds)
        return distribution
```
**Returns:** Distribution that samples to exactly `action_dim` dimensions

### Code Proof #3: Actor Loop
```python
# File: examples/async_drq_sim/async_drq_sim.py:138
actions = agent.sample_actions(observations=obs, seed=key)
actions = np.asarray(jax.device_get(actions))
next_obs, reward, done, truncated, info = env.step(actions)
```
**Confirmed:** Direct action → env.step(), no processing

---

## Environment Action Spaces (Verified)

| Environment | Action Dim | Action Format | Code Location |
|------------|-----------|---------------|---------------|
| **Franka Sim** | 4 | `[dx, dy, dz, gripper]` | `franka_sim/envs/panda_pick_gym_env.py:134` |
| **Franka Real** | 7 | `[dx, dy, dz, droll, dpitch, dyaw, gripper]` | `serl_robot_infra/franka_env/envs/franka_env.py:125` |
| **Trossen Sim** | 4 | `[dx, dy, dz, gripper]` | `trossen_sim/envs/trossen_pick_gym_env.py:125` |
| **Trossen Dual** | 8 | `[L_dx, L_dy, L_dz, L_grip, R_dx, R_dy, R_dz, R_grip]` | `trossen_sim/envs/trossen_dual_stack_gym_env.py` |

### Verified Action Usage (Franka Real):
```python
# File: serl_robot_infra/franka_env/envs/franka_env.py:195
def step(self, action: np.ndarray):
    xyz_delta = action[:3]  # Delta position
    self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]
    
    self.nextpos[3:] = (  # Delta orientation
        Rotation.from_euler('xyz', action[3:6] * self.action_scale[1])
        * Rotation.from_quat(self.currpos[3:])
    ).as_quat()
    
    gripper_action = action[6] * self.action_scale[2]
```
**Confirmed:** Actions are deltas, not absolute poses

---

## 1. Two Controller Types (MuJoCo Only)

### **A. End-Effector (EE) Control (`ee_sim_env.py`)**

**What it is:**
- Uses **Motion Capture (mocap) bodies** to control the robot
- Directly commands the **end-effector position and orientation** in Cartesian space
- MuJoCo's inverse kinematics solver figures out joint angles automatically

**How it works:**
```python
# In ee_sim_env.py before_step():
action = [left_pos_x, left_pos_y, left_pos_z, left_quat_w, left_quat_x, left_quat_y, left_quat_z, left_gripper,
          right_pos_x, right_pos_y, right_pos_z, right_quat_w, right_quat_x, right_quat_y, right_quat_z, right_gripper]

# Set mocap position and quaternion
physics.data.mocap_pos[0] = action_left[:3]  # XYZ position
physics.data.mocap_quat[0] = action_left[3:7]  # Quaternion orientation
physics.data.qpos[6] = action_left[7]  # Gripper
```

**MuJoCo XML (`trossen_ai_scene.xml`):**
```xml
<body mocap="true" name="mocap_left" pos="-0.2062 -0.019 0.1835">
    <site pos="0 0 0" size="0.0003 0.03 0.0003" type="box" name="mocap_left_site1"/>
</body>
```

**Pros:**
- ✅ Smooth, natural movements
- ✅ Easy to script demo policies
- ✅ Good for data collection
- ✅ No joint limit violations

**Cons:**
- ❌ Mocap bodies visible in simulation (red boxes)
- ❌ Doesn't match real robot control exactly
- ❌ Can't test joint-level control strategies

**Used for:**
- Generating scripted demonstrations
- Testing task feasibility
- Quick prototyping

---

### **B. Joint Control (`sim_env.py`)**

**What it is:**
- Uses **position controllers** for each joint
- Directly commands **joint angles** (like real hardware)
- No mocap bodies - pure joint actuation

**How it works:**
```python
# In sim_env.py before_step():
action = [joint1_angle, joint2_angle, joint3_angle, joint4_angle, joint5_angle, joint6_angle, 
          gripper_left, gripper_right,  # Left arm
          joint1_angle, joint2_angle, joint3_angle, joint4_angle, joint5_angle, joint6_angle,
          gripper_left, gripper_right]  # Right arm (16 total)

# Set joint positions directly
physics.data.ctrl[:] = action
```

**MuJoCo XML (`trossen_ai_scene_joint.xml`):**
```xml
<actuator>
    <position name="left/joint_1" joint="left/joint_1" kp="50"/>
    <position name="left/joint_2" joint="left/joint_2" kp="50"/>
    ...
</actuator>
```

**Pros:**
- ✅ Matches real robot control
- ✅ Clean visualization (no mocap bodies)
- ✅ Tests joint-level policies
- ✅ Realistic joint dynamics

**Cons:**
- ❌ More complex to script
- ❌ Can hit joint limits
- ❌ Requires inverse kinematics for Cartesian goals

**Used for:**
- Replaying joint trajectories from ee_sim_env
- Training joint-level policies
- Sim-to-real transfer

---

## 2. All SERL Algorithms - Same Output Format

**All SERL algorithms output the same thing: action vectors**

| Algorithm | Output Shape | Code File |
|-----------|-------------|-----------|
| **DrQ** | `(action_dim,)` | `serl_launcher/agents/continuous/drq.py` |
| **SAC** | `(action_dim,)` | `serl_launcher/agents/continuous/sac.py:302` |
| **BC** | `(action_dim,)` | `serl_launcher/agents/continuous/bc.py:79` |
| **VICE** | `(action_dim,)` | `serl_launcher/agents/continuous/vice.py` (inherits from DrQ) |
| **RLPD** | `(action_dim,)` | Same as DrQ (RLPD = DrQ + demo buffer) |

### Example - All Algorithms:
```python
# BC:
actions = bc_agent.sample_actions(obs, argmax=True)  # shape: (action_dim,)

# SAC:
actions = sac_agent.sample_actions(obs, seed=key)  # shape: (action_dim,)

# DrQ:
actions = drq_agent.sample_actions(obs, seed=key)  # shape: (action_dim,)

# All output same format, all feed directly to env.step()
next_obs, reward, done, truncated, info = env.step(actions)
```

---

## 3. Training Examples with Outputs

### Sim Training (async_drq_sim)
```python
# File: examples/async_drq_sim/async_drq_sim.py
env = gym.make("PandaPickCube-v0")  # Action dim: 4
agent = make_drq_agent(..., sample_action=env.action_space.sample())

# Actor loop:
actions = agent.sample_actions(obs, seed=key)  # Output: (4,)
# actions = [dx, dy, dz, gripper]
next_obs, reward, done, truncated, info = env.step(actions)
```

### Real Robot Training (async_peg_insert_drq)
```python
# File: examples/async_peg_insert_drq/async_drq_randomized.py
env = gym.make("FrankaPegInsert-Vision-v0")  # Action dim: 7
agent = make_drq_agent(..., sample_action=env.action_space.sample())

# Actor loop:
actions = agent.sample_actions(obs, seed=key)  # Output: (7,)
# actions = [dx, dy, dz, droll, dpitch, dyaw, gripper]
next_obs, reward, done, truncated, info = env.step(actions)
```

### BC Training (bc_policy)
```python
# File: examples/bc_policy.py:220
agent = BCAgent.create(...)  # BC agent

# Evaluation:
actions = agent.sample_actions(obs, argmax=True)  # Output: (action_dim,)
next_obs, reward, done, truncated, info = env.step(actions)
```

---

## 4. Key Facts (Verified)

✅ **Fact 1:** All policies output `np.ndarray` with shape `(action_dim,)`
   - Source: `serl_launcher/agents/continuous/sac.py:302`

✅ **Fact 2:** Actions are **delta commands**, not absolute poses
   - Source: `serl_robot_infra/franka_env/envs/franka_env.py:195`

✅ **Fact 3:** No action chunking by default (1 action per step)
   - Source: All `examples/*/async_*.py` files show direct `env.step(actions)`

✅ **Fact 4:** All SERL envs use **end-effector control**, not joint control
   - Franka: OSC controller (operational space control)
   - Trossen/Panda Sim: Mocap bodies (MuJoCo IK)

✅ **Fact 5:** Action dimensions match environment exactly
   - `agent.action_dim == env.action_space.shape[0]`
   - Source: `serl_launcher/agents/continuous/drq.py:86`

---

## 5. Controller Types (MuJoCo Sim Only)

### EE Control vs Joint Control

**These ONLY apply to `trossen_arm_mujoco` package, NOT to main SERL**

**These ONLY apply to `trossen_arm_mujoco` package, NOT to main SERL**

| Feature | EE Control | Joint Control |
|---------|-----------|---------------|
| **File** | `trossen_arm_mujoco/ee_sim_env.py` | `trossen_arm_mujoco/sim_env.py` |
| **XML** | `trossen_ai_scene.xml` | `trossen_ai_scene_joint.xml` |
| **Action Dim** | 23 (pos+quat+grip × 2 + env) | 16 (8 joints × 2 arms) |
| **Control Type** | Cartesian (mocap bodies) | Joint angles |
| **Use Case** | Demo collection | Replay demos |

**Note:** Main SERL (franka_sim, trossen_sim, real robots) uses **end-effector delta control**, not these.

---

## 6. What About Action Chunking?

**By default: NO action chunking**

```python
# Default: 1 action per step
actions = agent.sample_actions(obs)  # shape: (action_dim,)
env.step(actions)
```

**With ChunkingWrapper: YES**
```python
# File: examples/async_drq_sim/async_drq_sim.py:330
env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

# Now actions can be chunked sequences
# But policy still outputs (action_dim,) - wrapper handles chunking
```

**Proof:** Only 1 file uses ChunkingWrapper:
```bash
$ grep -r "ChunkingWrapper" examples/
examples/async_drq_sim/async_drq_sim.py:    from serl_launcher.wrappers.chunking import ChunkingWrapper
examples/async_drq_sim/async_drq_sim.py:    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
```

---

## 7. Summary Table

| Question | Answer | Proof File |
|----------|--------|-----------|
| What do policies output? | `np.ndarray` shape `(action_dim,)` | `sac.py:302`, `bc.py:79` |
| Delta or absolute? | **Delta** (change in pose) | `franka_env.py:195` |
| Action chunking? | **No** (by default) | All `examples/*/async_*.py` |
| Joint control? | **No** (EE delta control) | All env files |
| Same for all algorithms? | **Yes** (DrQ, SAC, BC, VICE) | All agent files |
| Action dim for Franka? | **7** (dx,dy,dz,droll,dpitch,dyaw,grip) | `franka_env.py:125` |
| Action dim for Trossen? | **4** (dx,dy,dz,grip) | `trossen_pick_gym_env.py:125` |

---

## 8. Code References

**Policy Output:**
- `serl_launcher/serl_launcher/agents/continuous/sac.py:302` - `sample_actions()`
- `serl_launcher/serl_launcher/agents/continuous/bc.py:79` - `sample_actions()`  
- `serl_launcher/serl_launcher/networks/actor_critic_nets.py:178` - `Policy` network

**Action Spaces:**
- `franka_sim/franka_sim/envs/panda_pick_gym_env.py:134` - Sim action space (4D)
- `serl_robot_infra/franka_env/envs/franka_env.py:125` - Real action space (7D)
- `trossen_sim/trossen_sim/envs/trossen_pick_gym_env.py:125` - Trossen action space (4D)

**Action Usage:**
- `serl_robot_infra/franka_env/envs/franka_env.py:195` - How deltas are applied
- `examples/async_drq_sim/async_drq_sim.py:138` - Actor loop example

---

## 9. Common Mistakes

❌ **"Policies output joint angles"**
- ✅ They output **EE deltas**: `[dx, dy, dz, ...]`

❌ **"Actions are absolute poses"**  
- ✅ They are **delta movements** added to current pose

❌ **"SERL uses action chunking"**
- ✅ **No chunking by default** - 1 action per step

❌ **"Different algorithms output different things"**
- ✅ **All output** `(action_dim,)` action vectors

❌ **"EE control is for training"**
- ✅ EE control (`trossen_arm_mujoco`) is for **demo collection only**

---
