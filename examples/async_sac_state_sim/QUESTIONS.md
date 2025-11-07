# Training Questions & Answers

## Q1: What data does actor put in buffer for each transition?

**Exact data stored per transition:**
```python
{
    "observations": obs,          # Flattened 23D array: 
                                  #   [0-5]:   left/joint_pos (6D)
                                  #   [6]:     left/gripper_pos (1D)
                                  #   [7-9]:   left/tcp_pos (3D) - gripper position in 3D space
                                  #   [10-15]: right/joint_pos (6D)
                                  #   [16]:    right/gripper_pos (1D)
                                  #   [17-19]: right/tcp_pos (3D) - gripper position in 3D space
                                  #   [20-22]: cube_pos (3D) ← CUBE POSITION IS INCLUDED!
    
    "actions": actions,           # 14D array: [left_joints(6), left_gripper(1), 
                                  #             right_joints(6), right_gripper(1)]
    "next_observations": next_obs,# Same 23D array after action
    "rewards": reward,            # Single float from reward function
    "masks": 1.0 - done,          # 1.0 if episode continues, 0.0 if done
    "dones": done or truncated    # Boolean: episode ended?
}
```

**The policy CAN plan to the object because:**
- `obs[20:23]` = cube position [x, y, z]
- `obs[7:10]` = left gripper position [x, y, z]
- Policy learns: "move joints so TCP position approaches cube position"

**NOT included:** Images (image_obs=False), joint velocities, torques, raw MuJoCo data

---

## Q2: How do robots move in MuJoCo UI?

**Step-by-step execution:**
1. Actor gets `obs` (current state)
2. Policy outputs `actions` (14D joint positions)
3. `env.step(actions)` does:
   - Denormalizes actions: [-1,1] → actual joint limits
   - Sets actuator targets in MuJoCo
   - Runs physics sim for `control_dt=0.02` seconds (20ms)
   - Returns new `obs`, `reward`, `done`
4. This happens **per timestep** (not batch, not episode)
5. Actor collects 1 transition, repeats

**Learner applies nothing to robots** - it only trains on buffered data

---

## Q3: When does environment reset?

**Reset triggers:**
1. **Time limit:** After 100 steps (`max_episode_steps=100` in gym registration)
   - 100 steps × 0.02s = 2 seconds per episode
2. **Done flag:** Never set in current code (no success/failure termination)
3. **Manual:** Only when actor calls `env.reset()`

**Reset happens:**
- Episode ends (100 steps elapsed)
- Actor immediately calls `env.reset()`
- Continues collecting from new episode

---

## Q4: Is cube spawn position random?

**YES - randomized every episode:**
```python
# In reset():
cube_pose = sample_box_pose()
# sample_box_pose() returns:
x: random in [-0.1, 0.2]
y: random in [-0.15, 0.15]  
z: fixed at 0.0125 (table height)
orientation: identity (no rotation)
```

Position randomizes to prevent overfitting to single location.

---

## Q5: Does inference need same time constraint?

**No strict requirement, but:**
- Trained policy learns 100-step behavior
- Inference can run unlimited steps
- But policy optimized for ~2 second tasks
- Longer episodes may show degraded performance
- Best practice: similar episode length as training

---

## Q6: Actor vs Learner - who does what?

**Actor (data collection):**
- Runs environment (MuJoCo simulation)
- Gets observations → samples actions → steps env
- Stores transitions in local buffer
- Flushes to learner every 30 steps
- Receives policy updates from learner
- **ONE step at a time** (not batches)

**Learner (training):**
- Never touches environment
- Receives transitions from actor
- Stores in replay buffer (1M capacity)
- Samples batches (2048 transitions)
- Updates policy/critic networks
- Sends new policy to actor
- **Batch learning** (256×8 transitions per update)

---

## Summary Table

| Aspect | Value |
|--------|-------|
| Transition data | obs(23D), action(14D), next_obs(23D), reward(1D), done(bool) |
| Episode length | 100 steps = 2 seconds |
| Step frequency | 50 Hz (0.02s per step) |
| Cube position | Random each reset: x∈[-0.1,0.2], y∈[-0.15,0.15] |
| Reset trigger | After 100 steps (time limit) |
| Learner updates | Per step (samples 2048 transitions from buffer) |
| Actor updates | Every 30 steps (receives new policy) |
| Demo data | None (pure RL from scratch) |
