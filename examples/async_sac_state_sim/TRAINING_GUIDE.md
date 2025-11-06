# async_sac_state_sim.py — Complete Training Guide

This guide explains **exactly** what happens when you run this script, with clear visual diagrams for each stage.

---
## Overview: Two Separate Processes

```
Terminal 1                          Terminal 2
┌─────────────────────┐            ┌─────────────────────┐
│  python ... py      │            │  python ... py      │
│    --learner        │            │    --actor          │
│    --ip localhost   │            │    --ip localhost   │
└─────────────────────┘            └─────────────────────┘
```

They run the **same Python script** but with different flags. They communicate over TCP network.

---
## Part 1: Startup Sequence (What Happens First)

### Step 1: You start LEARNER first
```bash
python async_sac_state_sim.py --learner --ip localhost
```

```
LEARNER STARTUP:
┌────────────────────────────────────────────────────────┐
│ 1. Create SAC Agent (random initial weights)          │
│    - Actor network (policy)                            │
│    - 2 Critic networks                                 │
│    - 2 Target critic networks                          │
│    - Temperature (α)                                   │
│                                                        │
│ 2. Create Replay Buffer (empty)                       │
│    capacity = 1,000,000 transitions                    │
│    (stores on GPU/CPU RAM)                             │
│                                                        │
│ 3. Start TrainerServer                                │
│    server.register_data_store("actor_env", buffer)    │
│    server.start(threaded=True)                         │
│    → Listens on network for incoming data             │
│                                                        │
│ 4. WAIT (blocking)                                     │
│    while len(replay_buffer) < 300:  # training_starts │
│        sleep(1)                                        │
│                                                        │
│ Status: Server running, waiting for data...           │
└────────────────────────────────────────────────────────┘
```

**What is replay buffer?**
- It's a **list of transitions** stored in RAM
- Each transition = `(observation, action, reward, next_observation, done, mask)`
- `capacity=1,000,000` means it can hold **1 million transitions** (not memory size!)
- Example: if observation is 20 floats, action is 7 floats, that's ~27*4 = 108 bytes per transition
- Total memory ≈ 108 MB for 1M transitions (small!)

**Why wait for 300 samples?**
- Need some data before training can start
- Can't sample from empty buffer
- Actor will fill it in next step

---
### Step 2: You start ACTOR (in new terminal)
```bash
python async_sac_state_sim.py --actor --ip localhost
```

```
ACTOR STARTUP:
┌────────────────────────────────────────────────────────┐
│ 1. Create same SAC Agent (same random weights)         │
│                                                        │
│ 2. Create QueuedDataStore (local buffer)              │
│    size = 2000 transitions                             │
│                                                        │
│ 3. Create TrainerClient                               │
│    client.connect(ip, wait_for_server=True)            │
│    → Connects to learner's server                     │
│                                                        │
│ 4. Register callback for policy updates               │
│    def update_params(params):                          │
│        agent.state.params = params  # replace weights  │
│    client.recv_network_callback(update_params)         │
│                                                        │
│ 5. Create environment                                  │
│    env = gym.make("HalfCheetah-v4")                    │
│                                                        │
│ 6. START collecting data (while learner waits)        │
└────────────────────────────────────────────────────────┘
```

---
## Part 2: Initial Data Collection (First 300 Steps)

### Understanding the Two Queues

**IMPORTANT: There are TWO separate storage locations:**

1. **Local Queue (Actor side)**: `QueuedDataStore(2000)`
   - Temporary buffer on actor process
   - Holds transitions until next flush
   - **Empties completely** every 30 steps when flushed
   - Max capacity: 2000 transitions (rarely fills up)

2. **Replay Buffer (Learner side)**: `replay_buffer`
   - Permanent storage for training
   - Receives data from actor flushes
   - **Never empties** (keeps accumulating)
   - Capacity: 1,000,000 transitions

```
ACTOR PROCESS (Steps 0-299):
┌─────────────────────────────────────────────────────────┐
│ obs = env.reset()  # get initial observation           │
│                                                         │
│ for step in range(300):  # warmup phase                │
│     ┌────────────────────────────────────────────────┐ │
│     │ Step A: Sample RANDOM action                   │ │
│     │   action = env.action_space.sample()           │ │
│     │   (no policy used yet - pure exploration)      │ │
│     └────────────────────────────────────────────────┘ │
│                                                         │
│     ┌────────────────────────────────────────────────┐ │
│     │ Step B: Execute in environment                 │ │
│     │   next_obs, reward, done = env.step(action)    │ │
│     └────────────────────────────────────────────────┘ │
│                                                         │
│     ┌────────────────────────────────────────────────┐ │
│     │ Step C: Store in LOCAL queue (actor side)     │ │
│     │   data_store.insert({                          │ │
│     │       observations: obs,        # 20D vector   │ │
│     │       actions: action,          # 7D vector    │ │
│     │       rewards: reward,          # 1 scalar     │ │
│     │       next_observations: next_obs,  # 20D      │ │
│     │       dones: done,              # bool         │ │
│     │       masks: 1.0 - done         # float        │ │
│     │   })                                           │ │
│     │   Local queue grows: 1, 2, 3, ... up to 30    │ │
│     └────────────────────────────────────────────────┘ │
│                                                         │
│     ┌────────────────────────────────────────────────┐ │
│     │ Step D: Every 30 steps, FLUSH to learner      │ │
│     │   if step % 30 == 0:                           │ │
│     │       client.update()                          │ │
│     │         1. Takes ALL items from local queue    │ │
│     │         2. Sends them over TCP to learner      │ │
│     │         3. LOCAL QUEUE NOW EMPTY (0 items)     │ │
│     └─────────┼─────────────────────────────────────┘ │
│               │                                        │
└───────────────┼────────────────────────────────────────┘
                │
                │ Network (TCP)
                │ Sends ~30 transitions
                ▼
┌────────────────────────────────────────────────────────┐
│ LEARNER PROCESS (still waiting)                        │
│                                                        │
│ server receives data → replay_buffer.insert()          │
│ (auto-wired via register_data_store)                   │
│                                                        │
│ After 1st flush (step 30):                             │
│   Replay buffer: [31 transitions]                      │
│   Progress: 31/300 (10%)                               │
│                                                        │
│ After 2nd flush (step 60):                             │
│   Replay buffer: [61 transitions]                      │
│   Progress: 61/300 (20%)                               │
│                                                        │
│ After 10th flush (step 300):                           │
│   Replay buffer: [301 transitions]                     │
│   Progress: 301/300 ✓ READY TO TRAIN!                 │
└────────────────────────────────────────────────────────┘
```

**Detailed Timeline of First 300 Steps:**
```
Step  Local Queue (actor)    Replay Buffer (learner)   Learner Status
─────────────────────────────────────────────────────────────────────
0     [t0]                   []                        Waiting...
1     [t0,t1]                []                        Waiting...
29    [t0..t29]              []                        Waiting...
30    FLUSH! → []            [t0..t30] (31 items)      Waiting 31/300
31    [t31]                  [t0..t30]                 Waiting...
59    [t31..t59]             [t0..t30]                 Waiting...
60    FLUSH! → []            [t0..t60] (61 items)      Waiting 61/300
90    FLUSH! → []            [t0..t90] (91 items)      Waiting 91/300
120   FLUSH! → []            [t0..t120] (121 items)    Waiting 121/300
150   FLUSH! → []            [t0..t150] (151 items)    Waiting 151/300
180   FLUSH! → []            [t0..t180] (181 items)    Waiting 181/300
210   FLUSH! → []            [t0..t210] (211 items)    Waiting 211/300
240   FLUSH! → []            [t0..t240] (241 items)    Waiting 241/300
270   FLUSH! → []            [t0..t270] (271 items)    Waiting 271/300
300   FLUSH! → []            [t0..t300] (301 items)    STARTS TRAINING! ✓
```

**Visual: Two-Queue System**
```
┌─────────────────────────────────────────────────────────────┐
│                    ACTOR PROCESS                            │
│                                                             │
│  Step 0-29: Collecting...                                   │
│  ┌───────────────────────────────┐                          │
│  │ Local Queue (QueuedDataStore) │ ← TEMPORARY              │
│  │ [t0, t1, t2, ..., t29, t30]   │    (empties on flush)    │
│  │ Size: 31/2000                 │                          │
│  └───────────────────────────────┘                          │
│                                                             │
│  Step 30: client.update() ──────────────┐                   │
│  (Takes all 31 items and sends)         │                   │
│                                         │                   │
│  ┌───────────────────────────────┐     │                   │
│  │ Local Queue AFTER flush       │     │                   │
│  │ []  ← EMPTY!                  │     │                   │
│  │ Size: 0/2000                  │     │                   │
│  └───────────────────────────────┘     │                   │
└────────────────────────────────────────┼───────────────────┘
                                         │
                                         │ TCP (31 transitions)
                                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   LEARNER PROCESS                           │
│                                                             │
│  server.register_data_store("actor_env", replay_buffer)    │
│      ↓ (auto-inserts received data)                         │
│                                                             │
│  ┌────────────────────────────────────┐                     │
│  │ Replay Buffer (permanent storage)  │ ← ACCUMULATES       │
│  │ [t0, t1, ..., t30]                 │    (never empties)  │
│  │ Size: 31/1,000,000                 │                     │
│  └────────────────────────────────────┘                     │
│                                                             │
│  while len(replay_buffer) < 300:                            │
│      time.sleep(1)  # Keep waiting...                       │
└─────────────────────────────────────────────────────────────┘

... 9 more flushes happen (steps 60, 90, 120, ..., 300) ...

┌─────────────────────────────────────────────────────────────┐
│                   LEARNER PROCESS                           │
│                                                             │
│  ┌────────────────────────────────────┐                     │
│  │ Replay Buffer (after 10 flushes)   │                     │
│  │ [t0, t1, ..., t299, t300]          │                     │
│  │ Size: 301/1,000,000                │                     │
│  └────────────────────────────────────┘                     │
│                                                             │
│  while len(replay_buffer) < 300:  ← FALSE! (301 >= 300)    │
│                                                             │
│  server.publish_network(agent.state.params)                 │
│  print("sent initial network to actor")                    │
│                                                             │
│  ✓ START TRAINING LOOP!                                     │
└─────────────────────────────────────────────────────────────┘
```

---
## Part 3: Learner Starts Training

Once replay buffer has 300 transitions:

```
LEARNER PROCESS:
┌────────────────────────────────────────────────────────┐
│ Replay buffer filled! (300/300)                        │
│                                                        │
│ server.publish_network(agent.state.params)             │
│   → Sends initial policy to actor                     │
│                                                        │
│ print("sent initial network to actor")                │
│                                                        │
│ for step in range(1,000,000):  # main training loop   │
│     ┌──────────────────────────────────────────────┐  │
│     │ STEP 1: Sample batch from replay buffer      │  │
│     │                                              │  │
│     │ batch_size = 256                             │  │
│     │ utd_ratio = 8  (critic_actor_ratio flag)     │  │
│     │ total_batch = 256 * 8 = 2048 transitions     │  │
│     │                                              │  │
│     │ batch = replay_buffer.sample(2048)           │  │
│     │   → randomly pick 2048 transitions           │  │
│     │   → each is (obs, action, reward, next_obs)  │  │
│     └──────────────────────────────────────────────┘  │
│                                                        │
│     ┌──────────────────────────────────────────────┐  │
│     │ STEP 2: Update agent (SAC training)          │  │
│     │                                              │  │
│     │ agent.update_high_utd(batch, utd_ratio=8)    │  │
│     │   (see detailed breakdown below)             │  │
│     └──────────────────────────────────────────────┘  │
│                                                        │
│     ┌──────────────────────────────────────────────┐  │
│     │ STEP 3: Publish updated policy               │  │
│     │                                              │  │
│     │ server.publish_network(agent.state.params)   │  │
│     │   → Actor receives new weights               │  │
│     └──────────────────────────────────────────────┘  │
│                                                        │
│ Logs: critic_loss, actor_loss, temperature → WandB    │
└────────────────────────────────────────────────────────┘
```

---
## Part 4: What is UTD Ratio? (The 8:1 Update Pattern)

**UTD = Update-To-Data ratio** = how many gradient updates per environment sample

```
batch_size = 256
utd_ratio = 8
total_samples = 256 * 8 = 2048 transitions sampled from replay buffer
```

**What `agent.update_high_utd(batch, utd_ratio=8)` does:**

```
┌────────────────────────────────────────────────────────────┐
│ Input: batch of 2048 transitions                          │
│                                                            │
│ STEP 1: Split into 8 mini-batches of 256 each             │
│   mini_batch_0 = batch[0:256]                              │
│   mini_batch_1 = batch[256:512]                            │
│   mini_batch_2 = batch[512:768]                            │
│   ...                                                      │
│   mini_batch_7 = batch[1792:2048]                          │
│                                                            │
│ STEP 2: Update CRITICS 8 times (once per mini-batch)      │
│   for i in range(8):                                       │
│       ┌────────────────────────────────────────────┐      │
│       │ mini_batch = batch[i]  (256 transitions)   │      │
│       │                                            │      │
│       │ # Compute target (what Q should predict)   │      │
│       │ next_actions = policy(next_obs)            │      │
│       │ Q_target_1 = target_critic_1(next_obs, a′) │      │
│       │ Q_target_2 = target_critic_2(next_obs, a′) │      │
│       │ Q_min = min(Q_target_1, Q_target_2)        │      │
│       │ y = reward + 0.99*(1-done)*Q_min           │      │
│       │     - α*log π(a′|s′)  # entropy term       │      │
│       │                                            │      │
│       │ # Update both critics                      │      │
│       │ Q1_pred = critic_1(obs, action)            │      │
│       │ Q2_pred = critic_2(obs, action)            │      │
│       │ loss = MSE(Q1_pred, y) + MSE(Q2_pred, y)   │      │
│       │ critics ← critics - lr * ∇loss             │      │
│       │                                            │      │
│       │ # Soft update target networks              │      │
│       │ target ← 0.995*target + 0.005*critic       │      │
│       └────────────────────────────────────────────┘      │
│                                                            │
│ STEP 3: Update ACTOR once (on full 2048 batch)            │
│   ┌────────────────────────────────────────────────┐      │
│   │ actions, log_probs = policy(obs)               │      │
│   │ Q1 = critic_1(obs, actions)                    │      │
│   │ Q2 = critic_2(obs, actions)                    │      │
│   │ Q_mean = (Q1 + Q2) / 2                         │      │
│   │                                                │      │
│   │ loss = mean(α*log_probs - Q_mean)              │      │
│   │ policy ← policy - lr * ∇loss                   │      │
│   └────────────────────────────────────────────────┘      │
│                                                            │
│ STEP 4: Update TEMPERATURE once (on full 2048 batch)      │
│   ┌────────────────────────────────────────────────┐      │
│   │ entropy = -mean(log_probs)                     │      │
│   │ target_entropy = -3.5  # typically -action_dim/2│      │
│   │                                                │      │
│   │ loss = α * (entropy - target_entropy)          │      │
│   │ α ← α - lr * ∇loss                             │      │
│   └────────────────────────────────────────────────┘      │
│                                                            │
│ Output: updated agent (new weights for all networks)      │
└────────────────────────────────────────────────────────────┘
```

**Why 8 critic updates but only 1 actor update?**
- Critics learn faster and benefit from more updates
- Actor can overfit if updated too often on same data
- 8:1 ratio gives better sample efficiency (learn more from same data)

---
## Part 5: Actor Continues Collecting (With Updated Policy)

```
ACTOR PROCESS (After receiving initial policy):
┌─────────────────────────────────────────────────────────┐
│ for step in range(300, 1,000,000):  # continue         │
│                                                         │
│     ┌────────────────────────────────────────────────┐ │
│     │ Step A: Sample action from POLICY (not random) │ │
│     │   rng, key = jax.random.split(rng)             │ │
│     │   action = agent.sample_actions(               │ │
│     │       obs, seed=key, deterministic=False       │ │
│     │   )                                            │ │
│     │   → Uses current policy weights                │ │
│     │   → Stochastic (explores via Gaussian noise)  │ │
│     └────────────────────────────────────────────────┘ │
│                                                         │
│     ┌────────────────────────────────────────────────┐ │
│     │ Step B: Execute in environment                 │ │
│     │   next_obs, reward, done = env.step(action)    │ │
│     └────────────────────────────────────────────────┘ │
│                                                         │
│     ┌────────────────────────────────────────────────┐ │
│     │ Step C: Store in local queue                   │ │
│     │   data_store.insert(transition)                │ │
│     └────────────────────────────────────────────────┘ │
│                                                         │
│     ┌────────────────────────────────────────────────┐ │
│     │ Step D: Every 30 steps                         │ │
│     │   client.update()                              │ │
│     │     1. Flush buffered data → learner           │ │
│     │     2. Receive new policy params ← learner     │ │
│     │     3. update_params() callback triggered      │ │
│     │        agent.state.params = new_params         │ │
│     └────────────────────────────────────────────────┘ │
│                                                         │
│ Result: Always using latest policy (or close to it)    │
└─────────────────────────────────────────────────────────┘
```

**Policy update frequency:**
- Learner publishes new policy **after every training step** (every 0.1-1 seconds)
- Actor receives updates **every 30 environment steps** (every ~0.6 seconds)
- Actor might lag behind by 30 steps max, but that's OK (off-policy learning)

---
## Part 6: Complete Data Flow (Ongoing Training)

```
┌──────────────────────────────────────────────────────────────┐
│                     COMPLETE CYCLE                           │
└──────────────────────────────────────────────────────────────┘

Actor (Step N):
  obs → policy → action → env → (obs, act, rew, next_obs, done)
                                         ↓
                                 data_store.insert()
                                         ↓
                            [Queue: 30 transitions]
                                         ↓
                         (every 30 steps: client.update())
                                         ↓
                                    ═══TCP═══
                                         ↓
Learner:                                 ↓
                        server receives 30 transitions
                                         ↓
                         replay_buffer.insert(30)
                                         ↓
                    [Buffer: 300 → 301 → ... → 1,000,000]
                                         │
                                         │ (continuously sample)
                                         ↓
                         batch = buffer.sample(2048)
                                         ↓
                         ┌────────────────────────┐
                         │ agent.update_high_utd  │
                         │   8× critic updates    │
                         │   1× actor update      │
                         │   1× temp update       │
                         └────────┬───────────────┘
                                  │ new policy params
                                  ↓
                    server.publish_network(params)
                                  ↓
                             ═══TCP═══
                                  ↓
Actor:                            ↓
              client receives new params
                                  ↓
              update_params(params) callback
                                  ↓
              agent.state.params ← new_params
                                  ↓
              Continue collecting with better policy
                                  ↓
                            [cycle repeats]
```

---
## Part 7: Batch Size Explained Simply

**What is a batch?**
A batch is a **collection of transitions** sampled from the replay buffer.

**Example transition:**
```python
{
    'observations': [1.2, 0.5, -0.3, ...],      # 20 floats (robot state)
    'actions': [0.1, -0.2, 0.5, ...],           # 7 floats (motor commands)
    'rewards': 2.3,                             # 1 float (reward from env)
    'next_observations': [1.3, 0.6, -0.2, ...], # 20 floats (next state)
    'dones': False,                             # 1 bool (episode ended?)
    'masks': 1.0                                # 1 float (1.0 - done)
}
```

**batch_size = 256 means:**
```python
batch = {
    'observations': array of shape (256, 20),      # 256 robot states
    'actions': array of shape (256, 7),            # 256 actions taken
    'rewards': array of shape (256,),              # 256 rewards
    'next_observations': array of shape (256, 20), # 256 next states
    'dones': array of shape (256,),                # 256 booleans
    'masks': array of shape (256,)                 # 256 floats
}
```

**With utd_ratio=8:**
```python
total_batch = batch_size * utd_ratio = 256 * 8 = 2048 transitions

# Split into 8 mini-batches:
mini_batch_0 = batch[0:256]      # first 256 transitions
mini_batch_1 = batch[256:512]    # next 256 transitions
...
mini_batch_7 = batch[1792:2048]  # last 256 transitions
```

Each mini-batch is used for one critic update. Then the full batch (2048) is used for one actor update.

---
## Part 8: Replay Buffer Capacity

```
replay_buffer_capacity = 1,000,000 transitions
```

**What happens as it fills up?**
```
Step 0-300:       [###                    ] 300/1,000,000 (0.03%)
Step 1000:        [##########             ] 1,000/1,000,000 (0.1%)
Step 10,000:      [####################   ] 10,000/1,000,000 (1%)
Step 100,000:     [###################### ] 100,000/1,000,000 (10%)
Step 1,000,000:   [########################] 1,000,000/1,000,000 (100% FULL)

After full:       Buffer becomes CIRCULAR (oldest data gets overwritten)
Step 1,000,001:   Buffer still 1,000,000 (removed transition 1, added 1,000,001)
```

**Why so large?**
- More diverse data = better learning
- Can sample from old experiences (off-policy)
- 1M transitions ≈ 100-200 MB RAM (small for modern GPUs)

---
## Part 9: Real Hardware vs Simulation

### Simulation (HalfCheetah-v4)
```
┌─────────────────────────────────────────┐
│ Actor speed: ~100-200 Hz (fast)         │
│ Learner speed: ~500-1000 Hz (GPU)       │
│                                         │
│ No real-time constraints                │
│ Can run multiple actors in parallel     │
│ Training completes in hours             │
└─────────────────────────────────────────┘
```

### Real Hardware (Franka Robot)
```
┌─────────────────────────────────────────┐
│ Actor speed: ~50 Hz (robot control rate)│
│ Learner speed: ~500-1000 Hz (same GPU)  │
│                                         │
│ REAL-TIME constraints (safety critical) │
│ Usually 1 actor (one physical robot)    │
│ Training takes days/weeks               │
│                                         │
│ Key difference: Actor is SLOW           │
│ → High UTD ratio (8-20) is essential    │
│ → Squeeze max learning from each sample │
└─────────────────────────────────────────┘
```

**Why async is critical for real robots:**
```
Without async (blocking):
  collect sample (20ms) → send to learner → wait for training (100ms) 
  → receive policy → repeat
  Total: 120ms per sample → 8 samples/sec → VERY SLOW

With async (non-blocking):
  Actor: collect sample (20ms) → queue → collect next (20ms) → ...
                                    ↓ (flush every 30 steps)
  Learner: ← receive batch → train → publish policy (actor doesn't wait)
  
  Result: 50 samples/sec from actor, 500 updates/sec from learner
```

---
## Part 10: Inference (Using Trained Policy)

After training, you can use the policy without the learner:

```python
# Load trained checkpoint
agent = SACAgent.load(checkpoint_path)

# Run policy in environment
obs = env.reset()
for step in range(1000):
    # Deterministic action (no exploration)
    action = agent.sample_actions(obs, argmax=True)
    
    obs, reward, done, info = env.step(action)
    
    if done:
        obs = env.reset()
```

**Key difference:**
- Training: `deterministic=False` (explores via Gaussian noise)
- Inference: `argmax=True` (uses mean action, no noise)

**What you DON'T need for inference:**
- Replay buffer
- Learner process
- Critic networks (only actor/policy is used)
- Temperature

**Minimal inference setup:**
```python
agent = SACAgent.load(checkpoint)
policy_fn = lambda obs: agent.sample_actions(obs, argmax=True)
# That's it! Just call policy_fn(obs) to get actions
```

---
## Part 11: Summary Timeline

```
Time     Learner Terminal              Actor Terminal
─────────────────────────────────────────────────────────
0:00     Start learner                 (not started)
         Create agent, buffer, server
         Wait for data...
         
0:05     (still waiting)               Start actor
                                       Connect to learner
                                       Collect random actions
                                       (steps 0-299)
                                       
0:10     Received 300 transitions!     (collecting step 300)
         Start training
         Publish initial policy   ───→ Receive policy
         
0:11     Sample batch (2048)            Use policy for actions
         Update critics 8×              (steps 301-330)
         Update actor 1×
         Update temp 1×
         Publish new policy      ───→  Receive new policy
         
0:12     Sample batch (2048)            Continue collecting
         Train...                       (steps 331-360)
         Publish policy          ───→  Receive policy
         
...      (training continues)           (collecting continues)
         ~1000 updates/sec              ~50-100 steps/sec
         
Hours later: Training complete         Stop actor
         Save checkpoint                Load checkpoint
                                       Run inference only
```

---
## Part 12: Key Takeaways

1. **Two processes**: Learner (trains) and Actor (collects data)
2. **Replay buffer**: Stores transitions (not memory size, but count of transitions)
3. **Batch**: 256 transitions sampled randomly from buffer
4. **UTD ratio**: 8 critic updates per 1 actor update (8:1 ratio)
5. **Total batch size**: `256 * 8 = 2048` transitions sampled per training step
6. **Buffer filling**: Actor sends data every 30 steps; learner waits until 300 transitions
7. **Policy sync**: Learner publishes after every update; actor receives every 30 steps
8. **Async benefit**: Actor never waits for learner; both run at max speed
9. **For real robots**: High UTD (8-20) essential because data collection is slow
10. **Inference**: Just load policy, no learner/buffer/critics needed

---
## Troubleshooting Visual

```
Problem: Learner stuck at "Filling up replay buffer 0/300"
Solution: Start actor! Learner can't train without data.

┌──────────────┐       ┌──────────────┐
│ Learner      │       │ Actor        │
│ Waiting...   │       │ NOT RUNNING  │ ← START THIS!
│ 0/300        │       └──────────────┘
└──────────────┘

─────────────────────────────────────────────────

Problem: Actor can't connect to learner
Solution: Start learner first! Actor needs server to be running.

┌──────────────┐       ┌──────────────┐
│ Learner      │       │ Actor        │
│ NOT RUNNING  │ ← START THIS FIRST! │ Connection refused │
└──────────────┘       └──────────────┘

─────────────────────────────────────────────────

Problem: Training unstable / diverging
Solution: Lower UTD ratio or increase batch size

critic_actor_ratio = 8  → try 4
batch_size = 256        → try 512
```

---
That's it! You now understand exactly how `async_sac_state_sim.py` works. 🎯
