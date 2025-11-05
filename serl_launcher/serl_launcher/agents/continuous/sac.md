# Soft Actor-Critic (SAC) — Complete Beginner's Guide

This document starts from zero RL knowledge and builds up to how SAC works, then maps everything to the SERL implementation with concrete answers to all your questions.

---

## Table of Contents
1. [Foundation: What is RL?](#1-foundation-what-is-rl)
2. [Value Functions: V(s) and Q(s,a) Explained](#2-value-functions-vs-and-qsa-explained)
3. [Reward vs Value Functions](#3-reward-vs-value-functions)
4. [Multiple Critics (min_i Q_target_i)](#4-multiple-critics-min_i-q_target_i)
5. [Neural Networks in RL](#5-neural-networks-in-rl)
6. [The Three Loss Functions](#6-the-three-loss-functions)
7. [Optimizers and make_optimizer](#7-optimizers-and-make_optimizer)
8. [Ensemblize and N Critics](#8-ensemblize-and-n-critics)
9. [Pixel-Based vs State-Based Policies](#9-pixel-based-vs-state-based-policies)
10. [UTD (Update-to-Data Ratio)](#10-utd-update-to-data-ratio)
11. [Complete SAC Architecture Summary](#11-complete-sac-architecture-summary)
12. [How SERL Implements It](#12-how-serl-implements-it)

---

## 1) Foundation: What is RL?

**The Setup:**
- **Environment**: the world (e.g., a robot arm, a video game)
- **Agent**: your AI that tries to learn
- **State (s)**: description of the world at time t (e.g., joint angles, object positions, or raw camera pixels)
- **Action (a)**: what the agent does (e.g., motor torques, button presses)
- **Reward (r)**: a number the environment gives after each action (e.g., +1 for success, -0.01 for each step, +10 for reaching goal)

**The Loop (one timestep):**
```
1. Agent sees state s_t
2. Agent chooses action a_t using policy π
3. Environment responds with:
   - new state s_{t+1}
   - reward r_t
   - done flag (episode ended?)
```

**Goal of RL:**
Learn a policy π that maximizes total reward over time (called the "return").

**Return (G_t):**
```
G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
```
where γ (gamma, discount factor) ∈ [0,1) makes future rewards worth less than immediate ones.

---

## 2) Value Functions: V(s) and Q(s,a) Explained

### The Core Question: "How do we know future rewards?"

**We don't know the actual future! But we can ESTIMATE it.**

### V(s) - State Value Function

**Intuitive definition:**
"If I'm in state s right now and follow my policy π from here on, how much total reward do I expect to get?"

**Mathematical definition:**
```
V^π(s) = E_π[ r_t + γ*r_{t+1} + γ²*r_{t+2} + ... | s_t = s ]
```

**Example (Robot Reaching Task):**
- State s1: robot arm near the target → V(s1) might be +8 (probably will succeed soon)
- State s2: robot arm far from target → V(s2) might be +2 (will take many steps)
- State s3: robot stuck in obstacle → V(s3) might be -5 (probably won't succeed)

**How do we learn V(s)?**
We approximate it with a neural network! The network takes state s as input and outputs a single number (the estimated value).

### Q(s,a) - Action Value Function

**Intuitive definition:**
"If I'm in state s, take action a RIGHT NOW, and then follow my policy π afterward, how much total reward do I expect?"

**Mathematical definition:**
```
Q^π(s,a) = E_π[ r_t + γ*r_{t+1} + γ²*r_{t+2} + ... | s_t = s, a_t = a ]
```

**Example (Robot with 2 action choices):**
- State s: robot arm at position X
  - Action a1 (move left): Q(s, a1) = +3 (okay path)
  - Action a2 (move right): Q(s, a2) = +9 (better path toward goal!)

The Q function tells us which action is better!

**Relationship between V and Q:**
```
V(s) = E_a~π[ Q(s,a) ]
```
(V is the average Q over all actions the policy might take)

**How do we learn Q(s,a)?**
Neural network! Input: state s and action a (concatenated). Output: one number (the Q-value).

---

## 3) Reward vs Value Functions

**This is a critical distinction!**

### Reward Function r(s,a) or r(s,a,s')
- **Given by the environment** (not learned)
- **Immediate**: tells you how good that ONE step was
- Example: "You moved closer to the goal, here's +0.1"

### Value Function V(s) or Q(s,a)
- **Learned by the agent** (approximated with neural nets)
- **Long-term**: estimates total future rewards (sum of all future immediate rewards)
- Example: "From this state, if you keep going optimally, you'll get +8 total"

**Analogy:**
- Reward = your salary this month ($5000)
- Value = your estimated lifetime earnings from now ($800,000)

The value function is trying to predict the sum of all future rewards!

**How do we learn the value function?**
Using the **Bellman equation** (recursive relationship):
```
Q(s,a) = r + γ * V(s')
      ≈ r + γ * E_a'~π[ Q(s', a') ]
```

In words: "The value of (s,a) equals immediate reward plus discounted value of next state."

We use this equation to create training targets for our neural network!

---

## 4) Multiple Critics (min_i Q_target_i)

### Why Multiple Critics?

**Problem with one Q-network:**
Q-networks tend to **overestimate** values (they're optimistic). This makes the policy think bad actions are better than they are, leading to poor performance.

**Solution: Twin Q or Ensemble Q**
Train 2 or more Q-networks (called critics) in parallel:
- Q_1(s,a) with parameters φ_1
- Q_2(s,a) with parameters φ_2
- ... (or more)

### What does min_i Q_target_i(s', a') mean?

When computing the target for training (the "y" we want our Q to match), we:
1. Evaluate ALL critics on the next state-action pair (s', a')
2. Take the MINIMUM value

**Example with 2 critics:**
```
State s', Action a'
Q_1_target(s', a') = 10.5  (optimistic)
Q_2_target(s', a') = 8.3   (more conservative)

min(Q_1_target, Q_2_target) = 8.3  ← use this for target!
```

**Why minimum?**
- If we used the maximum or average, the overestimation bias compounds
- Taking the min gives us a conservative (underestimated) value, which prevents the policy from being misled by overly optimistic Q-values

**In SAC:**
- Typically 2 critics (called "twin Q") or 10+ critics (ensemble)
- SERL often uses `critic_ensemble_size=10` with `critic_subsample_size=2` (REDQ-style)
- This means: train 10 critics, but when computing target, randomly pick 2 and take their min

### Do we have separate V and Q for each critic?

**No! SAC only uses Q-functions, not V.**

In SAC:
- We have N Q-networks (critics): Q_1(s,a), Q_2(s,a), ..., Q_N(s,a)
- We DON'T explicitly learn V(s)
- When we need V(s), we compute it as: V(s) ≈ E_a~π[ Q(s,a) - α*log π(a|s) ]

So if you have 2 critics in SAC:
- 2 Q-networks (that's it!)
- NO separate V networks

---

## 5) Neural Networks in RL

### Are Q-values just neural network weights?

**No! Let me clarify this confusion:**

**Q-value = OUTPUT of a neural network**
**Neural network weights = the PARAMETERS (θ or φ) we're learning**

**Example:**
```python
# Neural network (2-layer MLP)
def q_network(state, action, weights):
    x = concat([state, action])          # [s, a]
    x = weights['layer1'] @ x + bias1    # hidden layer
    x = relu(x)
    q = weights['layer2'] @ x + bias2    # output (single number)
    return q

# Usage
state = [1.0, 2.0, 3.0]  # robot joint angles
action = [0.5, -0.2]      # motor commands
weights = {...}           # learned parameters (weights/biases)

q_value = q_network(state, action, weights)  # returns e.g., 7.3
```

**In this example:**
- `q_value = 7.3` is the Q-VALUE (the prediction)
- `weights` are the neural network PARAMETERS (what we optimize during training)

When people say "Q-function" or "Q-network", they mean the neural network itself. When they say "Q-value", they mean the output number.

---

## 6) The Three Loss Functions

### Why do we need THREE loss functions?

In SAC we're learning THREE things simultaneously:
1. **Critic (Q-networks)** - estimate how good actions are
2. **Actor (Policy)** - choose good actions
3. **Temperature (α)** - balance exploration vs exploitation

Each needs its own loss function to train!

### 6.1) Critic Loss

**What it does:** Teaches the Q-network to accurately predict future returns

**Target (what we want Q to predict):**
```
y = r + γ * (1 - done) * [ min_i Q_target_i(s', a') - α * log π(a'|s') ]
```

Breaking it down:
- `r`: immediate reward (from environment)
- `γ * (next state value)`: discounted future value
- `min_i Q_target_i(s', a')`: minimum over target critics (pessimistic estimate)
- `- α * log π(a'|s')`: entropy bonus (encourages exploration)
- `(1 - done)`: if episode ended, no future rewards

**Loss:**
```
L_critic = mean[ (Q_i(s,a) - y)^2 ] for each critic i
```

This is just mean squared error (MSE) - we want our prediction Q_i(s,a) to match the target y.

**What it does in training:**
Adjusts the weights of the Q-network so its predictions get closer to the targets.

### 6.2) Actor Loss

**What it does:** Teaches the policy to choose actions that maximize Q-values and entropy

**The objective we want to MAXIMIZE:**
```
J = E[ Q(s,a) - α * log π(a|s) ]
```

This says: "Choose actions that have high Q-value but stay somewhat random (high entropy)"

**Loss (we minimize the negative):**
```
L_actor = E[ α * log π(a|s) - Q(s,a) ]
```

**How it works in practice:**
1. Sample state s from replay buffer
2. Sample action a from current policy π(·|s)
3. Evaluate Q(s,a) using critics (average over all critics)
4. Compute log π(a|s) (log probability of that action)
5. Compute loss = α * log π(a|s) - Q(s,a)
6. Backpropagate through the policy network

**What it does in training:**
Adjusts policy weights to favor high-Q actions while maintaining some randomness.

### 6.3) Temperature Loss

**What it does:** Automatically tunes α to keep policy entropy near a target

**Why?**
- If α too large → policy too random (poor performance)
- If α too small → policy too deterministic (poor exploration)
- We want to automatically find the sweet spot!

**Target entropy:**
Usually set to `-action_dim` (e.g., if action is 7D, target = -7)

**Loss:**
```
L_temperature = α * E[ -log π(a|s) - H_target ]
```

Where:
- `-log π(a|s)` is the entropy of the policy
- `H_target` is our target entropy

**What it does:**
- If policy entropy too low (deterministic) → increase α → more exploration
- If policy entropy too high (random) → decrease α → more exploitation

**Implementation note:**
In SERL, temperature is a Lagrange multiplier (constrained optimization), implemented as a small neural network with softplus activation to keep α positive.

---

## 7) Optimizers and make_optimizer

### What is make_optimizer?

**Short answer:** It's a wrapper that creates an Adam (or AdamW) optimizer with learning rate scheduling.

### Why not just use Adam directly?

You CAN! But `make_optimizer` adds useful features:
- **Warmup**: gradually increase learning rate from 0 to target over first N steps (stabilizes early training)
- **Cosine decay**: optionally decay learning rate toward 0 over training
- **Gradient clipping**: optionally clip gradients to prevent exploding gradients
- **Weight decay**: optionally add L2 regularization (AdamW)

**Code example (simplified):**
```python
def make_optimizer(learning_rate=3e-4, warmup_steps=2000):
    # Schedule: 0 → lr (over warmup) → lr (constant)
    schedule = warmup_then_constant(learning_rate, warmup_steps)
    
    # Use Adam optimizer with the schedule
    return optax.adam(learning_rate=schedule)
```

**Why this matters:**
- Warmup prevents unstable early updates when networks are randomly initialized
- Prevents using the wrong learning rate at the wrong time

**What optimizer does:**
Takes gradients (computed by backprop) and updates neural network weights:
```
weights_new = weights_old - learning_rate * gradients
```
(Adam is fancier with momentum and adaptive learning rates, but that's the core idea)

---

## 8) Ensemblize and N Critics

### What does ensemblize do?

**Function signature:**
```python
ensemblize(critic_class, num_qs=10)
```

**What it creates:**
N completely independent Q-networks with separate weights, all in one "vectorized" module.

**Concrete example:**
```python
# Without ensemble - single critic
critic = Critic(...)  # one Q-network

# With ensemble - N critics
critic_cls = partial(Critic, ...)
critic_ensemble = ensemblize(critic_cls, num_qs=10)  # 10 Q-networks!
```

### How many neural networks does this create?

**YES, it creates N separate neural networks!**

If `num_qs=10`:
- 10 separate Q-networks
- Each has its own weights (parameters)
- Each gets trained independently on the same data
- Each makes slightly different predictions due to random initialization

**Implementation (JAX magic):**
Instead of manually creating 10 networks, SERL uses `nn.vmap` which:
- Creates N copies of the network
- Stores weights in a stacked tensor with shape `(N, ...original_shape...)`
- Efficiently evaluates all N networks in parallel (vectorized on GPU)

**Why ensemble?**
- Reduces overestimation (we take min)
- More robust predictions (average of diverse networks)
- REDQ-style: can subsample a subset for each update (more updates per sample)

**Memory cost:**
If one critic has 1M parameters, ensemble of 10 has 10M parameters. But it's worth it for stability!

---

## 9) Pixel-Based vs State-Based Policies

### State-Based (Simple Case)

**Input:** Low-dimensional vector (e.g., `[joint_angle_1, joint_angle_2, ..., gripper_pos]`)
**Network:** Regular MLP (fully connected layers)

```python
state = [0.5, 1.2, -0.3, ...]  # 10D vector
↓
MLP (256 → 256 → action_dim)
↓
action = [torque_1, torque_2, ...]
```

**Policy network:** `MLP` with input size = state_dim, output = action mean & std

### Pixel-Based (Vision Case)

**Input:** Raw images (e.g., `(H, W, C)` where C=3 for RGB)
**Network:** CNN encoder + MLP head

```python
image = (128, 128, 3)  # RGB image
↓
CNN Encoder (ResNet or custom CNN)
↓
image_features = (256,)  # compressed representation
↓
MLP (256 → 256 → action_dim)
↓
action = [torque_1, torque_2, ...]
```

**Policy network:** `Encoder` (CNN) → `MLP` (fully connected)

### Mixed State + Images

**Input:** Both images and proprioceptive state (joint angles, forces, etc.)

**Two common approaches in SERL:**

**Approach 1: Late Fusion (used in SERL)**
```python
# Separate encoding
image = (128, 128, 3)
state = [0.5, 1.2, -0.3]  # joint angles

image → CNN → image_features (256,)
state → MLP → state_features (64,)

# Concatenate
combined = concat([image_features, state_features])  # (320,)

# Policy head
combined → MLP → action
```

**Approach 2: Early Fusion (alternative)**
```python
# Flatten and concatenate everything
image_flat = flatten(image)  # (49152,) ← huge!
combined = concat([image_flat, state])  # (49152 + state_dim,)

# Train MLP on everything
combined → MLP → action
```

**Why SERL uses Late Fusion:**
- CNN learns visual features efficiently (translation invariance, hierarchy)
- Flattening images loses spatial structure and creates huge input dimensions
- Separate encoding allows different learning rates for vision vs proprioception

**What create_pixels does:**
```python
create_pixels(
    encoder_def=ResNet(...),  # CNN for images
    use_proprio=True          # adds state_features after CNN
)
```

It builds:
1. CNN encoder for images
2. Small MLP for proprioceptive state (if `use_proprio=True`)
3. Concatenates encoded image + state
4. Feeds to policy/critic MLP heads

**Do we have 2 neural nets for policy?**

Yes! But they're components of ONE policy:
- Policy = Encoder + MLP_head
- Encoder is a CNN (one network)
- MLP head is fully connected (another network)
- Total: ONE policy network with two parts

Same for critic:
- Critic = Encoder + MLP_head
- May share encoder with policy or have separate encoder

---

## 10) UTD (Update-to-Data Ratio)

### What is UTD?

**Definition:**
```
UTD = (number of gradient updates) / (number of new environment samples)
```

**Examples:**
- UTD = 1: for every new sample from environment, do 1 gradient update (traditional)
- UTD = 20: for every new sample, do 20 gradient updates (high UTD)
- UTD = 0.5: for every 2 new samples, do 1 gradient update (low UTD)

### Why does UTD matter?

**Sample efficiency:**
Collecting environment samples is SLOW (especially on real robots). Gradient updates are FAST (GPU computations).

High UTD = squeeze more learning from each sample = fewer environment interactions needed!

### High UTD in SAC

**Problem:**
If you do too many actor updates per sample, the actor trains on the same data repeatedly → overfitting

**Solution (used in SERL):**
```python
utd_ratio = 20

# For each new batch:
for _ in range(utd_ratio):
    agent.update(batch, networks_to_update={'critic'})  # critic only
    
# After all critic updates:
agent.update(batch, networks_to_update={'actor', 'temperature'})  # actor once
```

**Why this works:**
- Critic learns from fixed policy data (safe to overfit temporarily)
- Actor updates only once (avoids overfitting)
- Ratio: 20 critic updates : 1 actor update

**What update_high_utd does:**
Implements exactly this! Splits batch into minibatches, does many critic updates, then one actor update.

**Practical numbers (from SERL):**
- Real robot training: UTD = 20-40 (sample collection is expensive)
- Simulation: UTD = 1-4 (samples are cheap)

---

## 11) Complete SAC Architecture Summary

### How many neural networks total?

Let's count for typical SAC setup:

**For state-based SAC:**
1. **Actor (policy)**: 1 MLP → outputs action distribution
2. **Critics (Q-networks)**: N MLPs (e.g., 2 or 10) → each outputs Q-value
3. **Target critics**: N MLPs (copies of critics, slowly updated)
4. **Temperature**: 1 tiny network (just learns a single scalar α)

**Total active networks: 1 + N + N + 1 = 2N + 2**

For N=2: **6 neural networks**
For N=10: **22 neural networks**

**For pixel-based SAC (DrQ):**
1. **Actor**: 1 CNN encoder + 1 MLP head
2. **Critics**: N × (CNN encoder + MLP head)
3. **Target critics**: N × (CNN encoder + MLP head)
4. **Temperature**: 1 tiny network

**Total: More networks due to encoders!**

### Memory layout example (N=2 critics):

```python
agent.state.params = {
    'actor': {
        'encoder': {...},  # CNN weights (if pixel-based)
        'network': {...},  # MLP weights
        'output': {...}    # action mean/std heads
    },
    'critic': {
        'ensemble': {      # shape (2, ...) for 2 critics
            'encoder': {...},
            'network': {...},
            'output': {...}
        }
    },
    'temperature': {
        'lagrange': [α]    # single scalar or small vector
    }
}

agent.state.target_params = {
    'critic': {
        'ensemble': {...}  # slowly updated copy of critics
    }
}
```

### Data flow in one update step:

```
1. Sample batch from replay buffer
   ↓
2. CRITIC UPDATE:
   - Compute next actions: a' ~ π(·|s')
   - Evaluate target critics: Q_target(s', a')
   - Compute target: y = r + γ * (min Q_target - α*log π)
   - Compute critic loss: MSE(Q(s,a), y)
   - Backprop through critic networks
   - Update critic weights
   ↓
3. ACTOR UPDATE:
   - Sample actions: a ~ π(·|s)
   - Evaluate critics: Q(s, a)
   - Compute actor loss: α*log π(a|s) - Q(s,a)
   - Backprop through actor network
   - Update actor weights
   ↓
4. TEMPERATURE UPDATE:
   - Compute entropy: -log π(a|s)
   - Compute temp loss: α * (entropy - H_target)
   - Update α
   ↓
5. SOFT TARGET UPDATE:
   - θ_target ← τ*θ + (1-τ)*θ_target
```

---

## 12) How SERL Implements It

### File structure:

```
serl_launcher/serl_launcher/
├── agents/continuous/
│   ├── sac.py          # Main SAC agent class
│   ├── drq.py          # SAC + image augmentation
│   ├── vice.py         # SAC + learned reward classifier
│   └── bc.py           # Behavior cloning (supervised)
├── networks/
│   ├── actor_critic_nets.py  # Policy and Critic modules
│   ├── lagrange.py            # Temperature (α) module
│   └── mlp.py                 # MLP building blocks
├── common/
│   ├── common.py       # JaxRLTrainState (manages params/optimizers)
│   ├── optimizers.py   # make_optimizer wrapper
│   └── encoding.py     # EncodingWrapper for pixels+proprio
└── utils/
    └── launcher.py     # make_sac_agent, make_drq_agent helpers
```

### Key classes:

**SACAgent** (`agents/continuous/sac.py`):
- Main agent class (Flax PyTreeNode)
- Methods: `update()`, `sample_actions()`, `forward_policy()`, `forward_critic()`
- Loss functions: `critic_loss_fn()`, `policy_loss_fn()`, `temperature_loss_fn()`

**Policy** (`networks/actor_critic_nets.py`):
- Actor network module
- Outputs a distribution (Gaussian with learned mean/std)
- Supports tanh squashing for bounded actions

**Critic** (`networks/actor_critic_nets.py`):
- Q-network module
- Input: state + action
- Output: Q-value (scalar)

**JaxRLTrainState** (`common/common.py`):
- Holds: params, target_params, optimizers, RNG
- Method `apply_loss_fns()` automates gradient computation + update

### Creating an agent:

```python
from serl_launcher.utils.launcher import make_sac_agent, make_drq_agent

# State-based SAC
agent = make_sac_agent(
    seed=0,
    sample_obs=sample_state_vector,
    sample_action=sample_action_vector,
    discount=0.99
)

# Pixel-based DrQ
agent = make_drq_agent(
    seed=0,
    sample_obs={'image': sample_image, 'state': sample_state},
    sample_action=sample_action_vector,
    image_keys=['image'],
    encoder_type='resnet',  # or 'small' CNN
    discount=0.96
)
```

### Training loop (simplified):

```python
# Initialize
agent = make_sac_agent(...)
replay_buffer = make_replay_buffer(env, capacity=1000000)

# Collect initial data
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    replay_buffer.insert(transition)

# Training
for step in range(num_steps):
    # Sample batch
    batch = replay_buffer.sample(batch_size=256)
    
    # Update agent
    agent, info = agent.update_high_utd(batch, utd_ratio=20)
    
    # Collect new data
    action = agent.sample_actions(obs, seed=rng)
    next_obs, reward, done, info = env.step(action)
    replay_buffer.insert((obs, action, reward, next_obs, done))
    obs = next_obs
```

### Where each component is used:

| Component | File | Purpose |
|-----------|------|---------|
| Policy network | `actor_critic_nets.Policy` | Choose actions |
| Critic networks | `actor_critic_nets.Critic` | Evaluate actions |
| Target critics | `JaxRLTrainState.target_params` | Stable targets |
| Temperature | `lagrange.GeqLagrangeMultiplier` | Auto-tune α |
| Optimizers | `optimizers.make_optimizer` | Update weights |
| Ensemble | `actor_critic_nets.ensemblize` | Create N critics |
| Replay buffer | `data.data_store.ReplayBuffer` | Store transitions |
| Image encoder | `vision.resnet_v1` or `vision.small_encoders` | Extract visual features |

---

## 13) Quick Reference: All Your Questions Answered

**Q: What is min_i Q_target_i(s', a')?**
A: Take the minimum Q-value across all target critic networks. Reduces overestimation bias.

**Q: If 2 critics, do we have 2 V and 2 Q functions?**
A: No! Just 2 Q functions. V is computed from Q when needed.

**Q: How do we know future rewards?**
A: We don't! We ESTIMATE with value functions trained using the Bellman equation.

**Q: Difference between reward and value function?**
A: Reward = immediate (from env). Value = estimated future total (learned by agent).

**Q: What are critic/actor/temperature losses?**
A: Three loss functions to train the three components of SAC (see section 6).

**Q: Are Q-values the neural network weights?**
A: No! Weights are parameters we optimize. Q-values are the network's outputs.

**Q: What is UTD?**
A: Update-to-Data ratio. How many gradient updates per environment sample.

**Q: How many neural networks?**
A: 2N+2 for N critics (actor + N critics + N target critics + temperature).

**Q: What does ensemblize create?**
A: N independent Q-networks with separate weights (vectorized for efficiency).

**Q: What is make_optimizer?**
A: Wrapper that creates Adam/AdamW with learning rate warmup and scheduling.

**Q: What is create_pixels?**
A: Helper that builds pixel-based SAC with CNN encoders + MLP heads.

**Q: Pixel-based policy = CNN?**
A: Yes! CNN encoder → MLP head. State-based = just MLP.

**Q: Mixed images + state?**
A: Late fusion: CNN for images, MLP for state, concatenate, then policy head.

---

## 14) Next Steps

1. **Read the code with this guide open**
   - Start with `agents/continuous/sac.py` functions: `critic_loss_fn`, `policy_loss_fn`
   - Check `networks/actor_critic_nets.py` to see Policy and Critic modules

2. **Run a simple experiment**
   - Use example scripts in `examples/async_drq_sim/`
   - Try modifying hyperparameters (critic_ensemble_size, UTD, temperature_init)

3. **Debug with fake data**
   - Create tiny state/action arrays
   - Call `agent.update()` and inspect `info` dict
   - Check shapes of `agent.state.params`

4. **Visualize**
   - Log Q-values during training
   - Plot entropy over time
   - Compare behavior with different α values

---

If you still have questions about ANY part, ask! I'll explain further or create code examples.
