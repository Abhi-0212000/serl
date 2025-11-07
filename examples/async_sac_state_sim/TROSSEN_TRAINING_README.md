# Trossen Bimanual Training Guide

## 🚀 Quick Start

### **Option 1: Automatic Launch (Recommended)**
```bash
cd /home/qte9489/personal_abhi/serl/examples/async_sac_state_sim
./tmux_launch_trossen.sh
```

This will:
- ✅ Start learner in one tmux window
- ✅ Start actor with rendering in another window
- ✅ Show you the monitor window with controls

**Tmux Controls:**
- `Ctrl+b` then `0` → Switch to learner window
- `Ctrl+b` then `1` → Switch to actor window  
- `Ctrl+b` then `2` → Switch to monitor window
- `Ctrl+b` then `d` → Detach (training continues in background)
- `Ctrl+C` → Stop current process

**To reattach:** `tmux attach -t trossen_training`  
**To kill:** `tmux kill-session -t trossen_training`

---

### **Option 2: Manual Launch (Two Terminals)**

**Terminal 1 - Learner:**
```bash
conda activate serl
cd /home/qte9489/personal_abhi/serl/examples/async_sac_state_sim
./run_trossen_learner.sh
```

**Terminal 2 - Actor (wait 10 seconds after learner starts):**
```bash
conda activate serl
cd /home/qte9489/personal_abhi/serl/examples/async_sac_state_sim
./run_trossen_actor.sh
```

---

## 📊 What You'll See

### **Learner Window Output:**
```
Filling up replay buffer: 100%|████████| 300/300
sent initial network to actor
learner:   0%|          | 0/1000000
  critic_loss: 0.1234
  actor_loss: -5.6789
  temperature: 0.1
  replay buffer: 350/1000000
```

### **Actor Window Output:**
```
actor_env: 100%|████████| 1000000/1000000
Step 330: cube_pos=[0.087, -0.053, 0.125]
  reward=0.245
  
eval: 100%|████████| 5/5
  average_return: 12.34
  success_rate: 0.2
```

### **Live Visualization:**
- The actor window will show a **MuJoCo viewer** with the Trossen dual-arm robot
- You'll see:
  - Both arms moving (left picks, right receives)
  - Red cube being manipulated
  - Table and workspace environment
- Press `Esc` or close window to stop actor (learner continues)

---

## 🎮 Rendering Options

### **With Rendering (Default):**
```bash
./run_trossen_actor.sh  # Has --render flag
```
- ✅ See what the robot is doing in real-time
- ✅ Great for debugging and visualization
- ❌ Slower training (~10-20 FPS)

### **Without Rendering (Faster Training):**
Edit `run_trossen_actor.sh` and change:
```bash
--render \          # Remove or change to:
--render false \
```
- ✅ Much faster training (~50-100 FPS)
- ❌ No visualization
- 💡 Use for overnight training runs

---

## 📁 Files and Outputs

### **Training Files:**
- `run_trossen_learner.sh` - Learner script
- `run_trossen_actor.sh` - Actor script  
- `tmux_launch_trossen.sh` - Automatic launcher
- `async_sac_state_sim.py` - Main training code (no changes needed!)

### **Outputs:**
- **Checkpoints:** `./checkpoints/trossen_bimanual/checkpoint_<step>`
  - Saved every 10,000 steps
  - Contains trained policy weights
  
- **Wandb Logs:** (if --debug removed)
  - Go to wandb.ai to see training curves
  - Reward, loss, success rate over time

---

## 🔧 Configuration

### **Training Parameters** (in `run_trossen_learner.sh`):
```bash
--max_steps 1000000        # Total training steps
--training_starts 300      # When to start training (after random exploration)
--random_steps 300         # Random exploration steps
--batch_size 256           # Samples per update
--critic_actor_ratio 8     # UTD ratio (8 critic updates per actor update)
--checkpoint_period 10000  # Save checkpoint every N steps
--eval_period 2000         # Run evaluation every N steps
```

### **Actor Parameters** (in `run_trossen_actor.sh`):
```bash
--steps_per_update 30      # Send data to learner every 30 steps
--eval_n_trajs 5          # Number of evaluation episodes
--render                   # Enable/disable visualization
```

---

## 📈 Monitoring Training

### **What to Watch:**

1. **Reward (Actor Output):**
   - Should increase over time
   - Random policy: ~0.0-0.1
   - Trained policy: ~5-15 (depending on task)

2. **Success Rate (Eval Output):**
   - Percentage of successful pick-and-place episodes
   - Target: >80% after sufficient training

3. **Cube Height:**
   - Watch `cube_z` value increase
   - Initial: 0.0125 (on table)
   - Lifted: >0.15 (15cm above table)

4. **Losses (Learner Output):**
   - `critic_loss`: Should decrease and stabilize
   - `actor_loss`: Negative value, magnitude decreases
   - `temperature`: Auto-adjusts for exploration

---

## 🐛 Troubleshooting

### **"Environment not found" Error:**
```bash
# Make sure trossen_sim is installed:
conda activate serl
cd /home/qte9489/personal_abhi/serl/trossen_sim
pip install -e .
```

### **Actor can't connect to Learner:**
- Make sure learner starts FIRST (wait 10 seconds)
- Check `--ip localhost` is correct
- Both must use same `--exp_name`

### **Rendering is slow:**
- Set `--render false` for faster training
- Or reduce `control_dt` in environment

### **Training not improving:**
- Check reward function is giving signal
- Increase `--training_starts` for more exploration
- Check action/observation spaces are correct

---

## 🎯 Expected Timeline

- **Steps 0-300:** Random exploration (filling replay buffer)
- **Steps 300-1000:** Initial learning (reward starts increasing)
- **Steps 1000-5000:** Policy improves, some successful grasps
- **Steps 5000-20000:** Reliable grasping, learning handover
- **Steps 20000-50000:** Smooth bimanual coordination
- **Steps 50000+:** Fine-tuning and optimization

**Typical training time:**
- 100k steps ≈ 2-4 hours (with rendering)
- 100k steps ≈ 30-60 minutes (without rendering)

---

## 💾 Loading Trained Policy

To test a trained policy:

```python
import gym
import trossen_sim
from serl_launcher.agents.continuous.sac import SACAgent

# Load environment
env = gym.make('TrossenBimanualPickPlace-v0', render_mode='human')

# Load checkpoint
agent = SACAgent.load('./checkpoints/trossen_bimanual/checkpoint_50000')

# Run policy
obs, _ = env.reset()
for _ in range(1000):
    action = agent.sample_actions(obs['state'], argmax=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done or truncated:
        obs, _ = env.reset()
```

---

## 📚 References

- **SERL Paper:** [Link to paper if available]
- **SAC Algorithm:** Soft Actor-Critic ([arXiv:1801.01290](https://arxiv.org/abs/1801.01290))
- **Training Guide:** `/home/qte9489/personal_abhi/serl/examples/async_sac_state_sim/TRAINING_GUIDE.md`

---

## ✅ Summary

**To start training RIGHT NOW:**

```bash
cd /home/qte9489/personal_abhi/serl/examples/async_sac_state_sim
./tmux_launch_trossen.sh
```

**Then:**
1. Watch the learner window for training progress
2. Switch to actor window (`Ctrl+b` then `1`) to see the robot
3. Detach with `Ctrl+b` then `d` to let it train in background
4. Come back later to check progress!

**That's it!** 🚀
