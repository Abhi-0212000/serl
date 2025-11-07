#!/bin/bash
# Tmux launcher for Trossen Bimanual Pick-and-Place Training
# This script creates a tmux session with learner and actor in separate windows

SESSION_NAME="trossen_training"

# Check if session already exists
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? == 0 ]; then
    echo "Session '$SESSION_NAME' already exists. Attaching..."
    tmux attach-session -t $SESSION_NAME
    exit 0
fi

# Create new session with learner
echo "Creating new training session: $SESSION_NAME"
tmux new-session -d -s $SESSION_NAME -n "learner"

# Send commands to learner window
tmux send-keys -t $SESSION_NAME:learner "conda activate serl" C-m
tmux send-keys -t $SESSION_NAME:learner "cd /home/qte9489/personal_abhi/serl/examples/async_sac_state_sim" C-m
tmux send-keys -t $SESSION_NAME:learner "echo 'Starting LEARNER in 3 seconds...'" C-m
tmux send-keys -t $SESSION_NAME:learner "sleep 3" C-m
tmux send-keys -t $SESSION_NAME:learner "./run_trossen_learner.sh" C-m

# Create new window for actor
tmux new-window -t $SESSION_NAME -n "actor"
tmux send-keys -t $SESSION_NAME:actor "conda activate serl" C-m
tmux send-keys -t $SESSION_NAME:actor "cd /home/qte9489/personal_abhi/serl/examples/async_sac_state_sim" C-m
tmux send-keys -t $SESSION_NAME:actor "echo 'Waiting for learner to start (10 seconds)...'" C-m
tmux send-keys -t $SESSION_NAME:actor "sleep 10" C-m
tmux send-keys -t $SESSION_NAME:actor "echo 'Starting ACTOR with rendering...'" C-m
tmux send-keys -t $SESSION_NAME:actor "./run_trossen_actor.sh" C-m

# Create monitoring window
tmux new-window -t $SESSION_NAME -n "monitor"
tmux send-keys -t $SESSION_NAME:monitor "conda activate serl" C-m
tmux send-keys -t $SESSION_NAME:monitor "cd /home/qte9489/personal_abhi/serl/examples/async_sac_state_sim" C-m
tmux send-keys -t $SESSION_NAME:monitor "echo ''" C-m
tmux send-keys -t $SESSION_NAME:monitor "echo '========================================'" C-m
tmux send-keys -t $SESSION_NAME:monitor "echo 'TROSSEN BIMANUAL TRAINING - MONITOR'" C-m
tmux send-keys -t $SESSION_NAME:monitor "echo '========================================'" C-m
tmux send-keys -t $SESSION_NAME:monitor "echo ''" C-m
tmux send-keys -t $SESSION_NAME:monitor "echo 'Tmux Controls:'" C-m
tmux send-keys -t $SESSION_NAME:monitor "echo '  Ctrl+b then 0  → Switch to learner window'" C-m
tmux send-keys -t $SESSION_NAME:monitor "echo '  Ctrl+b then 1  → Switch to actor window'" C-m
tmux send-keys -t $SESSION_NAME:monitor "echo '  Ctrl+b then 2  → Switch to monitor window'" C-m
tmux send-keys -t $SESSION_NAME:monitor "echo '  Ctrl+b then d  → Detach (training continues)'" C-m
tmux send-keys -t $SESSION_NAME:monitor "echo '  Ctrl+C        → Stop current process'" C-m
tmux send-keys -t $SESSION_NAME:monitor "echo ''" C-m
tmux send-keys -t $SESSION_NAME:monitor "echo 'To reattach: tmux attach -t $SESSION_NAME'" C-m
tmux send-keys -t $SESSION_NAME:monitor "echo 'To kill session: tmux kill-session -t $SESSION_NAME'" C-m
tmux send-keys -t $SESSION_NAME:monitor "echo ''" C-m
tmux send-keys -t $SESSION_NAME:monitor "echo 'Checkpoints saved to: ./checkpoints/trossen_bimanual/'" C-m
tmux send-keys -t $SESSION_NAME:monitor "echo ''" C-m

# Select learner window and attach
tmux select-window -t $SESSION_NAME:learner
tmux attach-session -t $SESSION_NAME

echo "Training session started!"
