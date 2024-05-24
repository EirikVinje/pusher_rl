#!/bin/bash -e

device="cuda"
epochs=50000
batch_size=64
memory=50000
run_name="run_step_300_2"
max_episode_steps=300

echo ""
echo "seed: $seed"
echo "device: $device"
echo "epochs: $epochs"
echo "batch_size: $batch_size"
echo "memory: $memory"
echo "run_name: $run_name"
echo "max_episode_steps: $max_episode_steps"
echo ""

python ddpg.py --device $device --epochs $epochs --run_name $run_name --batch_size $batch_size --memory $memory --max_episode_steps $max_episode_steps