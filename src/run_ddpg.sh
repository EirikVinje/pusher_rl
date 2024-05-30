#!/bin/bash -e

device="cuda"
record=0
epochs=1_000_000
batch_size=32
memory=10000
run_name="r200"
max_episode_steps=200

echo ""
echo "seed: $seed"
echo "device: $device"
echo "record: $record"
echo "epochs: $epochs"
echo "batch_size: $batch_size"
echo "memory: $memory"
echo "run_name: $run_name"
echo "max_episode_steps: $max_episode_steps"
echo ""

python ddpg.py --device $device --epochs $epochs --run_name $run_name --batch_size $batch_size --memory $memory --max_episode_steps $max_episode_steps --record $record