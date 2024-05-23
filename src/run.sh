#!/bin/bash -e

device="cuda"
save_n=5000
epochs=40000
batch_size=64
render=0

echo ""
echo "seed: $seed"
echo "device: $device"
echo "epochs: $epochs"
echo "save_n: $save_n"
echo "batch_size: $batch_size"
echo "memory: $memory"
echo "render: $render"

run_name="run_step_100"
max_episode_steps=100
memory=10000

echo "run_name: $run_name"
echo "max_episode_steps: $max_episode_steps"
echo ""

python ddpg.py --device $device --epochs $epochs --save_n $save_n --run_name $run_name --batch_size $batch_size --memory $memory --render $render --max_episode_steps $max_episode_steps

run_name="run_step_200"
max_episode_steps=200
memory=25000

echo ""
echo "run_name: $run_name"
echo "max_episode_steps: $max_episode_steps"

python ddpg.py --device $device --epochs $epochs --save_n $save_n --run_name $run_name --batch_size $batch_size --memory $memory --render $render --max_episode_steps $max_episode_steps
