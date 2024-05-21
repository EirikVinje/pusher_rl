#!/bin/bash -e

seed=-1
device="cuda"
save_n=-1
epochs=30000
batch_size=64
render=0
max_episode_steps=100

run_name="run100_mem_25000"
memory=25000

python ddpg.py --seed $seed --device $device --epochs $epochs --save_n $save_n --run_name $run_name --batch_size $batch_size --memory $memory --render $render --max_episode_steps $max_episode_steps

run_name="run100_mem_50000"
memory=50000

python ddpg.py --seed $seed --device $device --epochs $epochs --save_n $save_n --run_name $run_name --batch_size $batch_size --memory $memory --render $render --max_episode_steps $max_episode_steps

run_name="run100_mem_75000"
memory=75000

python ddpg.py --seed $seed --device $device --epochs $epochs --save_n $save_n --run_name $run_name --batch_size $batch_size --memory $memory --render $render --max_episode_steps $max_episode_steps

