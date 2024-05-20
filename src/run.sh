#!/bin/bash -e

seed=-1
device="cuda"
epochs=50000
save_n=20000
run_name="max_ep_100"
batch_size=64
memory=50000
render=0
max_episode_steps=100

echo ""
echo "seed: $seed"
echo "device: $device"
echo "epochs: $epochs"
echo "save_n: $save_n"
echo "run_name: $run_name"
echo "batch_size: $batch_size"
echo "memory: $memory"
echo "render: $render"
echo "max_episode_steps: $max_episode_steps"
echo ""

python ddpg.py --seed $seed --device $device --epochs $epochs --save_n $save_n --run_name $run_name --batch_size $batch_size --memory $memory --render $render --max_episode_steps $max_episode_steps
echo "Done!"

# seed=-1
# device="cuda"
# epochs=100000
# save_n=10000
# run_name="max_ep_200"
# batch_size=64
# memory=10000
# render=0
# max_episode_steps=200

# echo ""
# echo "seed: $seed"
# echo "device: $device"
# echo "epochs: $epochs"
# echo "save_n: $save_n"
# echo "run_name: $run_name"
# echo "batch_size: $batch_size"
# echo "memory: $memory"
# echo "render: $render"
# echo "max_episode_steps: $max_episode_steps"
# echo ""

# python dqn.py --seed $seed --device $device --epochs $epochs --save_n $save_n --run_name $run_name --batch_size $batch_size --memory $memory --render $render --max_episode_steps $max_episode_steps
# echo "Done!"