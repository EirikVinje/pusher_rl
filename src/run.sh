#!/bin/bash -e

seed=50
device="cuda"
epochs=10_000
save_n=1_000
run_name="run_1"
batch_size=32

python dqn.py --seed $seed --device $device --epochs $epochs --save_n $save_n --run_name $run_name --batch_size $batch_size