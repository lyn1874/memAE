#!/bin/bash
trap "exit" INT
version=${1?Error: experiment version is not defined}
ckpt_step=${2?Error: ckpt step is not defined}
python Testing.py --dataset_type Avenue --dataset_path /tmp/bo/ --dataset_augment_test_type frames/testing/ --version $version --EntropyLossWeight 0 --lr 1e-4 --exp_dir /project/bo/exp_data/memory_normal/ --ckpt_step $ckpt_step










