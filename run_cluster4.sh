#!/bin/bash
dataset=Avenue
expdir=/tmp/bo/$dataset
if [ -d "$expdir" ]; then
    echo "folder $expdir already exists"
    echo "next step, download the statistics to reproduce figures"
else
    mkdir /tmp/bo/
    mkdir $expdir
    echo "make folder $expdir for saving statistics"
    cp -r /project/bo/anomaly_data/$dataset /tmp/bo/
fi
# python Train.py --dataset_path /tmp/bo/ --dataset_type UCSDped2 --dataset_augment_type original --exp_dir /project/bo/exp_data/memory_normal/ --version 3 --EntropyLossWeight 0 --lr 1e-4 --batch_size 4  --> this is the one that makes it work!

python Train.py --dataset_path /tmp/bo/ --dataset_type Avenue --dataset_augment_type training --dataset_augment_test_type testing --version 1 --EntropyLossWeight 5e-4 --lr 1e-4 --exp_dir /project/bo/exp_data/memory_normal/ --batch_size 4

python Train.py --dataset_path /tmp/bo/ --dataset_type Avenue --dataset_augment_type training --dataset_augment_test_type testing --version 1 --EntropyLossWeight 2e-4 --lr 1e-4 --exp_dir /project/bo/exp_data/memory_normal/ --batch_size 4










