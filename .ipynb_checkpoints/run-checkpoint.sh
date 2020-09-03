#!/bin/bash
python Train.py --dataset_path /project_scratch/bo/anomaly_data/ --dataset_type Avenue --dataset_augment_type training --dataset_augment_test_type testing --version 0 --EntropyLossWeight 0 --lr 1e-4 --exp_dir /project/bo/exp_data/memory_normal/







