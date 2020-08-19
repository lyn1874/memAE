#!/bin/bash
python Train.py --dataset_path /project_scratch/bo/anomaly_data/ --dataset_type Avenue --dataset_augment_type training --dataset_augment_test_type testing --version 0 --EntropyLossWeight 0 --lr 1e-4 --exp_dir /project/bo/exp_data/memory_normal/

python Train.py --dataset_path /project_scratch/bo/anomaly_data/ --dataset_type Avenue --dataset_augment_type training --dataset_augment_test_type testing --version 1 --EntropyLossWeight 5e-5 --lr 1e-4 --exp_dir /project/bo/exp_data/memory_normal/

python Train.py --dataset_path /project_scratch/bo/anomaly_data/ --dataset_type Avenue --dataset_augment_type training --dataset_augment_test_type testing --version 1 --EntropyLossWeight 1e-5 --lr 1e-4 --exp_dir /project/bo/exp_data/memory_normal/

# python Train.py --dataset_path /project_scratch/bo/anomaly_data/ --dataset_type Avenue --dataset_augment_type training --dataset_augment_type_test testing --version 1 --EntropyLossWeight 5e-4 --lr 1e-4

# python Train.py --dataset_path /project_scratch/bo/anomaly_data/ --dataset_type Avenue --dataset_augment_type training --dataset_augment_type_test testing --version 1 --EntropyLossWeight 2e-4 --lr 1e-4

# python Train.py --dataset_path /project_scratch/bo/anomaly_data/ --dataset_type UCSDped2 --dataset_augment_type original --exp_dir /project/bo/exp_data/memory_normal/ --version 1 --EntropyLossWeight 0.00005 --lr 1e-4 








