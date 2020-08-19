#!/bin/bash
python Testing.py --dataset UCSDped2 --augment_type original --version 1 --lr 1e-4 --EntropyLossWeight 5e-5 --ckpt_step 30


# python Train.py --dataset_path /project_scratch/bo/anomaly_data/ --dataset_type UCSDped2 --dataset_augment_type original --exp_dir /project/bo/exp_data/memory_normal/ --version 1 --EntropyLossWeight 0.00005 --lr 1e-4 








