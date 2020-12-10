#!/bin/bash
datatype=${1?Error: which dataset am I using?}
datapath=${2?Error: where is the dataset}
expdir=${3?Error: where to save the experiment}

python Train.py --dataset_path $datapath --dataset_type $datatype --version 0 --EntropyLossWeight 0 --lr 1e-4 --exp_dir $expdir







