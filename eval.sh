#!/bin/bash
trap "exit" INT
datatype=${1?Error: which data am I testing? Avenue, UCSDped2}
datapath=${2?Error: where are the dataset?}
version=${3?Error: experiment version is not defined}
ckpt_step=${4?Error: ckpt step is not defined}
exp_dir=${5?Error: where are the ckpt}
python Testing.py --dataset_type $datatype --dataset_path $datapath --version $version --EntropyLossWeight 0 --lr 1e-4 --exp_dir $exp_dir --ckpt_step $ckpt_step










