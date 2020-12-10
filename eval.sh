#!/bin/bash
trap "exit" INT
datatype=${1?Error: which data am I testing? Avenue, UCSDped2}
datapath=${2?Error: where are the dataset?}
version=${3?Error: experiment version is not defined}
ckptstep=${4?Error: model ckpt step is not defined}
expdir=${5?Error: the model path is not defined}

python Testing.py --dataset_type $datatype --dataset_path $datapath --version $version --EntropyLossWeight 0 --lr 1e-4 --exp_dir $expdir --ckpt_step $ckptstep










