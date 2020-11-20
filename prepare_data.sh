#!/bin/bash
trap "exit" INT
ds=${1?Error: which datasets am I loading? Avenue, UCSDped2}
datapath=${2?Error: where do you want to save the data, default: frames/}

expdir=$(pwd)

if [ $ds = Avenue ]; then
    datapath=$datapath/$ds/
    if [ -d "${datapath}" ]; then
        echo "$ds dataset exists"
        trainfolder=${datapath}training/
        if [ -d "$trainfolder" ]; then
            echo "Frames already exist, YEAH!"
        else
            echo "Extract frames"
            python3 video.py --dataset $ds --datapath $datapath
        fi        
    else
        echo "Download the Avenue dataset...."
        mkdir $datapath
        cd $datapath
        wget http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/Avenue_Dataset.zip
        unzip Avenue_Dataset.zip
        mv 'Avenue Dataset'/* .
        echo "Successfully download the dataset, next extract frames from the Avenue dataset"
        cd $expdir
        python3 video.py --dataset $ds --datapath $datapath 
    fi
elif [ $ds = UCSDped2 ]; then
    datapathsub=$datapath/$ds/
    if [ -d "${datapathsub}" ]; then
        echo "$ds dataset exists"
        trainfolder=${datapathsub}Train_jpg/
        if [ -d "$trainfolder" ]; then
            echo "Frames already exist, YEAH!"
        else
            echo "Extract frames"
            python3 video.py --dataset $ds --datapath $datapathsub
        fi        
    else
        echo "Download the UCSD dataset............."
        cd $datapath
        wget http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz
        tar -xvf UCSD_Anomaly_Dataset.tar.gz
        mv UCSD_Anomaly_Dataset.v1p2/* .
        rm -rf UCSD_Anomaly_Dataset.v1p2
        echo "Successfully download the dataset, next extract frames from the UCSD dataset"
        cd $expdir
        python3 video.py --dataset $ds --datapath $datapathsub
    fi
fi