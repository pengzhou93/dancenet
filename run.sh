#!/bin/bash


if [ $1 = demo ]
then
    
    source $HOME/anaconda3/bin/activate tf_1_9
    export LD_LIBRARY_PATH=/usr/local/cudnn-9.0-v7.1/lib64:/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH

    jupyter notebook --browser=google-chrome

fi
