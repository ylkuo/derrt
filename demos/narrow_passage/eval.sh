#!/bin/bash

mode=$1  # all (train and test) or test (load model for testing)
model=$2  # none, rnn
n_test_rounds=$3
gpu=$4  # use gpu if >= 0

mode_flag=""
model_flag=""
gpu_flag=""


if [ $mode = "test" ]; then
    mode_flag="--load_model"
fi

if [ $model = "rnn" ]; then
    model_flag="--use_rnn"
fi

if (( $gpu >= 0 )); then
    gpu_flag="--use_cuda --device_id $gpu"
fi


python -m demos.narrow_passage.narrow_passage \
    --n_test_rounds $n_test_rounds \
    $mode_flag $model_flag $gpu_flag
