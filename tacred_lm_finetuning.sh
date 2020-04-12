#!/bin/bash
local_test=0

while getopts o:t option
do
case "${option}"
in
o) output_dir=${OPTARG};;
t) local_test=1;;
esac
done

export PYTHONPATH=$PYTHONPATH:../transformers

if [ $local_test == "1" ];then
    python run_lm_finetuning.py \
        --output_dir=$output_dir \
        --model_type=gpt2 \
        --model_name_or_path=gpt2 \
        --do_train \
        --train_data_file=data/tacred/for_generation/train.src \
        --special_tokens_file=data/tacred/for_generation/train.special_tokens \
        --block_size=512 \
        --per_gpu_train_batch_size=2 \
        --num_train_epochs=2 \
        --save_steps=1000
else
    python run_lm_finetuning.py \
        --output_dir=$output_dir \
        --model_type=gpt2 \
        --model_name_or_path=gpt2-medium \
        --do_train \
        --train_data_file=data/tacred/for_generation/train.src \
        --special_tokens_file=data/tacred/for_generation/train.special_tokens \
        --block_size=512 \
        --per_gpu_train_batch_size=2 \
        --num_train_epochs=10 \
        --save_steps=1000 \
        --fp16
fi