#!/bin/bash

python run_glue.py \
  --model_type roberta-rc \
  --model_name_or_path roberta-large \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir ../datasets/glue/glue_data/MRPC/ \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/mrpc_output/