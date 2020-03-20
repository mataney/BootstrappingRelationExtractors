#!/bin/bash
source activate hugging_face

python run_classification.py \
    --data_dir data/tacred \
    --model_type roberta-rc \
    --model_name_or_path roberta-large \
    --task_name tacred \
    --output_dir classification_outputs/multi_class_tacred_new_hp \
    --do_multi_class_train \
    --do_multi_class_eval_dev \
    --do_multi_class_eval_test \
    --evaluate_during_training \
    --patience 10 \
    --relation_name all \
    --num_positive_examples -1 \
    --ratio_negative_examples -1 \
    --type_independent_neg_sample \
    --num_train_epochs 10 \
    --fp16 \
    --logging_steps 1000 \
    --save_steps 1000 \
    --save_only_best \
    --warmup_steps 1000 \
    --per_gpu_train_batch_size 8 \
    --learning_rate 3e-5 \
    --seed 123 \
    --max_seq_length 256 \
    --gradient_accumulation_steps 8 > log_multi_class_tacred.txt 2>&1