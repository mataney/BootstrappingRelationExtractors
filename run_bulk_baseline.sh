#!/usr/bin/env bash
# This should be in the home directory

# activate some conda environment
source activate hugging_face

# change working dir
cd matan/dev/relation_generation_using_gpt2

relation_name=$(jq ".relation_name" "$OTO_INPUT")
num_positive_examples=$(jq ".num_positive_examples" "$OTO_INPUT")
num_negative_examples=$(jq ".num_negative_examples" "$OTO_INPUT")
output_dir=classification_outputs/"$relation_name"/"$num_positive_examples"_"$num_negative_examples"

python run_classification.py \
    --data_dir data/DocRED/ \
    --model_type roberta-rc \
    --model_name_or_path roberta-large \
    --task_name docred \
    --output_dir $output_dir \
    --do_train \
    --do_eval_train_eval \
    --do_full_train_eval \
    --do_full_dev_eval \
    --evaluate_during_training \
    --patience 5 \
    --relation_name "$relation_name" \
    --num_positive_examples "$num_positive_examples" \
    --num_negative_examples "$num_negative_examples" \
    --type_independent_neg_sample \
    --num_train_epochs 200 \
    --logging_steps 100 \
    --fp16 \
    --warmup_steps 100 \
    --per_gpu_train_batch_size 8 \
    --learning_rate 2e-5 \
    --seed 100 \
    --gradient_accumulation_steps 5 > log_"$relation_name"_"$num_positive_examples"_"$num_negative_examples".txt 2>&1

jq -n --slurpfile train_eval_content "$output_dir/full_train_eval_results" \
      --slurpfile dev_eval_content "$output_dir/full_dev_eval_results" \
      '{"full_dev_eval_results":$dev_eval_content, "full_train_eval_results":$train_eval_content}' \
      > "$OTO_OUTPUT"