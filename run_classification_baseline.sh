#!/bin/bash
# This should be in the home directory

# activate some conda environment
source activate hugging_face

# change working dir
cd matan/dev/relation_generation_using_gpt2

relation_name=$(jq -r ".relation_name" "$OTO_INPUT")
num_positive_examples=$(jq ".num_positive_examples" "$OTO_INPUT")
num_negative_examples=$(jq ".num_negative_examples" "$OTO_INPUT")
output_dir=classification_outputs/$relation_name/"$num_positive_examples"_"$num_negative_examples"

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
    --relation_name $relation_name \
    --num_positive_examples $num_positive_examples \
    --num_negative_examples $num_negative_examples \
    --type_independent_neg_sample \
    --num_train_epochs 200 \
    --logging_steps 100 \
    --fp16 \
    --warmup_steps 100 \
    --per_gpu_train_batch_size 8 \
    --learning_rate 2e-5 \
    --seed 100 \
    --gradient_accumulation_steps 5 > log_"$relation_name"_"$num_positive_examples"_"$num_negative_examples".txt 2>&1

python -m classification.evaluation.evaluation --gold_dir data/DocRED --gold_file eval_split_from_annotated.json --relation_name $relation_name --pred_file "$output_dir/full_train_eval_results.json" --confidence_threshold 0 --output_file "$output_dir/full_train_eval_scores.json"

confidence_threshold_on_train_eval=$(jq -r ".best_confidence" "$output_dir/full_train_eval_scores.json")

python -m classification.evaluation.evaluation --gold_dir data/DocRED --gold_file eval_split_from_annotated.json --relation_name $relation_name --pred_file "$output_dir/full_dev_eval_results.json" --confidence_threshold $confidence_threshold_on_train_eval --output_file "$output_dir/full_dev_eval_scores.json"

jq -n --slurpfile train_eval_content "$output_dir/full_train_eval_results.json" \
      --slurpfile dev_eval_content "$output_dir/full_dev_eval_results.json" \
      --slurpfile train_eval_scores "$output_dir/full_train_eval_scores.json" \
      --slurpfile dev_eval_scores "$output_dir/full_dev_eval_scores.json" \
      '{
          dev_F1:$dev_eval_scores[0].F1,
          dev_precision:$dev_eval_scores[0].precision,
          dev_recall:$dev_eval_scores[0].recall,
          confidence:$dev_eval_scores[0].best_confidence,
          dev_eval:$dev_eval_scores,
          train_eval:$train_eval_scores,
          full_dev_eval_results:$dev_eval_content,
          full_train_eval_results:$train_eval_content
        }' \
      > "$OTO_OUTPUT"