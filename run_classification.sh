#!/bin/bash
# This should be in the home directory

start=`date +%s`
# activate some conda environment
source activate hugging_face

# change working dir
cd matan/dev/relation_generation_using_gpt2

relation_name=$(jq -r ".relation_name" "$OTO_INPUT")
num_positive_examples=$(jq ".num_positive_examples" "$OTO_INPUT")
ratio_negative_examples=$(jq ".ratio_negative_examples" "$OTO_INPUT")
logging_steps=$(jq ".logging_steps" "$OTO_INPUT")
seed=$(jq ".seed" "$OTO_INPUT")
data_dir="data/DocRED/"

if [[ $seed = null ]]; then seed=1; fi
if [[ $logging_steps = null ]]; then logging_steps=100; fi

training_method=$(jq -r ".training_method" "$OTO_INPUT")
if [[ $training_method = null ]]; then training_method="annotated"; fi
if [[ $training_method = "annotated" ]]
then
  do_train_type='--do_train'
elif [[ $training_method = "distant" ]]
then
  do_train_type='--do_distant_train'
else
  echo "Wrong training method"
fi

output_dir=classification_outputs/$relation_name/$training_method/"$num_positive_examples"_"$ratio_negative_examples"

python run_classification.py \
  --data_dir $data_dir \
  --model_type roberta-rc \
  --model_name_or_path roberta-large \
  --task_name docred \
  --output_dir $output_dir \
  "$do_train_type" \
  --do_eval_train_eval \
  --do_full_train_eval \
  --do_full_dev_eval \
  --evaluate_during_training \
  --patience 6 \
  --relation_name $relation_name \
  --num_positive_examples $num_positive_examples \
  --ratio_negative_examples $ratio_negative_examples \
  --type_independent_neg_sample \
  --num_train_epochs 200 \
  --logging_steps $logging_steps \
  --fp16 \
  --warmup_steps 100 \
  --per_gpu_train_batch_size 8 \
  --learning_rate 2e-5 \
  --seed $seed \
  --gradient_accumulation_steps 5 > log_"$relation_name"_"$num_positive_examples"_"$ratio_negative_examples".txt 2>&1

python -m scripts.check_num_of_examples $data_dir $OTO_BACKUP/num_examples.json

python -m classification.evaluation.evaluation --gold_dir data/DocRED --gold_file eval_split_from_annotated.json --relation_name $relation_name --pred_file "$output_dir/full_train_eval_results.json" --confidence_threshold 0 --output_file "$output_dir/full_train_eval_scores.json"

confidence_threshold_on_train_eval=$(jq -r ".best_confidence" "$output_dir/full_train_eval_scores.json")

python -m classification.evaluation.evaluation --gold_dir data/DocRED --gold_file dev.json --relation_name $relation_name --pred_file "$output_dir/full_dev_eval_results.json" --confidence_threshold $confidence_threshold_on_train_eval --output_file "$output_dir/full_dev_eval_scores.json"

jq -n --slurpfile train_eval_content "$output_dir/full_train_eval_results.json" \
  --slurpfile dev_eval_content "$output_dir/full_dev_eval_results.json" \
  --slurpfile train_eval_scores "$output_dir/full_train_eval_scores.json" \
  --slurpfile dev_eval_scores "$output_dir/full_dev_eval_scores.json" \
  '{
    dev_F1:$dev_eval_scores[0].F1,
    dev_precision:$dev_eval_scores[0].precision,
    dev_recall:$dev_eval_scores[0].recall,
    confidence:$dev_eval_scores[0].best_confidence,
    train_F1:$train_eval_scores[0].F1,
    train_precision:$train_eval_scores[0].precision,
    train_recall:$train_eval_scores[0].recall,
    dev_eval:$dev_eval_scores,
    train_eval:$train_eval_scores,
    full_dev_eval_results:$dev_eval_content,
    full_train_eval_results:$train_eval_content
    }' \
  > "$OTO_BACKUP/full_results.json"


end=`date +%s`
secs=$((end-start))
time="$(($secs/3600))h$(($secs%3600/60))m$(($secs%60))s"

jq -n --arg time $time \
  --slurpfile train_eval_scores "$output_dir/full_train_eval_scores.json" \
  --slurpfile dev_eval_scores "$output_dir/full_dev_eval_scores.json" \
  --slurpfile num_examples "$OTO_BACKUP/num_examples.json" \
  '{
    dev_F1:$dev_eval_scores[0].F1,
    dev_precision:$dev_eval_scores[0].precision,
    dev_recall:$dev_eval_scores[0].recall,
    confidence:$dev_eval_scores[0].best_confidence,
    train_F1:$train_eval_scores[0].F1,
    train_precision:$train_eval_scores[0].precision,
    train_recall:$train_eval_scores[0].recall,
    num_examples:$num_examples,
    time:$time
    }' \
  > "$OTO_OUTPUT"