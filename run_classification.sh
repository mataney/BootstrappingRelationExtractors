#!/bin/bash
# This should be in the home directory

start=`date +%s`
# activate some conda environment
source activate hugging_face

# change working dir
cd $WORKING_DIR

# Set all these before running
relation_name=$RELATION_NAME
num_positive_examples=$NUM_POSITIVE_EXAMPLES
ratio_negative_examples=$RATIO_NEGATIVE_EXAMPLES
logging_steps=$LOGGING_STEPS
training_method=$TRAINING_METHOD
num_train_epochs=$NUM_TRAIN_EPOCHS
seed=$SEED
task=$TASK
log_dir=$LOG_DIR

if [[ $seed = null ]]; then seed=1; fi
if [[ $logging_steps = null ]]; then logging_steps=100; fi
if [[ $num_train_epochs = null ]]; then num_train_epochs=500; fi

if [[ $training_method = null ]]; then training_method="train"; fi

if [[ $task = "docred" ]]
then
  data_dir="data/DocRED/"
  dev_file="eval_split_from_annotated.json"
  test_file="dev.json"
elif [[ $task = "tacred" ]]
then
  data_dir="data/tacred/"
  dev_file="dev.json"
  test_file="test.json"
else
  echo "Wrong task"
fi

output_dir=classification_outputs/$relation_name/$training_method/"$num_positive_examples"_"$ratio_negative_examples"

python run_classification.py \
  --data_dir $data_dir \
  --model_type roberta-rc \
  --model_name_or_path roberta-large \
  --task_name $task \
  --output_dir $output_dir \
  --training_method $training_method \
  --do_full_dev_eval \
  --do_full_test_eval \
  --evaluate_during_training \
  --patience 8 \
  --relation_name $relation_name \
  --num_positive_examples $num_positive_examples \
  --ratio_negative_examples $ratio_negative_examples \
  --num_train_epochs $num_train_epochs \
  --fp16 \
  --logging_steps $logging_steps \
  --save_steps $logging_steps \
  --save_only_best \
  --warmup_steps 100 \
  --per_gpu_train_batch_size 8 \
  --learning_rate 2e-5 \
  --seed $seed \
  --gradient_accumulation_steps 5 > log_"$relation_name"_"$num_positive_examples"_"$ratio_negative_examples".txt 2>&1

python -m scripts.check_num_of_examples $data_dir $log_dir/num_examples.json

python -m classification.evaluation."$task"_evaluation --gold_dir $data_dir --gold_file $dev_file --relation_name $relation_name --pred_file "$output_dir/full_dev_eval_results.json" --output_file "$output_dir/full_dev_eval_scores.json"

confidence_threshold_on_dev_eval=$(jq -r ".best_confidence" "$output_dir/full_dev_eval_scores.json")

python -m classification.evaluation."$task"_evaluation --gold_dir $data_dir --gold_file $test_file --relation_name $relation_name --pred_file "$output_dir/full_test_eval_results.json" --confidence_threshold $confidence_threshold_on_dev_eval --output_file "$output_dir/full_test_eval_scores.json"

jq -n --slurpfile dev_eval_content "$output_dir/full_dev_eval_results.json" \
  --slurpfile test_eval_content "$output_dir/full_test_eval_results.json" \
  --slurpfile dev_eval_scores "$output_dir/full_dev_eval_scores.json" \
  --slurpfile test_eval_scores "$output_dir/full_test_eval_scores.json" \
  '{
    test_F1:$test_eval_scores[0].F1,
    test_precision:$test_eval_scores[0].precision,
    test_recall:$test_eval_scores[0].recall,
    confidence:$test_eval_scores[0].best_confidence,
    dev_F1:$dev_eval_scores[0].F1,
    dev_precision:$dev_eval_scores[0].precision,
    dev_recall:$dev_eval_scores[0].recall,
    test_eval:$test_eval_scores,
    dev_eval:$dev_eval_scores,
    full_test_eval_results:$test_eval_content,
    full_dev_eval_results:$dev_eval_content
    }' \
  > "$log_dir/full_results.json"


end=`date +%s`
secs=$((end-start))
time="$(($secs/3600))h$(($secs%3600/60))m$(($secs%60))s"

jq -n --arg time $time \
  --slurpfile dev_eval_scores "$output_dir/full_dev_eval_scores.json" \
  --slurpfile test_eval_scores "$output_dir/full_test_eval_scores.json" \
  --slurpfile num_examples "$log_dir/num_examples.json" \
  '{
    test_F1:$test_eval_scores[0].F1,
    test_precision:$test_eval_scores[0].precision,
    test_recall:$test_eval_scores[0].recall,
    confidence:$dev_eval_scores[0].best_confidence,
    dev_F1:$dev_eval_scores[0].F1,
    dev_precision:$dev_eval_scores[0].precision,
    dev_recall:$dev_eval_scores[0].recall,
    num_examples:$num_examples,
    time:$time
    }' \
  > "$log_dir/output"
