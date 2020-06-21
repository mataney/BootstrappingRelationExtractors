#!/bin/bash
source activate hugging_face

cd matan/dev/relation_generation_using_gpt2

model_type=$(jq -r ".model_type" "$OTO_INPUT")
model_name_or_path=$(jq -r ".model_name_or_path" "$OTO_INPUT")
thin_classifier=$(jq -r ".thin_classifier" "$OTO_INPUT")
export THIN_CLASSIFIER=$thin_classifier

output_dir="classification_outputs/multi_class_tacred_"$model_type"_thin_classifier_"$thin_classifier

python run_classification.py \
    --data_dir data/tacred \
    --model_type $model_type \
    --model_name_or_path $model_name_or_path \
    --task_name tacred \
    --output_dir $output_dir \
    --do_multi_class_train \
    --do_multi_class_eval_dev \
    --do_multi_class_eval_test \
    --evaluate_during_training \
    --patience 10000 \
    --relation_name all \
    --num_positive_examples -1 \
    --ratio_negative_examples -1 \
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


python classification/evaluation/original_tacred_eval.py data/tacred/dev.json $output_dir"/dev_multi_class_results.json" > dev_results.txt
 
python classification/evaluation/original_tacred_eval.py data/tacred/test.json $output_dir"/test_multi_class_results.json" > test_results.txt

dev_results=`cat dev_results.txt`
test_results=`cat test_results.txt`

echo $dev_results > $OTO_OUTPUT
echo $test_results >> $OTO_OUTPUT