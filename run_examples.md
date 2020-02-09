# Run Examples

# run_classifier

python run_classification.py --data_dir ../datasets/DocRED/ --model_type roberta-rc --model_name_or_path roberta-large --task_name docred --output_dir classification_outputs/rrr/xx_yyy --do_train --do_eval --do_full_eval --relation_name rrr --num_positive_examples xx --num_negative_examples yyy --num_train_epochs 200 --patience 10 --evaluate_during_training --logging_steps 100 --fp16 --gradient_accumulation_steps 5 > log_rrr_xx_yyy.txt 2>&1
