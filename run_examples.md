# Run Examples

# run_classifier

python run_classification.py --data_dir ../datasets/DocRED/ --model_type roberta-rc --model_name_or_path roberta-large --task_name docred --output_dir classification_outputs/mother --do_train --do_eval --relation_name mother --num_positive_examples 10 --num_negative_examples 100 --num_train_epochs 10