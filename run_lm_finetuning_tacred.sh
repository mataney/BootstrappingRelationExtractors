while getopts o:t: option
do
case "${option}"
in
o) OUTPUT_DIR=${OPTARG};;
t) LOCAL_TEST=${OPTARG};;
esac
done

export PYTHONPATH=$PYTHONPATH:../pytorch-transformers

# Local config
if ! $LOCAL_TEST;
then
    python run_lm_finetuning.py \
        --output_dir=$OUTPUT_DIR \
        --model_type=gpt2 \
        --model_name_or_path=gpt2-medium \
        --do_train \
        --train_data_file=$HOME/matan/dev/datasets/tacred/data/json/train.json \
        --block_size=512 \
        --per_gpu_train_batch_size=2 \
        --num_train_epochs=2 \
        --save_steps=500
        # --fp16
else
    python run_lm_finetuning.py \
        --output_dir=$OUTPUT_DIR \
        --model_type=gpt2 \
        --model_name_or_path=gpt2 \
        --do_train \
        --train_data_file=$HOME/matan/dev/datasets/tacred/data/json/train.json \
        --block_size=512 \
        --per_gpu_train_batch_size=2 \
        --num_train_epochs=2 \
        --save_steps=500
        # --fp16
fi