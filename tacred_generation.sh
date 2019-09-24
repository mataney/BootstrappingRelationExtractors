MODEL_DIR=$1
export PYTHONPATH=$PYTHONPATH:.

python examples/run_generation.py \
    --model_type=gpt2 \
    --model_name_or_path=$MODEL_DIR \
    --num_samples=10 \
    --top_k=40 \
    --top_p=0.0 \
    --length=50
