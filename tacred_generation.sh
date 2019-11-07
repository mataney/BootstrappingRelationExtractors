num_samples=10

while getopts o:f:s: option
do
case "${option}"
in
o) model_dir=${OPTARG};;
f) file_to_write=${OPTARG};;
s) num_samples=${OPTARG};;
esac
done
export PYTHONPATH=$PYTHONPATH:../pytorch-transformers

python run_generation.py \
    --model_type=gpt2 \
    --model_name_or_path=$model_dir \
    --file_to_write=$file_to_write \
    --num_samples=$num_samples \
    --length=50 \
    --top_p=0.9 \
    # --top_k=5 \
