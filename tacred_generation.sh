source activate hugging_face

num_samples=10

while getopts m:o:s:p:t: option
do
case "${option}"
in
m) model_dir=${OPTARG};;
o) out_file=${OPTARG};;
s) num_samples=${OPTARG};;
t) prompt=${OPTARG};;
p) p=${OPTARG};;
esac
done

python run_generation.py \
    --model_type=gpt2 \
    --model_name_or_path=$model_dir \
    --out_file=$out_file \
    --num_return_sequences=$num_samples \
    --prompt="$prompt" \
    --length=50 \
    --p=$p \
    # --k=5 \