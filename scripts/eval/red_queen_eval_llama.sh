model_path=$1
benchmark=data/test/multi_turn/red_queen_turn5.json 

model_name=$(basename ${model_path})
output_dir=results/red_queen/${model_name}
results_file=${output_dir}/log.txt
batch_size=8
if [[ "${model_name}" == *"14B"* ]]; then
    batch_size=4
elif [[ "${model_name}" == *"32B"* ]]; then
    batch_size=4
fi

mkdir -p ${output_dir}

counter=1

python evaluation/evaluate.py \
    -m ${model_path} \
    -j cais/HarmBench-Llama-2-13b-cls \
    --benchmark ${benchmark} \
    --batch_size ${batch_size} \
    --output_dir ${output_dir} \
    --do_sample True \
    --temperature 1.2 \
    2>&1 | tee -a ${results_file} > /dev/null&

sleep 0.3s;
tail -f ${results_file}
