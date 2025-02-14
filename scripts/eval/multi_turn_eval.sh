model_path=$1 
benchmark=data/test/multi_turn/SafeMTData_Attack_test_600.json

model_name=$(basename ${model_path})
save_folder=$(dirname $(dirname "$model_path"))
output_dir=results/multi_turn/${model_name}
results_file=${output_dir}/log.txt
batch_size=8
if [[ "${model_name}" == *"14B"* ]]; then
    batch_size=4
elif [[ "${model_name}" == *"32B"* ]]; then
    batch_size=4
fi

mkdir -p ${output_dir}

python evaluation/evaluate.py \
    -m ${model_path} \
    -j cais/HarmBench-Llama-2-13b-cls \
    --benchmark ${benchmark} \
    --batch_size ${batch_size} \
    --output_dir ${output_dir} \
    2>&1 | tee -a ${results_file} > /dev/null&

sleep 0.3s;
tail -f ${results_file}