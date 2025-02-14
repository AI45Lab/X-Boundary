model_path=$1
benchmark=data/test/harmbench_test.json 

model_name=$(basename ${model_path})

save_folder=$(dirname $(dirname "$model_path"))
output_dir=results/single_turn/${model_name}
results_file=${output_dir}/log.txt
mkdir -p ${output_dir}

python evaluation/evaluate.py \
    -m ${model_path} \
    -j cais/HarmBench-Llama-2-13b-cls \
    --benchmark ${benchmark} \
    --batch_size 16 \
    --output_dir ${output_dir} \
    2>&1 | tee -a ${results_file} > /dev/null&

sleep 0.3s;
tail -f ${results_file}