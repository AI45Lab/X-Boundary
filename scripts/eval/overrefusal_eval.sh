model_path=$1
benchmark=$2
save_folder=$(dirname $(dirname "$model_path"))
output_dir=results/overrefusal/$(basename ${model_path})
results_file=${output_dir}/log.txt
mkdir -p ${output_dir}

python evaluation/evaluate.py \
    -m ${model_path} \
    -j cais/HarmBench-Llama-2-13b-cls \
    --benchmark ${benchmark} \
    --output_dir ${output_dir} \
    --batch_size 16 \
    2>&1 | tee -a ${results_file} > /dev/null&

sleep 0.3s;
tail -f ${results_file}