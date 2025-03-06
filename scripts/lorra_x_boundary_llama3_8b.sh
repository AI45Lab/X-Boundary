export WANDB_MODE=offline
export MASTER_PORT=$((29000 + RANDOM % 1000))
export CUBLAS_WORKSPACE_CONFIG=:16:8

### Llama-3-8B Config ###
boundary_data_size=500
max_steps=180
multi_turn_data_path=data/train/SafeMT_train_600.json
model_name_or_path=meta-llama/Meta-Llama-3-8B-Instruct
lorra_alpha=10
layers="10,20"
transform_layers="-1"

output_dir="./models/Llama3_8B_X_Boundary_adapter"
results_file=${output_dir}/log.txt
mkdir -p ${output_dir}

echo "model_name_or_path=$model_name_or_path"
echo "output_dir=$output_dir"

accelerate launch --config_file configs/accelerate_zero1.yaml \
    --num_processes 1 --main_process_port $MASTER_PORT --deepspeed_hostfile ds_hostfile \
    src/lorra_x_boundary.py \
    --model_name_or_path $model_name_or_path \
    --target_layers $layers \
    --transform_layers $transform_layers \
    --lorra_alpha $lorra_alpha \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --multi_turn_data_path $multi_turn_data_path \
    --boundary_data_size $boundary_data_size \
    --output_dir  $output_dir \
    --overwrite_output_dir \
    --max_steps $max_steps \
    --bf16 True \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --do_eval \
    --save_total_limit 0 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --lr_scheduler_type "constant" \
    --logging_strategy "steps" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 8192 \
    --q_lora False \
    --gradient_checkpointing True \
    --report_to none \
    --log_every 1 \
    2>&1 | tee -a ${results_file} > /dev/null&

sleep 0.3s;
tail -f ${results_file}
