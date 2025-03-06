import argparse
from dataclasses import asdict
from datetime import datetime
import gc
import json
import numpy as np
import os
from pathlib import Path
import pickle

import torch
from transformers import set_seed

from api import EvalInstance
from judge import Judge, JudgeConfig
from utils import (
    generate,  
    handle_non_serializable, 
    load_model_and_tokenizer,
)
from overrefusal_eval import (
    overrefusal_judge,
    overrefusal_analysis
)
from multi_jailbreak_eval import (
    multi_turn_generate,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--local_rank", type=str, default=None)
    parser.add_argument("-m", "--model_name_or_path", type=str, default=None)
    parser.add_argument("-j", "--judge_name_or_path", type=str, default="cais/HarmBench-Llama-2-13b-cls")
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_norms", action="store_true")
    parser.add_argument("--vlm_acc", type=bool, default=False)
    # Generation arguments
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--do_sample", type=bool, default=False)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--prefill", type=bool, default=False)
    parser.add_argument("--use_template", type=bool, default=True)
    # Judge arguments
    parser.add_argument("--judge_temperature", type=int, default=None)
    parser.add_argument("--judge_max_new_tokens", type=int, default=1)
    parser.add_argument("--judge_overrefusal_mode", type=str, default='strmatch')
    args = parser.parse_args()
    return args

def evaluate(
    model_name_or_path,
    benchmark_path, 
    start,
    limit,
    seed,
    overrefusal_judge_mode,
    gen_kwargs,
    judge_config,
    vlm_acc
):
    if seed is not None:
        set_seed(seed)
        # torch.use_deterministic_algorithms(True, warn_only=True)

    model, tokenizer = load_model_and_tokenizer(model_name_or_path, vlm_acc)
    
    with open(benchmark_path) as f:
        benchmark = json.load(f)
        instances = []
        for d in benchmark["data"]:
            instance = EvalInstance(**d)
            if instance.context is not None:
                instance.messages[-1]["content"] = instance.context + "\n\n---\n\n" + instance.messages[-1]["content"]
            instances.append(instance)

        if limit is not None:
            instances = instances[start:limit]
        else:
            instances = instances[start:]
    
    # Generate completions from the model
    gen_kwargs['vlm_acc'] = vlm_acc
    gen_kwargs['model_name_or_path'] = model_name_or_path
    if 'multi_turn' in benchmark_path.lower(): 
        multi_turn_generate(model, tokenizer, instances, gen_kwargs)
    else:
        generate(model, tokenizer, instances, gen_kwargs)

    print(f"Sample input: {instances[0].messages}")
    print(f"Sample generation: {instances[0].generation}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Run evaluation using LLM judge
    if 'harmbench' in benchmark_path.lower() or 'multi_turn' in benchmark_path.lower(): 
        judge_config = JudgeConfig(**judge_config)
        judge = Judge(judge_config)
        if 'multi_turn' in benchmark_path.lower(): 
            score = judge.multi_turn_eval(instances)
        else:         
            score = judge.evaluate(instances)
    else: # overrefusal
        overrefusal_judge(instances, overrefusal_judge_mode, model_name_or_path)
        score = overrefusal_analysis(instances, benchmark_path)
        
    return score, instances

def main():
    args = parse_args()
    args_dict = vars(args)
    for k, v in args_dict.items():
        print(f"{k}: {v}")
    
    gen_kwargs = {
        "temperature": args.temperature,
        "do_sample": args.do_sample,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "compute_norms": args.save_norms,
        "prefill": args.prefill,
        "use_template": args.use_template,
    }

    judge_config = {
        "model_name_or_path": args.judge_name_or_path,
        "temperature": args.judge_temperature,
        "max_new_tokens": args.judge_max_new_tokens,
        "seed": args.seed,
    }
    
    score, instances = evaluate(
        args.model_name_or_path,
        args.benchmark,
        args.start,
        args.limit,
        args.seed,
        args.judge_overrefusal_mode,
        gen_kwargs,
        judge_config,
        args.vlm_acc
    )

    print(f"Evaluation score: {score}")

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        samples = []
        activation_norms = []
        generated_tokens = []
        for instance in instances:
            sample = asdict(instance)
            activation_norms.append(instance.activation_norms)
            generated_tokens.append(instance.tokens)
            samples.append({k:v for k, v in sample.items() if k not in ("activation_norms, tokens")})
        
        result = {"args": args_dict, "time": datetime.now(), "score": score, "samples": samples}
        dumped = json.dumps(result, indent=2, default=handle_non_serializable, ensure_ascii=False)
        file_name = os.path.basename(args.benchmark)[:6]
        with open(output_dir.joinpath(f"{file_name}_{args.start}.json"), "w") as f:
            f.write(dumped)

        if args.save_norms:
            with open(output_dir.joinpath("activation_norms.pkl"), "wb") as f:
                pickle.dump(activation_norms, f)

            with open(output_dir.joinpath("tokens.pkl"), "wb") as f:
                pickle.dump(generated_tokens, f)
        
if __name__ == "__main__":
    main()
