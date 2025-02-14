import gc

import numpy as np
import torch

import sys
from tqdm import tqdm
from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer

from api import EvalInstance

# from pyreft import ReftModel

if sys.version_info >= (3, 12):
    from itertools import batched
else:
    from itertools import islice
    # Implementation of https://docs.python.org/3/library/itertools.html#itertools.batched
    # for python versions < 3.12
    def batched(iterable, n):
        # batched('ABCDEFG', 3) --> ABC DEF G
        if n < 1:
            raise ValueError('n must be at least one')
        it = iter(iterable)
        while batch := tuple(islice(it, n)):
            yield batch

def handle_non_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'isoformat'):
        return obj.isoformat()
    else:
        return str(obj)

def load_model_and_tokenizer(model_name_or_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if 'adapter' in model_name_or_path.lower():
        # Define the hook function
        # def capture_hidden_states(name):
        #     def hook(module, input, output):
        #         hook_storage[name].append(output)
        #     return hook
        if 'llama' in model_name_or_path.lower():
            base_model_path = 'checkpoints/Meta-Llama-3-8B-Instruct'
        elif 'vicuna' in model_name_or_path.lower():
            base_model_path = 'checkpoints/vicuna-7b-v1.5-16k'
        elif 'qwen' in model_name_or_path.lower():
            if '7b' in model_name_or_path.lower():
                base_model_path = 'checkpoints/Qwen2.5-7B-Instruct'
            elif '14b' in model_name_or_path.lower():
                base_model_path = 'checkpoints/Qwen2.5-14B-Instruct'
            elif '32b' in model_name_or_path.lower():
                base_model_path = 'checkpoints/Qwen2.5-32B-Instruct'
            elif '72b' in model_name_or_path.lower():
                base_model_path = 'checkpoints/Qwen2.5-72B-Instruct'
        elif 'mistral' in model_name_or_path.lower():
            base_model_path = 'checkpoints/Mistral-7B-Instruct-v0.2'
        elif 'gemma' in model_name_or_path.lower():
            base_model_path = 'checkpoints/Gemma-2-9b-it'

        if '32b' in model_name_or_path.lower():
            model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16) # dist
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path, torch_dtype=torch.bfloat16).to(device)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        model.load_adapter(model_name_or_path)
        # hook_storage = {}
        # Register a forward hook on the model to capture hidden states before LoRA
        # for name, module in model.named_modules():
        #     if "layers.0" in name.lower():
        #         hook_storage[name] = []
        #         hook = module.register_forward_hook(capture_hidden_states(name))
    else:
        if '32b' in model_name_or_path.lower():
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16) # dist
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model.eval()
    if 'qwen' in model_name_or_path.lower():
        tokenizer.bos_token = '<|begin|>'
    return model, tokenizer

def generate(model, tokenizer, instances: List[EvalInstance], gen_kwargs: dict) -> None:
    batch_size = gen_kwargs.pop("batch_size")
    compute_norms = gen_kwargs.pop("compute_norms")
    prefill = gen_kwargs.pop("prefill")
    use_template = gen_kwargs.pop("use_template")
    target_layer = 20
    tokenizer.padding_side = "left"
    if tokenizer.pad_token:
        pass
    elif tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        print("Setting pad token to [PAD]")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    pbar = tqdm(total=len(instances), desc="Generating completions...")
    for instance_batch in list(batched(instances, batch_size)):
        if use_template:
            if not prefill: 
                if "vicuna" in model.config.name_or_path.lower():                    
                    chat_template = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions. USER: {input} ASSISTANT:'
                    contexts = [chat_template.format(input=i.messages[0]['content']) for i in instance_batch]
                else:
                    contexts = [tokenizer.apply_chat_template(i.messages, tokenize=False, add_generation_prompt=True) for i in instance_batch]
            else:
                contexts = []
                for instance in instance_batch:
                    message = tokenizer.apply_chat_template(instance.messages, tokenize=False, add_generation_prompt=True)
                    message += instance.default_target + ":\n\n"
                    contexts.append(message)
                
            # Remove BOS token if it is present -- this gets taken care of while tokenizing
            contexts = [c.replace(tokenizer.bos_token, "") for c in contexts]
        else:
            if prefill:
                raise ValueError("Prefill attack uses a template")
            contexts = []
            for instance in instance_batch:
                instance.messages.append({"role": "assistant", "content": instance.default_target})
                message = "\n".join(d['content'] for d in instance.messages)
                contexts.append(message)

        inputs = tokenizer(
            contexts,
            truncation=False,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=True
        )

        inputs["input_ids"] = inputs["input_ids"].to('cuda')
        inputs["attention_mask"] = inputs["attention_mask"].to('cuda')

        output = model.generate(
            **inputs,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
            output_hidden_states=compute_norms,
            return_dict_in_generate=True,
            **gen_kwargs
        )
        generated_tokens = output.sequences[:, inputs["input_ids"].shape[1]:]
        generations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # Compute activation norms
        offset = 1 if tokenizer.bos_token and output.sequences[0][0] == tokenizer.bos_token_id else 0
        if compute_norms:
            activation_norms = []
            num_layers = len(output.hidden_states[0])

            for i in range(num_layers):
                hidden_states = torch.cat([output.hidden_states[j][i] for j in range(len(output.hidden_states))], dim=1)
                activation_norm = hidden_states.norm(dim=-1)
                activation_norms.append(activation_norm)

            # (num_layers, batch_size, seq_len) -> (batch_size, num_layers, seq_len)
            activation_norms = torch.stack(activation_norms, dim=0).permute(1, 0, 2)
            activation_norms = activation_norms.cpu().numpy()

        output.sequences = output.sequences.cpu().numpy()

        for i, instance in enumerate(instance_batch):
            instance.generation = generations[i]
            instance.tokens = output.sequences[i, offset:]
            if compute_norms:
                instance.activation_norms = activation_norms[i]

        # print(f"The number of samples using ref model: {ref_cnt}")
        # del generated_tokens
        gc.collect()
        torch.cuda.empty_cache()
        pbar.update(batch_size)