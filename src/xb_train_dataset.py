from torch.utils.data import Dataset
import datasets
from datasets import load_dataset
import transformers
from typing import Dict
import torch
import numpy as np
from tqdm import tqdm
import json
import random
import csv
random.seed(0)

class XBoundaryDataset(Dataset):
    
    def __init__(self, 
                tokenizer: transformers.PreTrainedTokenizer, 
                num_examples,
                lorra_args,
                model_name_or_path,
                ):
        super(XBoundaryDataset, self).__init__()

        self.model_name_or_path = model_name_or_path.lower()
        self.max_length = 1024
        self.use_refusal_retain = lorra_args.use_refusal_retain
        self.boundary_data_size = lorra_args.boundary_data_size
        self.multi_turn_data_path = lorra_args.multi_turn_data_path

        one_shot_template = "{user_tag}{instruction}{assistant_tag}<SEPARATOR>{response}"

        # ================ Model and Template Config  ================
        # Default configs
        sep_token = ""
        switch_select = [0]
        self.use_refusal_retain = False
        user_tag, assistant_tag = None, None
        if 'llama-3' in self.model_name_or_path:
            print("USING LLAMA TEMPLATE")
            user_tag="<|start_header_id|>user<|end_header_id|>\n\n"
            assistant_tag="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            switch_select = [0, 1]
            self.use_refusal_retain = True
        elif 'mistral' in self.model_name_or_path:
            print("USING MISTRAL TEMPLATE")
            # fix spacing issue in template
            tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
            user_tag="[INST] "
            assistant_tag=" [/INST]"
            sep_token = " "
        elif 'qwen' in self.model_name_or_path:
            print("USING QWEN TEMPLATE")
            user_tag = "<|im_start|>user\n"
            assistant_tag = "<|im_end|>\n<|im_start|>assistant\n"
            sep_token = "<|im_end|>"
            switch_select = [0, 1]
            self.use_refusal_retain = True
        elif 'gemma' in self.model_name_or_path:
            print("USING GEMMA TEMPLATE")
            user_tag = "<start_of_turn>user\n"
            assistant_tag = "<end_of_turn>\n<start_of_turn>model\n"
            sep_token = " "
            switch_select = [0, 1]
            self.use_refusal_retain = True
        else:
            raise NotImplementedError(f"Config {self.model_name_or_path} not found")
        
        assert user_tag and assistant_tag, "user_tag/assistant_tag not defined"

        self.user_tag = user_tag
        self.assistant_tag = assistant_tag
        self.sep_token = sep_token

        # ======================= Erase set ======================= #
        ## single-turn
        with open("data/train/circuit_breakers_train_2400.json") as file:
                dataset = json.load(file)
        ## multi-turn
        if self.multi_turn_data_path is not None:
            with open(self.multi_turn_data_path) as file:
                dataset.extend(json.load(file))

        x_boundary_orig = []

        for i, d in tqdm(enumerate(dataset)):
            cb_output = d['output']
            switch = np.random.choice(switch_select)
            if switch == 0:
                formatted_input = one_shot_template.format(
                    user_tag=user_tag, assistant_tag=assistant_tag,
                    instruction=d['prompt'], response=cb_output)
            elif switch == 1:
                formatted_input = one_shot_template.format(
                    user_tag=user_tag, assistant_tag=assistant_tag,
                    instruction="", response=cb_output)
            
            x_boundary_orig.append(formatted_input)
        
        self.x_boundary_orig = x_boundary_orig
        random.shuffle(self.x_boundary_orig)
        print("x_boundary_orig[0]", x_boundary_orig[0])
        print("X boundary length:", len(self.x_boundary_orig))

        # ======================= Retain Set ======================= #
        ## general safe QA
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
        orig_s = []
        for example in ds:
            messages = example["messages"]
            if len(messages) < 2: continue

            switch = np.random.choice(switch_select)
            if switch == 0:
                # Add special handling for Qwen model
                if 'qwen' in self.model_name_or_path.lower():
                    formatted_input = tokenizer.apply_chat_template(messages, tokenize=False)
                    if tokenizer.bos_token:
                        formatted_input = formatted_input.replace(tokenizer.bos_token, "")
                else:
                    formatted_input = tokenizer.apply_chat_template(messages, tokenize=False).replace(tokenizer.bos_token, "")
            elif switch == 1:
                formatted_input = one_shot_template.format(
                    user_tag=user_tag, assistant_tag=assistant_tag,
                    instruction="", response=messages[1]["content"])

            orig_s.append(formatted_input)

            if len(orig_s) > num_examples:
                break
        self.orig_s_retain = orig_s
        random.shuffle(self.orig_s_retain)
        print("orig_s_retain[0]", orig_s[0])
        print("Orig s length:", len(self.orig_s_retain))

        ## boundary-safe QA
        if self.boundary_data_size > 0:
            with open('data/train/ORbench_retain_set.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                data = [row for row in data if row['status'] == "1_full_compliance"]
                data = data[:self.boundary_data_size]

            refusal_response = 'I cannot provide with information or guidance on illegal or harmful activities. Can I help you with something else?'
            for i, d in enumerate(data):
                switch = np.random.choice(switch_select)
                # add into retain set
                retain_formatted_input = one_shot_template.format(
                    user_tag=user_tag, assistant_tag=assistant_tag,
                    instruction=d['prompt'], response=d['completion'])
                # add into boundary set
                bs_formatted_input = one_shot_template.format(
                    user_tag=user_tag, assistant_tag=assistant_tag,
                    instruction=d['prompt'], response=refusal_response)
                self.x_boundary_orig.append({"overrefusalQA": bs_formatted_input, "retainQA": retain_formatted_input})
                random.shuffle(self.x_boundary_orig)
            print("Added overrefusal circuit length:", len(self.x_boundary_orig))

        ## harmful Q + refusal A
        if self.use_refusal_retain:
            with open("data/train/circuit_breakers_train_2400.json") as file:
                dataset = json.load(file)
            if 'qwen' in self.model_name_or_path.lower():
                output_label = 'qwen_output'
            elif 'gemma' in self.model_name_or_path.lower():
                output_label = 'gemma_output'
            else:
                output_label = 'llama3_output'
            print("using refusal retain:", output_label)

            random.shuffle(dataset)
            dataset = dataset[:2000]
            refusal_retain_orig = []
            for i, d in tqdm(enumerate(dataset*2)):
                switch = np.random.choice(switch_select)
                if switch == 0:
                    formatted_input = one_shot_template.format(
                        user_tag=user_tag, assistant_tag=assistant_tag,
                        instruction=d['prompt'], response=d[output_label])
                elif switch == 1:
                    formatted_input = one_shot_template.format(
                        user_tag=user_tag, assistant_tag=assistant_tag,
                        instruction="", response=d[output_label])
                refusal_retain_orig.append(formatted_input)

            self.orig_s_retain += refusal_retain_orig
            random.shuffle(self.orig_s_retain)
            print("refusal_orig_s[0]", refusal_retain_orig[0])
            print("Orig s length:", len(self.orig_s_retain))

        # ======================= Val ======================= #
        with open("data/train/circuit_breakers_val.json") as file:
            dataset = json.load(file)
        val_orig = []
        for i, d in tqdm(enumerate(dataset)):
            val_orig.append(one_shot_template.format(
                user_tag=user_tag, assistant_tag=assistant_tag,
                instruction=d['prompt'], response=d['output']))

        self.val_orig = val_orig
        self.tokenizer = tokenizer

    def __len__(self):
        return min(len(self.orig_s_retain), len(self.x_boundary_orig))
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        orig_s_retain = self.orig_s_retain[i]
        x_boundary_orig = self.x_boundary_orig[i]
        val_orig = self.val_orig[i % len(self.val_orig)]

        cb_tokenized_kwargs = dict(max_length=512, padding='max_length', truncation=True, return_tensors="pt")
        tokenize_kwargs = dict(max_length=1024, padding="max_length", truncation=True, return_tensors="pt")

        # =========== Circuit Breaker Inputs =========== #
        # === split to [request, response] shape [512,512] to support different mask configs ===
        is_refusal_forget = False
        if type(x_boundary_orig) == dict:
            orig_s_retain = x_boundary_orig['retainQA']
            x_boundary_orig = x_boundary_orig['overrefusalQA']
            is_refusal_forget = True
        cb_request, cb_response = x_boundary_orig.split('<SEPARATOR>')
        self.tokenizer.padding_side = "left"
        tokenized_request_x_boundary = self.tokenizer(cb_request, **cb_tokenized_kwargs)
        self.tokenizer.padding_side = "right"
        response_tokenized_x_boundary = self.tokenizer(cb_response, add_special_tokens=False, **cb_tokenized_kwargs)
        self.tokenizer.padding_side = "left"

        combined_input_ids_x_boundary = torch.cat([tokenized_request_x_boundary["input_ids"], response_tokenized_x_boundary["input_ids"]], dim=1)
        combined_attention_mask_x_boundary = torch.cat([tokenized_request_x_boundary["attention_mask"], response_tokenized_x_boundary["attention_mask"]], dim=1)
        if is_refusal_forget:
            label_mask_x_boundary = torch.cat([tokenized_request_x_boundary['attention_mask'], torch.zeros(response_tokenized_x_boundary["attention_mask"].shape)], dim=1)
        else:
            label_mask_x_boundary = combined_attention_mask_x_boundary

        # ========== Retain Inputs ===========
        if type(orig_s_retain) == dict:
            orig_s_retain = orig_s_retain['refusal_retain']
            retain_request, retain_response = orig_s_retain.split('<SEPARATOR>')
            self.tokenizer.padding_side = "left"
            tokenized_request_retain = self.tokenizer(retain_request, **cb_tokenized_kwargs)
            self.tokenizer.padding_side = "right"
            response_tokenized_retain = self.tokenizer(retain_response, add_special_tokens=False, **cb_tokenized_kwargs)
            self.tokenizer.padding_side = "left"

            combined_input_ids_retain = torch.cat([tokenized_request_retain["input_ids"], response_tokenized_retain["input_ids"]], dim=1)
            combined_attention_mask_retain = torch.cat([tokenized_request_retain["attention_mask"], response_tokenized_retain["attention_mask"]], dim=1)
            label_mask_retain = torch.cat([tokenized_request_retain['attention_mask'], torch.zeros(response_tokenized_retain["attention_mask"].shape)], dim=1)
        else:
            tokenized_inputs_retain = self.tokenizer(orig_s_retain.replace('<SEPARATOR>', self.sep_token), **tokenize_kwargs)
            combined_input_ids_retain = tokenized_inputs_retain['input_ids']
            combined_attention_mask_retain = tokenized_inputs_retain['attention_mask']
            label_mask_retain = tokenized_inputs_retain['attention_mask']

        
        # =========== Val Inputs ===========
        tokenized_inputs_val = self.tokenizer(val_orig.replace('<SEPARATOR>', self.sep_token), **tokenize_kwargs)

        return dict(
            input_ids_x_boundary=combined_input_ids_x_boundary,
            attention_mask_x_boundary=combined_attention_mask_x_boundary,
            label_mask_x_boundary=label_mask_x_boundary,
            input_ids_retain=combined_input_ids_retain,
            attention_mask_retain=combined_attention_mask_retain,
            label_mask_retain=label_mask_retain,
            input_ids_val=tokenized_inputs_val["input_ids"],
            attention_mask_val=tokenized_inputs_val["attention_mask"],
            is_refusal=is_refusal_forget
        )