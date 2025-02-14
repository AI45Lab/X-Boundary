import os
import json
import torch
import matplotlib.pyplot as plt
from transformers import LlavaNextForConditionalGeneration, AutoModelForCausalLM

def save_model_and_tokenizer(model_name_or_path, model, tokenizer, drop_layers_after, output_dir, trainer):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n\nModel and tokenizer saving to {output_dir}\n\n")
    
    ### only save lora adapter
    model.save_pretrained(output_dir)
    
    # ### save merged model
    # # merge lora
    # merged_model = model.merge_and_unload() 
    # # merge original layers
    # if drop_layers_after is not None:
    #     anchor_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=merged_model.dtype, device_map="auto")
    #     merged_model.model.layers = merged_model.model.layers + anchor_model.model.layers[drop_layers_after+1:]
    #     merged_model.config = anchor_model.config

    # merged_model.save_pretrained(output_dir)
    # tokenizer.save_pretrained(output_dir)

    lorra_config_path = os.path.join(output_dir, "lorra_config.json")
    with open(lorra_config_path, "w", encoding="utf-8") as file:
        json.dump(trainer.lorra_args.to_dict(), file, indent=2)
    
    torch.use_deterministic_algorithms(False)
    if trainer.training_args.do_eval:
        trainer.evaluate()
    

def save_llava_model_and_tokenizer(model_name_or_path, model, processor, drop_layers_after, output_dir, trainer):
    os.makedirs(output_dir, exist_ok=True)
    print(f"MModel and processor saving to {output_dir}")
    
    # merge lora
    merged_model = model.merge_and_unload() 
    # merge original layers
    
    anchor_model = LlavaNextForConditionalGeneration.from_pretrained(model_name_or_path, device_map="auto", torch_dtype=merged_model.dtype)
    merged_model.language_model.model.layers = merged_model.language_model.model.layers + anchor_model.language_model.model.layers[drop_layers_after+1:]
    merged_model.config = anchor_model.config

    merged_model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    lorra_config_path = os.path.join(output_dir, "lorra_config.json")
    with open(lorra_config_path, "w", encoding="utf-8") as file:
        json.dump(trainer.lorra_args.to_dict(), file, indent=2)
    
    torch.use_deterministic_algorithms(False)
    if trainer.training_args.do_eval:
        trainer.evaluate()

def draw_loss_picture(loss, cb_loss, retain_loss, val_loss, save_dir):
    x=range(0,len(loss))
    plt.plot(x,loss,'o-',color = 'r',label="train loss") 
    plt.plot(x,cb_loss,'o-',color = 'g',label="cb loss") 
    plt.plot(x,retain_loss,'o-',color = 'b',label="retain loss") 
    if len(val_loss) != 0:
        plt.plot(x,val_loss,'o-',color = 'pink',label="val loss")
    plt.xlabel("step") 
    plt.ylabel("loss") 
    plt.legend(loc = "best")
    model_name = os.path.basename(save_dir)
    plt.savefig(os.path.join(save_dir, model_name + '_loss_figure.png'))