import argparse
import json

import torch
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, AutoConfig

from lora_model import LoraModelForCasualLM
from prompt import Prompter

def get_response(prompt, tokenizer, model, generation_config, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
                input_ids=inputs['input_ids'].cuda(),
                generation_config=generation_config,
                max_new_tokens=max_new_tokens,
                do_sample=True)
    output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return output



class Generater:
    def __init__(self, model_path, lora_weights_path, load_8bit, max_new_tokens: int =128):
        self.max_new_tokens =max_new_tokens 
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)
        architecture = config.architectures[0]
        if "Llama" in architecture:
            print("Setting EOS, BOS, UNK, and PAD tokens for LLama tokenizer")
            self.tokenizer.add_special_tokens(
                {
                    "eos_token": "</s>",
                    "bos_token": "</s>",
                    "unk_token": "</s>",
                }
            )
            self.tokenizer.pad_token_id = (
                0  # unk. we want this to be different from the eos token
            )
            self.tokenizer.padding_side = "left"
        
        model = AutoModelForCausalLM.from_pretrained(
                model_path,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map="auto")
        self.model = LoraModelForCasualLM.from_pretrained(
                model,
                lora_weights_path,
                torch_dtype=torch.float16)
        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.eval()
        
        self.prompter = Prompter()

    def generate(self, instruction: str, user_inp: str, temperature: float):

        top_k=40
        top_p=128
        temperature=0.1
        num_beams =1
    
        self.generation_config = GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams)

        # instruction = input("Your Instruction: ")
        # user_inp = input("Your input (enter n/a if there is no): ")
        if user_inp.lower().strip() == "n/a":
            user_inp = None
        prompt = self.prompter.generate_prompt(instruction, user_inp)
        output = get_response(prompt, self.tokenizer, self.model, self.generation_config, self.max_new_tokens)
        response = self.prompter.get_response(output)
        return response
        