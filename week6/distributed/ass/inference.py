from prompt import Prompter
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch

from peft import LoraConfig
from lora_model_solution import LoraModelForCasualLM

prompter = Prompter()
 
def tokenize( tokenizer, prompt, max_length, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length =max_length
        )

    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < max_length
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
    
    result["input_ids"] = torch.tensor(result['input_ids'])
    result["attention_mask"] = torch.tensor(result['attention_mask'])

    return result
    
def generate_and_tokenize_prompt(tokenizer, instruction, context, max_length):
    full_prompt = prompter.generate_prompt(
        instruction,
        context )
    tokenized_full_prompt = tokenize(tokenizer, full_prompt, max_length)
    return tokenized_full_prompt
    

def generate_inference(
        model, 
        tokenizer, 
        sample: dict = {},
        max_length: int = 128,  
        max_new_tokens: int = 64):
    
    device = 'cuda'
    model.to(device)

    if sample == {}:
        sample = {
            "instruction": "Describe what would happen if a person jumps off a cliff",
            "input": "No input",
        }

    token_encoded = generate_and_tokenize_prompt(
        tokenizer, 
        sample["instruction"], 
        sample["input"], 
        max_length
        )

    outputs = model.generate(
            input_ids=token_encoded['input_ids'].unsqueeze(0).to(device),
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens =  max_new_tokens
        )
    
    response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    print(f"\nresponse: {response}")

if __name__ == '__main__':
    model_path = 'bigscience/bloom-560m'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = LoraModelForCasualLM(model, lora_config)
    model.load_state_dict(torch.load('checkpoints/epoch_9/model.pt'))
    generate_inference(model, tokenizer)