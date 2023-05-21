from prompt import Prompter
import torch

prompter = Prompter()

from prompt import Prompter
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
    
    device = 'cuda:1'
    model.to(device)

    if sample == {}:
        sample = {
            "instruction": "Describe what would happen if a person jumps off a cliff",
            "input": "No input",
            "output": "If a person jumps off a cliff, they could suffer severe injuries or even death due to the fall. Depending on the height of the cliff, they could experience a free fall of several seconds before hitting the ground. If they do not jump far enough away from the cliff, they could also hit the rocks and cause serious injuries."
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