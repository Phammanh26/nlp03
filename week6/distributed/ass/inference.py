from prompt import Prompter
import torch

prompter = Prompter()

def tokenize(tokenizer, prompt, max_length, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        
        )

    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < max_length
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
    
    result["input_ids"] = torch.tensor(result['input_ids'])

    return result
    
def prepare_model_inputs(tokenizer, instruction, context, max_length):
    full_prompt = prompter.generate_prompt(
        instruction,
        context
        
    )
    tokenized_full_prompt = tokenize(tokenizer, full_prompt, max_length)
    return tokenized_full_prompt


def generate_inference(model, tokenizer, device, max_length):
    sample = {
        "instruction": "Describe what would happen if a person jumps off a cliff",
        "input": "No input",
        "output": "If a person jumps off a cliff, they could suffer severe injuries or even death due to the fall. Depending on the height of the cliff, they could experience a free fall of several seconds before hitting the ground. If they do not jump far enough away from the cliff, they could also hit the rocks and cause serious injuries."
    }

    token_encoded = prepare_model_inputs(tokenizer, sample["instruction"], sample["input"], max_length)

    outputs = model.generate(
            input_ids=token_encoded['input_ids'].reshape(1, max_length).to(device),
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens =  max_length
        )
    response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    print(f"\nsample: {sample}")
    print(f"\nresponse: {response}")