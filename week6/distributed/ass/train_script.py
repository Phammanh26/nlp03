import gdown
from utils import Prompter
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from accelerate import Accelerator
import torch
from trainer import Trainer
model_path = 'bigscience/bloom-560m'
data_path = 'alpaca_data.json'
output_dir = 'checkpoints/'
device = 'cuda:1'
size_valid_set = 0.1
max_length = 256
num_epochs = 3
batch_size = 8
gradient_accumulation_steps = 8


learning_rate = 1e-5
lr_scheduler_type = 'cosine'
num_warmup_steps = 100
weight_decay = 0.06

local_rank = 0
use_bf16 = False
seed = 0
log_freq = 1
eval_freq = 150


def create_datasets(tokenizer, max_length):
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < max_length
            and add_eos_token
        ):
            
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        
        return result
    
    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt
    
    prompter = Prompter()
    dataset = load_dataset('json', split='train', data_files=data_path)
    dataset = dataset.train_test_split(test_size=size_valid_set, seed=seed)

    train_data = dataset["train"].shuffle().map(generate_and_tokenize_prompt)
    valid_data = dataset["test"].map(generate_and_tokenize_prompt)

    
    
    dataset["test"].to_json('dataset/val_data.json')
    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    
    return train_data, valid_data


def load_pretrained_model():
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map={"": Accelerator().process_index},
    )
    model = prepare_model_for_int8_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def load_tokenizer_from_pretrained_model():

    print('Start config')
    config = AutoConfig.from_pretrained(model_path)
    architecture = config.architectures[0]
    print('Start tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print('End tokenizer')

    if "Llama" in architecture:
        print("Setting EOS, BOS, UNK, and PAD tokens for LLama tokenizer")
        tokenizer.add_special_tokens(
            {
                "eos_token": "</s>",
                "bos_token": "</s>",
                "unk_token": "</s>",
            }
        )
        tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
    
    return tokenizer

def download_from_driver(data_driver_path, location_path):
    gdown.download(data_driver_path, location_path, quiet=False, fuzzy=True)

if __name__ =="__main__":

    # Download data
    data_driver_path = 'https://drive.google.com/file/d/1QpgvQi6mFvN5-6ofmJunDbuz34tlLbLL/view?usp=sharing'
    download_from_driver(data_driver_path= data_driver_path, location_path= data_path)
    
    # Get tokenizer
    tokenizer = load_tokenizer_from_pretrained_model()
    # Prepare dataset
    train_dataset, eval_dataset = create_datasets(tokenizer = tokenizer, max_length=max_length)

    # Prepare model
    model = load_pretrained_model()

    # Prepare optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # prepare trainer
    trainer = Trainer(
        model = model, 
        optimizer = optimizer,
        num_epochs = num_epochs,
        max_length = max_length,
        batch_size = batch_size,
        device = device
        )
    
    # execute trainer 
    trainer.run(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
