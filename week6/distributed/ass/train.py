from typing import Union
import argparse
import os

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments, logging, set_seed
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group




model_path = 'bigscience/bloom-560m'
data_path = 'alpaca_data.json'
output_dir = 'checkpoints/'

size_valid_set = 0.1
seq_length = 512
num_epochs = 3
micro_batch_size = 8
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


import gdown
url_data_path = 'https://drive.google.com/file/d/1QpgvQi6mFvN5-6ofmJunDbuz34tlLbLL/view?usp=sharing'
gdown.download(url_data_path, data_path, quiet=False, fuzzy=True)


backend = "nccl"

class Prompter(object):
    __slots__ = ("template")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self.template = {
            "description": "Template used by Alpaca-LoRA.",
            "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
            "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
            "response_split": "### Response:"}

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()

# Step 2: Initialize the Trainer
training_args = TrainingArguments(
        output_dir=output_dir,
        dataloader_drop_last=True,
        num_train_epochs=num_epochs,
        evaluation_strategy="steps",
        eval_steps=eval_freq,
        # save_steps=save_freq,
        save_total_limit=2,
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=num_warmup_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
   
        bf16=use_bf16,
        weight_decay=weight_decay,
        ddp_find_unused_parameters=False,
        logging_dir="./logs",
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=log_freq,
    )


def create_datasets(tokenizer):
    print("Start create_datasets")
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=seq_length,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < seq_length
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

train_dataset, eval_dataset = create_datasets(tokenizer)

data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)


# Step 3: Configure DistributedDataParallel (DDP)
# world_size = torch.cuda.device_count()  # Number of available GPUs
init_process_group(backend=backend)  # Initialize the process group


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

# Get the DDP rank
ddp_rank = int(os.environ['RANK'])
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
# Get the DDP local rank
ddp_local_rank = int(os.environ['LOCAL_RANK'])
# Set the cuda device
device = f'cuda:{ddp_local_rank}'
model.to(device)


model = DistributedDataParallel(model)


# Step 4: Train the model using Trainer
trainer.train()

# Step 5: Save the trained model
trainer.save_model("./trained_model")


destroy_process_group()