import time
from typing import Union


import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments, logging, set_seed

import torch


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

class Trainer:
    def __init__(
            self, 
            model, 
            optimizer,
            train_dataset,
            eval_dataset,
            data_collator,
            batch_size):
        
        self.data_collator = data_collator
        self.model = model
        self.batch_size = batch_size
        # setup the optimizer
        self.optimizer = optimizer
        self.optimizer = None
        self.eval_dataset = eval_dataset
        self.train_dataset = train_dataset


        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def run(self):
        model, config = self.model, self.config

        # setup the dataloader
        # Create the DataLoaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.data_collator
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:
            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            logits, self.loss = model(x, y)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break

model_path = 'bigscience/bloom-560m'
data_path = 'alpaca_data.json'
output_dir = 'checkpoints/'


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


optimizer = model.get_optimizer()



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

trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        batch_size = micro_batch_size
        )

trainer.run()
        