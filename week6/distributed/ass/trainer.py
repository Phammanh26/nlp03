import time
from typing import Union

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from tqdm import tqdm
from transformers import (
    AutoConfig, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    DataCollatorForSeq2Seq, 
    TrainingArguments, 
    logging, 
    set_seed,
    default_data_collator)

class Trainer:
    def __init__(
            self,
            model, 
            optimizer,
            num_epochs,
            max_length,
            batch_size,
            device):
        
        # setup the model
        self.model = model
        # setup the optimizer
        self.optimizer = optimizer

        self.device = device
        self.num_epochs = num_epochs
        self.max_length = max_length
        self.batch_size = batch_size



    def run(self, train_dataset, eval_dataset):
        model = self.model
        
        # Create the DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size = self.batch_size,
            collate_fn=default_data_collator)
        
        total_loss = 0

        for epoch in range(self.num_epochs):
            model.train()
            for step, batch in enumerate(tqdm(train_loader)):
                input_ids = batch["input_ids"].to(self.device)
                attention_masks = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = model(input_ids = input_ids,  attention_mask=attention_masks, labels = labels)
                
                self.optimizer.zero_grad()
                
                loss = outputs.loss
                total_loss += loss.detach().float()
                loss.backward()
                self.optimizer.step()
            
            print(f"Epoch {epoch + 1}, train total loss: {total_loss}")

            # TODO
            # evaluate
            # eval_loader = DataLoader(
            #     eval_dataset,
            #     batch_size = self.batch_size,
            #     collate_fn=default_data_collator)
