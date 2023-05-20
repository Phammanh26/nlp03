import os
import torch
from tqdm import tqdm


from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import (
    AutoConfig, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
  )
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from accelerate import Accelerator
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, SequentialSampler


from utils.common import download_from_driver
from prepare_data import create_datasets
from inference import generate_inference

import warnings
warnings.filterwarnings('ignore')

class Trainer:
    def __init__( self, model, tokenizer, gpu_id: int, is_ddp_training: bool, output_dir: str = 'checkpoints/',  num_epochs: int = 10,max_length: int = 128, batch_size: int = 8 ):
        """
        Initialize the Trainer class.

        Args:
            model: Pretrained model object.
            tokenizer: Tokenizer object for text processing.
            num_epochs: Number of training epochs.
            max_length: Maximum sequence length.
            batch_size: Training batch size.
            gpu_id: GPU ID for training.
        """
        
        
        self.num_epochs = num_epochs
        self.max_length = max_length
        self.batch_size = batch_size
        self.gpu_id = gpu_id
        self.is_ddp_training = is_ddp_training
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.model = model.to(f"cuda:{self.gpu_id}")
        

    def set_ddp_training(self):
        self.model = DDP(self.model, device_ids=[self.gpu_id], output_device=self.gpu_id)
    
    def _is_master_process(self):
        if self.is_ddp_training:
            ddp_rank = int(os.environ['RANK'])
        else:
            ddp_rank = 0
        return ddp_rank == 0
        
    def _run_batch(self, batch):
        """
        Run a single training batch.

        Args:
            batch: Batch data.

        Returns:
            Loss value for the batch.
        """
        
        
        self.optimizer.zero_grad()
        outputs = self.model(**batch) 
        loss = outputs.loss
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _run_epoch(self, train_dataloader, epoch):
        """
        Run a single training epoch.

        Args:
            train_loader: Training data loader.
            epoch: Current epoch number.

        Returns:
            Total loss value for the epoch.
        """
        
        epoch_loss = 0
        self.model.train()
        
        if self._is_master_process():
            train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1} [Training]", position=0, leave=False)
        else:
            train_progress_bar = train_dataloader
        
        for step, batch in enumerate(tqdm(train_progress_bar)):
            if step == 0:
                print(f"Batch shape: {batch['input_ids'].shape}")

            batch = {key: value.to(self.gpu_id) for key, value in batch.items()}
            print(f"\nEpoch [{epoch}] | Batch [{step}] | on GPU [{self.gpu_id}]")
            loss = self._run_batch(batch)
            epoch_loss += loss
        
        return epoch_loss
    
    def _save_checkpoint(self, epoch):
        # ckp = self.model.module.state_dict()

        path_dir = f"{self.output_dir}/epoch_{epoch}"
        
        # check path_dir exited
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)

        # Test inference
        generate_inference(model = self.model.module, tokenizer= self.tokenizer, device=self.gpu_id, max_length = self.max_length)
        
        # save checkpoints
        torch.save(self.model.module.state_dict(), f'{path_dir}/model.pt')

        print(f"\nEpoch {epoch} | Training checkpoint saved at {path_dir}")

    def prepare_dataloader(self, train_dataset, eval_dataset):
        # Create the DataLoaders
        data_trainloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=DistributedSampler(train_dataset, rank=self.gpu_id) if self.is_ddp_training else None,
            collate_fn=lambda x: {
                "input_ids": torch.stack([sample["input_ids"].to(local_rank) for sample in x]),
                "attention_mask": torch.stack([sample["attention_mask"].to(local_rank) for sample in x]),
                "labels": torch.stack([sample["labels"].to(local_rank) for sample in x]),
            })

        # Create the DataLoaders
        data_testloader = DataLoader(
            eval_dataset,
            batch_size=self.batch_size,
            sampler=SequentialSampler(eval_dataset),
            collate_fn=lambda x: {
                "input_ids": torch.stack([sample["input_ids"].to(local_rank) for sample in x]),
                "attention_mask": torch.stack([sample["attention_mask"].to(local_rank) for sample in x]),
                "labels": torch.stack([sample["labels"].to(local_rank) for sample in x]),
            })
        

       
        return data_trainloader, data_testloader
    
    def _eval(self, eval_dataloader, epoch: int):
        # TODO: Evaluation
        avg_loss = 0
        model.eval()
        if self._is_master_process():
            eval_progress_bar = tqdm(eval_dataloader, desc=f"Epoch {epoch + 1} [Evaluation]", position=0, leave=False)
        else:
            eval_progress_bar = eval_dataloader
        
        for batch in eval_progress_bar:
            with torch.no_grad():
                outputs = self.model(**batch) 
            avg_loss += outputs.loss
        avg_loss = avg_loss/(len(eval_dataloader))
        return avg_loss
    
    def run(self, data_path: str, size_valid_set: int = 0.25, seed:int=123):
        """
        Run the training process.

        Returns:
            None
        """
        # Prepare dataset
        train_dataset, eval_dataset = create_datasets(
            tokenizer = self.tokenizer,
            max_length = self.max_length,
            data_path = data_path,
            size_valid_set = size_valid_set,
            seed = seed
           )
        
        
        train_dataloader, eval_dataloader = self.prepare_dataloader(train_dataset, eval_dataset)
        
        if self.is_ddp_training:
            self.set_ddp_training()

        # Setup the optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        avg_train_loss = 0
        if self._is_master_process():
            print(f"\nStart training | total epochs: {self.num_epochs}")
        
        for epoch in range(self.num_epochs):
            if self.is_ddp_training:
                train_dataloader.sampler.set_epoch(epoch)
            train_loss = self._run_epoch(train_dataloader, epoch)
            avg_train_loss += train_loss
            
            # Evaluate after each epoch
            eval_loss = self._eval(eval_dataloader = eval_dataloader, epoch = epoch)
            
            # Save checkpoint
            if self._is_master_process():
                self._save_checkpoint(epoch = epoch)
            
            print(f"\nCompleted training epoch: {epoch} | train loss = {train_loss} | eval loss = {eval_loss}")


       


def load_tokenizer_from_pretrained_model(model_path):
    print('Loading config....')
    config = AutoConfig.from_pretrained(model_path)
    architecture = config.architectures[0]
    print('Loading tokenizer.....')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print('Completed to load config & tokenizer')

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


if __name__ == "__main__":
    DEBUG = True
    USE_DDP_TRAINING = True
    OUTPUT_DIR = "checkpoints/"
    DRIVER_DATA_PATH = 'https://drive.google.com/file/d/1QpgvQi6mFvN5-6ofmJunDbuz34tlLbLL/view?usp=sharing'

    backend = "nccl"
    model_path = 'bigscience/bloom-560m'
    data_path = 'alpaca_data.json'
    size_valid_set = 0.1
    max_length = 128
    num_epochs = 3
    batch_size = 4
    gradient_accumulation_steps = 8

    learning_rate = 1e-5
    lr_scheduler_type = 'cosine'
    num_warmup_steps = 100
    weight_decay = 0.06

    use_bf16 = False
    seed = 0
    log_freq = 1
    eval_freq = 150
    
    if DEBUG == False:
        # Download data
        download_from_driver(path= DRIVER_DATA_PATH, location_path= data_path)
    
    if USE_DDP_TRAINING:
        init_process_group(backend=backend)
        local_rank =  int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        print(f"\ncurrent_device = { torch.cuda.current_device()} | local_rank = {local_rank}")
    else:
        local_rank = 0

    # Prepare model
    model = load_pretrained_model()
    # Get tokenizer
    tokenizer = load_tokenizer_from_pretrained_model(model_path = model_path)

    # prepare trainer
    trainer = Trainer(
        model = model, 
        num_epochs = num_epochs,
        max_length = max_length,
        batch_size = batch_size,
        gpu_id=local_rank,
        tokenizer=tokenizer,
        output_dir= OUTPUT_DIR,
        is_ddp_training = USE_DDP_TRAINING)
    
    # set ddp for wraping model
    # execute trainer 
    trainer.run(
        data_path = data_path,
        size_valid_set = size_valid_set,
        seed =seed
    )
    if USE_DDP_TRAINING:
        destroy_process_group()







