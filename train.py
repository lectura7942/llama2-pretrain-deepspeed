import os
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import datasets as ds
from deepspeed.utils import logger as ds_logger

from model import load_model
from configs import get_training_arguments, DATASET

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

if torch.cuda.is_available():
    device = torch.device("cuda")
    n_gpus = torch.cuda.device_count()
    ds_logger.info(f"device: {device}, {n_gpus}")
else:
    device = torch.device("cpu")
    ds_logger.info(f"device: {device}")

### if data_path is on local disk
dataset = ds.load_from_disk(DATASET)

### if data_path is huggingface_hub repo
# dataset = ds.load_dataset(DATASET, split="train")

ds_logger.info(f"dataset: {dataset}")
ds_logger.info(f"len(dataset[0]['input_ids']): {len(dataset[0]['input_ids'])}")

model, tokenizer = load_model()

model.train()
trainer = Trainer(
    model=model, 
    # tokenizer=tokenizer,
    args=get_training_arguments(),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    train_dataset=dataset
)
trainer.train()
ds_logger.info("Complete Train")
trainer.save_model()
ds_logger.info("Save Model")

# trainer.push_to_hub()