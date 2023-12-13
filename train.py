import os
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import datasets as ds
from deepspeed.utils import logger as ds_logger # 그냥 print 사용하면 GPU 개수 마다 반복함.

from model import load_model, get_max_length
from configs import get_training_arguments

data_path = "leeseeun/tokenzied_news_2gb_data" # at HF repo. 1024 chunk size
base_model_id = "meta-llama/Llama-2-7b-hf"
context_length = 1024 # 학습 데이터 chunk size와 동일하게 설정함. Llama2는 4096.

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

if torch.cuda.is_available():
    device = torch.device("cuda")
    n_gpus = torch.cuda.device_count()
    ds_logger.info(f"device: {device}, {n_gpus}")
else:
    device = torch.device("cpu")
    ds_logger.info(f"device: {device}")

### if data_path is on local disk
# dataset = ds.load_from_disk(data_path)

### if data_path is huggingface_hub repo
dataset = ds.load_dataset(data_path, split="train")

ds_logger.info(f"dataset: {dataset}")
ds_logger.info(f"len(dataset[0]['input_ids']): {len(dataset[0]['input_ids'])}")

model, tokenizer = load_model(base_model_id, context_length)

args = get_training_arguments()
model.train()
trainer = Trainer(
    model=model, 
    tokenizer=tokenizer,
    args=args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    train_dataset=dataset
)
trainer.train()
ds_logger.info("Complete Train")
trainer.save_model()
ds_logger.info("Save Model")