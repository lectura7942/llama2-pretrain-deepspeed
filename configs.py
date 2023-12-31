from transformers import AutoConfig, TrainingArguments
import os
from deepspeed.utils import logger as ds_logger

BASE_MODEL_ID = "meta-llama/Llama-2-7b-hf"
TOKENIZER_ID = "beomi/llama-2-ko-7b"
CHUNK_SIZE = 4096
DATASET = "data/naver_news_4096_llama-2-ko-7b"

def get_model_config():
    """Wrapper for AutoConfig.from_pretrained(). 
    
        return (Config, unused_kwargs)"""
    ds_logger.info("Setting Configuration...")
    return AutoConfig.from_pretrained(
        BASE_MODEL_ID,
        return_unused_kwargs=True,
        use_cache=False,    
### 밑 설정값을 바꿔서 모델 파라미터를 변경할 수 있다.
        max_position_embeddings=CHUNK_SIZE,
        # hidden_size = 4096,
        # intermediate_size = 11008,
        # num_hidden_layers = 32,
        # num_attention_heads = 32,
        # num_key_value_heads = 32,
        )


def get_training_arguments():
    """Wrapper for TrainingArguments"""
    
    return TrainingArguments(
        output_dir="checkpoints",
        deepspeed = "ds_config.json", # .json 파일 또는 dict. deepspeed 설정값 전달.
        fp16=True,
        gradient_checkpointing=True,

        optim="adamw_torch",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        weight_decay=0.1, 
        max_grad_norm = 1.0,
    #   num_train_epochs=1, # max_steps 설정되어 있으면 무시됨
        max_steps = 1,

        warmup_steps=0, # default 0
        learning_rate=0.0001, # default 0.00005
        lr_scheduler_type="cosine",

        logging_steps=1,
        report_to="none", # disable wandb if logged in
        save_strategy="no", # deepspeed zero3는 체크포인트 오류 난다.
        overwrite_output_dir=True,
        # save_on_each_node=True, # Multi-node 시 공유 메모리 없으면 각 노드에다 체크포인트 저장
    )