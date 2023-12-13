from transformers import AutoConfig, TrainingArguments
import os
from deepspeed.utils import logger as ds_logger


def get_model_config(model_name, context_length: int=4096):
    """Wrapper for AutoConfig.from_pretrained(). 
    
        return (Config, unused_kwargs)"""
    ds_logger.info("Setting Configuration...")
    return AutoConfig.from_pretrained(
        model_name,
        return_unused_kwargs=True,
        max_position_embeddings=context_length,
        use_cache=False,    
		### 밑 설정값을 바꿔서 모델 파라미터를 변경할 수 있다.
    ### below are attention is all you need transformer base parameters
        # hidden_size = 4096,
        # intermediate_size = 4096,
        # num_hidden_layers = 6,
        # num_attention_heads = 8,
        # num_key_value_heads = 8,
        )


def get_training_arguments():
    """Wrapper for TrainingArguments"""
    
    return TrainingArguments(
        output_dir="/home/ubuntu/llama2-test/checkpoints",
        deepspeed = "/home/ubuntu/llama2-test/pretrain/ds_config.json", # .json 파일 또는 dict. deepspeed 설정값 전달.
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
        report_to="none", # disable wandb
        save_strategy="steps",
        save_total_limit=1,
        overwrite_output_dir=True,
        save_on_each_node=True, # Multi-node 시 공유 메모리 없으면 각 노드에다 체크포인트 저장
    )