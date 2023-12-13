### model.py

import torch
from transformers import AutoConfig, LlamaForCausalLM, AutoTokenizer
import os
from deepspeed.utils import logger as ds_logger
import deepspeed

from configs import get_model_config

### 토크나이저 & 모델을 로드하는 함수 작성
def load_model(model_name, context_length:int):
    ds_logger.info("Setting Configuration...")
    config, unused_kwargs = get_model_config(model_name, context_length)
    ds_logger.info(f"unused kwargs: {unused_kwargs}")
    ds_logger.info("Llama2 Configuration Complete !\n")
    
    """모델 하나가 GPU 한 개에 다 못 올라가므로 분산해서 로드할 것임을 선언
    - 가중치 float16으로 변환
    - 다중 gpu로 분산
    - gpu 메모리 부족하면 cpu에 offload
    - ! zero stage 3 필요 !
    """
    with deepspeed.zero.Init():
        ds_logger.info("Load Model based on config...")
        model = LlamaForCausalLM(config)
        ds_logger.info("Loaded Initialized Model based on config!\n")
        

    os.environ["TOKENIZERS_PARALLELISM"] = "false" # avoid warning: avoid using tokenizer before the fork
    ds_logger.info("Load Pretrained Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        add_bos_token=True,
        add_eos_token=True
    )
    # Needed for Llama Tokenizer
    tokenizer.pad_token = tokenizer.eos_token # 토크나이저의 패딩토큰을 종료토큰으로 설정
    ds_logger.info("Loaded Pretrained Llama2 Tokenizer!\n")
    return model, tokenizer

### 모델 설정을 기반으로 최대 시퀀스 길이를 계산해서 반환하는 함수(기본값 2048)
def get_max_length(model):
    max_length = None

    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            ds_logger.info(f"Found max lenth: {max_length}")
            break

    if not max_length:
        max_length = 2048
        ds_logger.info(f"Using default max length: {max_length}")

    return max_length