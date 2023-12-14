### model.py

import torch
from transformers import AutoConfig, LlamaForCausalLM, AutoTokenizer
import os
from deepspeed.utils import logger as ds_logger
import deepspeed

from configs import get_model_config, BASE_MODEL_ID, TOKENIZER_ID

def load_model():
    """return (model, tokenizer)"""
    
    ds_logger.info("Setting Configuration...")
    config, unused_kwargs = get_model_config()
    ds_logger.info(f"unused kwargs: {unused_kwargs}")
    ds_logger.info("Llama2 Configuration Complete !\n")
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false" # avoid warning: avoid using tokenizer before the fork
    ds_logger.info("Load Pretrained Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ID if len(TOKENIZER_ID) > 0 else BASE_MODEL_ID,
        add_bos_token=True,
        add_eos_token=False
    )
    # Needed for Llama Tokenizer
    tokenizer.pad_token = tokenizer.unk_token
    ds_logger.info("Loaded Pretrained Llama2 Tokenizer!\n")
    
    """모델 하나가 GPU 한 개에 다 못 올라가므로 분산해서 로드할 것임을 선언
    - 가중치 float16으로 변환
    - 다중 gpu로 분산
    - gpu 메모리 부족하면 cpu에 offload
    - ! zero stage 3 필요 !
    """
    with deepspeed.zero.Init():
        ds_logger.info("Load Model based on config...")
        config.vocab_size = len(tokenizer) # 모델과 토크나이저가 다를 때 모델의 임베딩 크기 수정
        model = LlamaForCausalLM(config)
        ds_logger.info("Loaded Initialized Model based on config!\n")
        

    
    return model, tokenizer