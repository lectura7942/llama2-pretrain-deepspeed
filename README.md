# llama2-pretrain-deepspeed
LLaMA2-7B를 다중 gpu 환경에서 사전 학습하는 코드이다. HF Trainer와 DeepSpeed를 사용한다.

* 데이터: `daekeun-ml/naver-news-summarization-ko` (네이버 뉴스)
* 토크나이저: `beomi/llama-2-ko-7b` (영어 + 한국어)

[추가 설명 링크](https://jkspace.notion.site/Llama2-7B-DeepSpeed-Trainer-49e78c44b7304f0388c37a95964f953d)

# :round_pushpin: [실습] 10분 만에 (매우 작은) 라마 사전학습하기

* gpu: 24Gib x 8 (작은 모델을 사용할 거라서 하나로도 충분하다.)
* `BASE_MODEL_ID = nickypro/tinyllama-110M`
* `CHUNK_SIZE = 512`
* global batch: 32 (`per_device_train_batch_size=16`)
* `max_steps = 1000`

## 1. 환경 설정

### 1-1. 코드 다운받기

```
git clone https://github.com/lectura7942/llama2-pretrain-deepspeed.git
cd llama2-pretrain-deepspeed
```

### 1-2. virtualenv 가상환경 설정

다른 가상환경을 사용하면 넘어가도 좋다.

* viertualenv를 설치한다.

    ```python
    sudo apt python3-virtualenv
    ```

    sudo 권한이 없으면:

    ```python
    python3 -m pip install --user -U virtualenv
    ```

* 현재 위치에 가상환경 폴더를 생성한다. 
  

    ```
    virtualenv .venv -p python3.8
    ```

    :warning: 명시한 파이썬 버전이 설치되어 있지 않으면 오류가 난다.

* 가상환경을 활성화한다.
  
    ```
    source .venv/bin/activate
    ```

### 1-3. 패키지 다운받기

* 먼저 cuda에 맞는 torch가 설치되어 있는지 확인한다.
  * CUDA 버젼 확인: `nvcc -V`
  * `nvcc`가 설치되어 있지 않다고 하면 넘어가고 나중에 학습 코드 돌릴 때 오류 나는지 확인하면 된다.

  * **(참고) torch 2.1.1 CUDA 11.8**

    ```
    pip install torch --index-url https://download.pytorch.org/whl/cu118
    ```

* 패키지 설치
    ```
    pip install -r requirements.txt
    ```

(참고) mpi4py 오류 생기면
  
```
sudo apt install mpich
```

## 2. 데이터셋 생성

* 설정 확인
    ```python
    DATA_PATH = "daekeun-ml/naver-news-summarization-ko"
    DATA_TYPE = "hf" # hf or jsonl
    SOURCE_COLUMN = "document" # 데이터셋에서 학습 데이터로 사용할 텍스트 부분
    CHUNK_SIZE = 4096 # 모델 최대 입력 길이와 같다
    DATASET_PREFIX = "naver_news" # 생성할 데이터셋 이름
    TOKENIZER_ID = "beomi/llama-2-ko-7b"
    ```
    
    * **(1) HuggingFace 데이터셋 사용**

        `DATA_TYPE=hf`을 사용한다.

        `SOURCE_COLUMN`을 학습 데이터로 사용할 텍스트 부분의 칼럼명으로 수정한다.

    * **(2) jsonl 파일 사용**

        `DATA_TYPE=jsonl`을 사용한다.

        jsonl 파일에는 다음 형식으로 데이터가 들어있어야 한다.

        ```
        {"document": "An english news text string"}
        {"document": "In jsonlines format"}
        ```

        `SOURCE_COLUMN`을 키값으로 수정한다. ('document'가 아니면)

    :round_pushpin: **[실습]** `CHUNK_SIZE=512`

* 실행
    ```
    python preprocess.py
    ```

실행이 완료되면 데이터셋이 저장된 경로가 출력된다.

`Dataset saved at:  data/naver_news_4096_llama-2-ko-7b`

## 3. 학습

### 3-1. `configs.py`에서 모델, 학습 설정 확인

```python
BASE_MODEL_ID = "meta-llama/Llama-2-7b-hf" # 모델 구조만 사용한다.
TOKENIZER_ID = "beomi/llama-2-ko-7b"
CHUNK_SIZE = 4096
DATASET = "data/naver_news_4096_llama-2-ko-7b"
```

:round_pushpin: **[실습]**

* `BASE_MODEL_ID = nickypro/tinyllama-110M"`
* `CHUNK_SIZE = 512`
* `DATASET`은 2번에서 만든 데이터셋 경로로 수정한다.

#### :hammer: 모델 파라미터 수정

`BASE_MODEL_ID`의 구조에서 바꾸고 싶은 파라미터가 있다면 밑 함수에서 직접 수정해도 된다.

```python
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
```

:round_pushpin: **[실습]** 수정할 필요가 없다.

#### :hammer: 학습 설정값 수정

```python
def get_training_arguments():
    """Wrapper for TrainingArguments"""
    
    return TrainingArguments(
        output_dir="checkpoints", # 모델을 저장할 폴더 이름
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
```

:round_pushpin: **[실습]**

* `per_device_train_batch_size=16`
* `max_steps = 1000`

### 3-2. `ds_config.json`에서 DeepSpeed 설정 확인

GPU 메모리가 충분하면 cpu offload를 비활성화 하는 게 좋다. cpu offload는 학습 속도를 느리게 하기 때문이다.

`ds_config.json`에서 다음 부분을 삭제하면 된다.

```json
"offload_optimizer": {
    "device": "cpu"
},
"offload_param": {
    "device": "cpu"
},
```

:round_pushpin: **[실습]** cpu offload를 비활성화 한다.

### 3-3. 학습 실행

```
deepspeed train.py
```

자동으로 모든 GPU를 사용한다.

:warning:
다음 오류가 뜨면 참고해서 cuda 버전에 맞는 torch를 다시 설치해준다.

```
DeepSpeed Op Builder: Installed CUDA version 11.8 does not match the version torch was compiled with 12.1, unable to compile cuda/cpp extensions without a matching cuda version.
```

## 4. 결과 확인

GPU/CPU 하나만을 사용한다.

* 설정 확인
  
    ```
    MODEL_ID = "checkpoints"
    TOKENIZER_ID = "beomi/llama-2-ko-7b"
    EVAL_TEXTS = [
            "금융정보를 받은 통신사는 인공지능 AI 을 통해",
            "미국 증시가 "
        ]
    ```

    `MODEL_ID`는 3번에서 학습한 모델 경로로 수정한다.

* 실행
    ```
    python generate.py
    ```
