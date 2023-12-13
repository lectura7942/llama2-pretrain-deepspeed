import jsonlines
import datasets as ds
from functools import partial

def make_HFdataset(data_path, source: str = "jsonl", source_column:str="text"):
    """make HF dataset from jsonlines or HF dataset.
    - `data_path`: path of jsonlines file or HF dataset
    - `source`: `hf` to download from hub or `jsonl` to get from jsonlines file
    - `source_column`: This column name will be chaned to "text"
    """
    
    if source == "hf":
        
        """hf dataset -> text only hf dataset"""
        raw_dataset = ds.load_dataset(data_path, split="train")
        
        raw_dataset = raw_dataset.map(
            lambda x: {"text": x[source_column]}, 
            batched=True, remove_columns=raw_dataset.column_names
            )
        
    elif source == "jsonl":
        """jsonl -> hf dataset"""
        print("Transforming jsonl -> hf datasets...")
        with jsonlines.open(data_path) as f:
            data = [l for l in f] 

        data_dict = {
            'text': [d[source_column] for d in data]
        }

        raw_dataset = ds.Dataset.from_dict(data_dict)
        
    return raw_dataset

def split_chunk(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i+chunk_size]

def tokenize(batch, tokenizer):
    return tokenizer(
        batch['text'],
        # REMOVE max_length
    )

def preprocess_dataset(tokenizer, max_length, raw_dataset):
    print("Preprocessing datset...")

    _tokenize = partial(tokenize, tokenizer=tokenizer)
    tokenized_dataset = raw_dataset.map(
        _tokenize,
        batched=True,
        remove_columns=raw_dataset.column_names,
        num_proc=8
    )

    input_ids = []
    for ids in tokenized_dataset['input_ids']:
        input_ids.extend(ids)
    
    input_batch = []
    for chunk in list(split_chunk(input_ids, chunk_size=max_length)):
        if len(chunk) == max_length:
            input_batch.append(chunk)
    temp = {"input_ids": input_batch}
    dataset = ds.Dataset.from_dict(temp)
    return dataset


if __name__ == "__main__":
    import os
    from transformers import AutoTokenizer
    
    DATA_PATH = "daekeun-ml/naver-news-summarization-ko"
    CHUNK_SIZE = 1024 # Llama2-7b는 4096
    DATASET_PREFIX = "naver_news"
    TOKENIZER_ID = "beomi/llama-2-ko-7b"
    
    raw_dataset = make_HFdataset(DATA_PATH, source="hf", source_column="document")

    print("Load Pretrained Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ID,
        add_bos_token=True,
        add_eos_token=True,
    ) # Default Llama Tokenizer doesn't add eos_token automatically
    # tokenizer.pad_token = tokenizer.unk_token # 토크나이저의 패딩토큰을 종료토큰으로 설정
    print("Loaded Pretrained Llama2 Tokenizer!\n")

    dataset = preprocess_dataset(tokenizer, CHUNK_SIZE, raw_dataset)
    
    save_path = f"data/{DATASET_PREFIX}_{CHUNK_SIZE}_{TOKENIZER_ID.split('/')[-1]}"
    os.makedirs(save_path, exist_ok=True)
    dataset.save_to_disk(save_path)
    
    print("Dataset saved at: ", save_path)
    
    # dataset.push_to_hub("lectura/my-dataset", private=True) # HF hub에 업로드