import jsonlines
import datasets as ds
from functools import partial

def make_HFdataset(data_path):
    print("Transforming jsonl -> hf datasets...")
    with jsonlines.open(data_path) as f:
        data = [l for l in f] 

    data_dict = {
        'text': [d['text'] for d in data]
    }

    raw_dataset = ds.Dataset.from_dict(data_dict)
    return raw_dataset

def split_chunk(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i+chunk_size]

def tokenize(batch, tokenizer):
    return tokenizer(
        batch['text'],
        truncation=False
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
    
    DATA_PATH = 'data/news_data_2gb.jsonl'
    CHUNK_SIZE = 4096
    DATASET_PREFIX = "news"
    TOKENIZER_ID = "meta-llama/Llama-2-7b-hf"
    
    raw_dataset = make_HFdataset(DATA_PATH)

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