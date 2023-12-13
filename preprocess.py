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
        num_proc=16
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
    
    data_path = 'data/news_data_2gb.jsonl'
    data_chunk_size = 1024
    dataset_prefix = "2gb"
    # -> save_path = data/{dataset_prefix}_{data_chunk_size}
    
    raw_dataset = make_HFdataset(data_path)

    model_name = "meta-llama/Llama-2-7b-hf"
    os.environ["TOKENIZERS_PARALLELISM"] = "false" # avoid warning: avoid using tokenizer before the fork
    
    print("Load Pretrained Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        add_bos_token=True,
        add_eos_token=True,
    ) # Default Llama Tokenizer doesn't add eos_token automatically
    # tokenizer.pad_token = tokenizer.unk_token # 토크나이저의 패딩토큰을 종료토큰으로 설정
    print("Loaded Pretrained Llama2 Tokenizer!\n")

    dataset = preprocess_dataset(tokenizer, data_chunk_size, raw_dataset)
    
    save_path = f"data/{dataset_prefix}_{data_chunk_size}"
    os.makedirs(save_path, exist_ok=True)
    dataset.save_to_disk(save_path)
    
    print("Dataset saved at: ", save_path)
    
    # dataset.push_to_hub("lectura/my-dataset", private=True) # HF hub에 업로드