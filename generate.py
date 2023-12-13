### generate.py

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from model import load_model, get_max_length
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
from tqdm import tqdm 
model_path = "/home/ubuntu/llama2-test/checkpoints"
   
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-hf", 
    add_bos_token=True, 
    add_eos_token=True,
    ) # default don't add eos

# eod tokens added in training
model = LlamaForCausalLM.from_pretrained(
        model_path,
        # load_in_8bit = True,
        # device_map="auto", # need accelerate
        torch_dtype=torch.float16,
)
    # TODO add pad id. Currently not using
    # model.config.pad_token_id = tokenizer.pad_id
eval_dataset = [
        "I wanna go",
    ]
results = []
device = torch.device("cuda:0")
        
model.to(device).eval()
    
for idx, d in tqdm(enumerate(eval_dataset), desc="Generating", total=len(eval_dataset)):
    prompt = d
    model_input = tokenizer(prompt, return_token_type_ids=False,return_tensors="pt").to(device)
    with torch.no_grad():
        output_text = tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], )
    results.append(output_text)
for r in results: 
    print(r)