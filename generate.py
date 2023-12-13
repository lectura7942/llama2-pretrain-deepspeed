from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from model import load_model
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
from tqdm import tqdm 

MODEL_ID = "checkpoints"
TOKENIZER_ID = "beomi/llama-2-ko-7b" # "checkpoints"
   
tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_ID, 
    add_bos_token=True, 
    add_eos_token=False,
    ) # default don't add eos

# eod tokens added in training
model = LlamaForCausalLM.from_pretrained(
        MODEL_ID,
        # torch_dtype=torch.float16,
)
# TODO add pad id. Currently not using
# model.config.pad_token_id = tokenizer.pad_id
eval_dataset = [
        "금융정보를 받은 통신사는 인공지능 AI 을 통해",
    ]
results = []
device = torch.device(
    "cuda:0" if torch.cuda.is_available() 
    else "cpu"
    )
        
model.to(device).eval()
    
for idx, d in tqdm(enumerate(eval_dataset), desc="Generating", total=len(eval_dataset)):
    prompt = d
    model_input = tokenizer(prompt, return_token_type_ids=False,return_tensors="pt").to(device)
    with torch.no_grad():
        output_text = tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], )
    results.append(output_text)
    
for r in results: 
    print(r)