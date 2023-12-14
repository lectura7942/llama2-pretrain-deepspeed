from transformers import AutoTokenizer, LlamaForCausalLM
import torch
from tqdm import tqdm 

MODEL_ID = "checkpoints"
TOKENIZER_ID = "beomi/llama-2-ko-7b"
EVAL_TEXTS = [
        "금융정보를 받은 통신사는 인공지능 AI 을 통해",
        "미국 증시가 "
    ]

device = torch.device(
    "cuda:0" if torch.cuda.is_available() 
    else "cpu"
    )

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_ID, 
    add_bos_token=True, 
    add_eos_token=False,
    ) # default don't add eos

# eod tokens added in training
model = LlamaForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
# TODO add pad id. Currently not using
# model.config.pad_token_id = tokenizer.pad_id

results = []
        
model.to(device).eval()
    
for idx, d in tqdm(enumerate(EVAL_TEXTS), desc="Generating", total=len(EVAL_TEXTS)):
    prompt = d
    model_input = tokenizer(prompt, return_token_type_ids=False,return_tensors="pt").to(device)
    with torch.no_grad():
        output_text = tokenizer.decode(
            model.generate(**model_input, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)[0], 
            skip_special_tokens=True,
            )
    results.append(output_text)
    
for idx, r in enumerate(results): 
    print(f"--- {idx} ---")
    print(r)