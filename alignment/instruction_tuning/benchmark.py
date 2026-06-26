import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
import numpy as np
import re 
import datasets

NUM_SAMPLES = 100
BASE_MODEL = "Qwen/Qwen2.5-1.5B"
ADAPTER_PATH = "./qwen_no_robots_lora" 
DATASET = load_dataset("LLM-LAT/harmful-dataset")["train"]

def sample_from_dataset(DATASET: datasets.Dataset, NUM_SAMPLES: int) -> datasets.Dataset:
   
    total_samples = np.arange(0, len(DATASET))
    subset_samples = np.random.choice(total_samples,size= NUM_SAMPLES)

    SUBSET = [DATASET[sample] for sample in subset_samples]
    
    return SUBSET

def remove_redundant_artifats(text: str) -> str:
    pattern = re.compile(r"[Aa]ssistant")

    matches = list(re.finditer(pattern, text))

    if len(matches) > 0:
        last_match = matches[-1]
        end_loc = last_match.end()

        substring = text[end_loc:]
    else:
        substring = text

    return substring

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

# base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# attach LoRA
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

model.eval()

def generate(prompt):
    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

SUBSET = sample_from_dataset(DATASET, NUM_SAMPLES)
results = []

for sample in SUBSET:
    prompt = sample["prompt"]
    out = generate(prompt)

    results.append({
        "prompt": prompt,
        "response": out
    })
    output = remove_redundant_artifats(out)
    print(prompt)
    print(output)
    break