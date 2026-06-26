from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 

def load_base_model():
    pass

ds = load_dataset("HuggingFaceH4/no_robots")
print(ds["train"][0])