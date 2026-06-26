from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "Qwen/Qwen2.5-1.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

model = model.to(device)

inputs = tokenizer("The capital of germany is ", return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=25)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
