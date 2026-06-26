from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import PeftModel
from trl import DPOTrainer
import torch


BASE_MODEL = "Qwen/Qwen2.5-1.5B"
SFT_ADAPTER = "../qwen_no_robots_lora"

#Later, remove my username
HF_REPO = "chandanreddy/qwen2.5-1.5b-safety-dpo"

dataset = load_dataset("LLM-LAT/harmful-dataset")

print(dataset["train"].column_names)


tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)


model = PeftModel.from_pretrained(
    base_model,
    SFT_ADAPTER,
    is_trainable=True,
)

model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./qwen_safety_dpo",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    report_to="none",
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=dataset["train"],
    processing_class=tokenizer,
)

trainer.train()

trainer.save_model("./qwen_safety_dpo")
tokenizer.save_pretrained("./qwen_safety_dpo")

trainer.model.push_to_hub(
    HF_REPO,
    commit_message="Safety DPO LoRA",
)

tokenizer.push_to_hub(HF_REPO)

print("Finished.")