from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
import torch
import yaml
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from huggingface_hub import HfApi

print("Loading Config.yaml..................")
with open("config.yaml", "r") as file:
    CONFIG = yaml.safe_load(file)

MODEL_NAME = CONFIG["model_name"]
RANK = CONFIG["lora_config"]["r"]
ALPHA = CONFIG["lora_config"]["alpha"]
DROPOUT = CONFIG["lora_config"]["dropout"]

LR = CONFIG["train_config"]["lr"]
GRAD_ACC_STEPS = CONFIG["train_config"]["grad_acc_steps"]

TOKENIZER = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

# important for Qwen
if TOKENIZER.pad_token is None:
    TOKENIZER.pad_token = TOKENIZER.eos_token

MODEL = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
)

# LoRA config
lora_config = LoraConfig(
    r=RANK,
    lora_alpha=ALPHA,
    lora_dropout=DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

print("Loading Model with LORA Config.....................")
MODEL = get_peft_model(MODEL, lora_config)
MODEL.print_trainable_parameters()

#Dataset
ds = load_dataset("HuggingFaceH4/no_robots")

def format_example(sample):
    text = TOKENIZER.apply_chat_template(
        sample["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

ds = ds.map(format_example, remove_columns=ds["train"].column_names)

# Train Args
training_args = TrainingArguments(
    output_dir="./qwen2.5-1.5b-lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=GRAD_ACC_STEPS,
    learning_rate=LR,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=500,
    bf16=torch.cuda.is_available(),
    optim="adamw_torch",
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="none"
)
# Trainer
trainer = SFTTrainer(
    model=MODEL,
    train_dataset=ds["train"],
    args=training_args
)

print("Start Training...................")
trainer.train()

trainer.model.save_pretrained("./qwen_no_robots_lora")
TOKENIZER.save_pretrained("./qwen_no_robots_lora")

print("Pushing the Models to Hub.")
##Remove my user Name
trainer.model.push_to_hub(
    "chandanreddy/qwen-no-robots-lora"
)

TOKENIZER.push_to_hub(
    "chandanreddy/qwen-no-robots-lora"
)