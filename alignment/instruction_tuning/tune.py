from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 
import yaml 
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from transformers import TrainingArguments

with open("config.yaml", "r") as file:
    CONFIG = yaml.safe_load(file)

MODEL_NAME = CONFIG["model_name"]
RANK = CONFIG["lora_config"]["r"]
ALPHA = CONFIG["lora_config"]["alpha"]
DROPOUT = CONFIG["lora_config"]["dropout"]
LR = CONFIG["train_config"]["lr"]
GRAD_ACC_STEPS = CONFIG["train_config"]["grad_acc_steps"]

lora_config = LoraConfig(
    r=RANK,
    lora_alpha=ALPHA,
    lora_dropout=DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ]
)

TOKENIZER = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

MODEL = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cuda",
    torch_dtype="auto"
)
MODEL = get_peft_model(MODEL, lora_config)
MODEL.print_trainable_parameters()

def format_example(sample, tokenizer=TOKENIZER):
    text = tokenizer.apply_chat_template(
        sample["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

ds = load_dataset("HuggingFaceH4/no_robots")
training_args = TrainingArguments(
    output_dir="./qwen2.5-1.5b-lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=GRAD_ACC_STEPS,
    learning_rate=LR,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=500,
    bf16=True,
    optim="adamw_torch",
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="none"
)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    dataset_text_field="text",
    args=training_args,
    max_seq_length=1024
)
trainer.train()
model.save_pretrained("./qwen_no_robots_lora")
tokenizer.save_pretrained("./qwen_no_robots_lora")
print(ds["train"][0]["messages"])