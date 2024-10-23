# For Google Colab, logging in to HuggingFace.
!pip install datasets peft transformers
from google.colab import userdata
my_secret_key = userdata.get('Cli2')
from huggingface_hub import login
login(my_secret_key)

# Name for finetuned model and folder.
model_output = "./BudgetAdvisor"

# Dataset loading and manipulation.
from datasets import load_dataset
dataset = load_dataset("gbharti/finance-alpaca") # features: ['text', 'instruction', 'input', 'output']
# Remove empty columns from dataset.
dataset = dataset.remove_columns(["text", "input"])
# Splits dataset to test and train sets, 90 % for train and 10 % for test.
dataset = dataset["train"].train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

#Tokenizer and model settings.
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", use_fast=True)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))

# For memory efficiency.
model.gradient_checkpointing_enable()

from peft import LoraConfig, get_peft_model
# Define a PEFT configuration for LoRA
lora_config = LoraConfig(
    r=8,  # Reduced rank for faster training
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Check if cuda is available.
import torch
model = get_peft_model(model, lora_config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Preprocessing function
def preprocess_data(examples):
    # Combine instruction and input as the prompt
    inputs = [f"Instruction: {instr}\nInput: {inp}\n" for instr, inp in zip(examples['instruction'], examples['output'])]
    targets = [output for output in examples['output']]
    return {'input_text': inputs, 'target_text': targets}

train_dataset = train_dataset.map(preprocess_data, batched=True)
eval_dataset = eval_dataset.map(preprocess_data, batched=True)

# Tokenization function
def tokenize_data(examples):
    model_inputs = tokenizer(
        examples['input_text'],
        max_length=128,  # Reduced max_length for faster processing
        truncation=True,
        padding="max_length"
    )
    labels = tokenizer(
        examples['target_text'],
        max_length=128,
        truncation=True,
        padding="max_length"
    )["input_ids"]
    model_inputs["labels"] = labels
    return model_inputs

# Tokenize the datasets
train_dataset = train_dataset.map(tokenize_data, batched=True, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(tokenize_data, batched=True, remove_columns=eval_dataset.column_names)

# Set the format for PyTorch tensors
train_dataset.set_format(type="torch")
eval_dataset.set_format(type="torch")

# Training arguments and trainer.
training_args = TrainingArguments(
    output_dir=model_output, # "./BudgetAdvisor"
    per_device_train_batch_size=8,  # Increase if GPU memory allows
    per_device_eval_batch_size=8,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    num_train_epochs=3,  # Increased epochs for better training
    learning_rate=5e-5,
    fp16=True,  # Enable mixed precision for faster training
    logging_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none",  # Disable reporting to third-party services
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer    
)

# Message for testing.
print("Trainer is set up!")

# 3. Trains the model and saves it.
trainer.train()
print("Model trained!")
trainer.save_model(model_output) # "./BudgetAdvisor"
tokenizer.save_pretrained(model_output)# "./BudgetAdvisor"

!zip -r BudgetAdvisor.zip ./BudgetAdvisor