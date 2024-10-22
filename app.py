# Excercise 3 is used as a base.

# 1. Dataset loading, manipulation and tokenization!
# Loads dataset.
from datasets import load_dataset
#dataset = load_dataset("gbharti/finance-alpaca") # features: ['text', 'instruction', 'input', 'output']
dataset = load_dataset("json", data_files="final_dataset_clean.json") # features: ['text', 'instruction', 'input', 'output']

# Remove unnecessary features.
dataset = dataset.remove_columns(["input"])
# Splits dataset to test and train sets.
dataset = dataset["train"].train_test_split(test_size=0.2)

# Imports tokenizer
from transformers import AutoTokenizer, LlamaTokenizerFast
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

print("Tokenizer vocab_size: ", tokenizer.vocab_size) # 128000
# Fixing ValueError: Asking to pad but the tokenizer does not have a padding token.
tokenizer.add_special_tokens({'pad_token': '<pad>'}) # [PAD]
# Function for tokenization.
def tokenize_function(examples):
    return tokenizer(examples['instruction'], padding='max_length', truncation=True, max_length=64)#128)
tokenized_datasets = dataset.map(tokenize_function, batched=False)
print(tokenized_datasets)


# 2. Setting up trainer.
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
print("Tokenizer vocab_size: ") # 128000, Llama has 32000! -> IndexError: index out of range in self. https://huggingface.co/docs/transformers/model_doc/llama2#transformers.LlamaConfig
print(model)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,# 1
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy='epoch',
    logging_dir='./logs'
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'].shuffle().select(range(1000)),
    eval_dataset=tokenized_datasets['test'].shuffle().select(range(1000)),
    #tokenizer=tokenizer    
)
print("Trainer is set up!")
#print(trainer.tokenizer)


## 3. Trains the model and saves it.
trainer.train()
print("Model trained!")
trainer.save_model("BudgetAdvisor")
tokenizer.save_pretrained("BudgetAdvisor")
# https://stackoverflow.com/questions/74926735/the-training-function-is-throwing-an-index-out-of-range-in-self-error
# https://stackoverflow.com/questions/71984994/while-training-bert-variant-getting-indexerror-index-out-of-range-in-self
# https://discuss.huggingface.co/t/fine-tuning-throws-index-out-of-range-in-self/58899/3


"""
# Get HuggingFace Llama token from .env file and log in to HuggingFace.
import os
from dotenv import load_dotenv
from huggingface_hub import login
#try:
#    load_dotenv()
#    token_llama = os.getenv("HF_TOKEN")
#    login(token=token_llama)
#except ValueError:
#    print("Cant log in!")

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B")
print("Pipe")
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
print("tokenizer")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
print("model")

#import gradio as gr

print(ds)
"""








#from huggingface_hub import InferenceClient 

# Use a pipeline as a high-level helper
#from transformers import pipeline
#pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B")
## Load model directly
#from transformers import AutoTokenizer, AutoModelForCausalLM
#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
#model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")



"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
"""
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")


def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content

        response += token
        yield response


#For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface

demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)


if __name__ == "__main__":
    demo.launch()
"""