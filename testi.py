# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizerFast

tokenizer = LlamaTokenizerFast.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
print("1/2")
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("2/2")


def generate_text(prompt, max_length=50, temperature=1.0):
    # Tokenize the input prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate text using the model
    outputs = model.generate(
        inputs, 
        max_length=max_length, 
        temperature=temperature, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2,
        do_sample=True
    )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

print("3")
import gradio as gr

# Create the Gradio interface
def generate_interface(prompt, max_length, temperature):
    return generate_text(prompt, max_length=max_length, temperature=temperature)

interface = gr.Interface(
    fn=generate_interface,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your prompt here", label="Prompt"),
        gr.Slider(10, 100, value=50, label="Max Length"),
        gr.Slider(0.7, 1.3, value=1.0, label="Temperature")
    ],
    outputs="text",
    title="Text Generation with Llama",
    description="Muutettava"
)
print("4")
interface.launch()
    