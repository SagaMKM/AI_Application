from transformers import pipeline
import torch
import gradio as gr
# For loading finetuned model.
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, PeftConfig
from transformers import AutoModelForTokenClassification, AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer

# Load finetuned model from folder finetuned-financial-model.
# https://stackoverflow.com/questions/78552651/how-to-fix-error-oserror-model-does-not-appear-to-have-a-file-named-config-j
model_llm = "meta-llama/Llama-3.2-1B"
model_fine_tuned = "./finetuned-financial-model"
tokenizer_id = "./finetuned-financial-model" # Fix!
model_id = AutoModelForCausalLM.from_pretrained(model_llm) 
model_id = PeftModel.from_pretrained(model_id, model_fine_tuned)
#model_id = AutoModelForCausalLM.from_pretrained(model_llm) # AutoModelForCausalLM
#model_id = LlamaForCausalLM.from_pretrained(model_id, model_fine_tuned) # PeftModel
#PeftModelForCausalLM
pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=AutoTokenizer.from_pretrained(model_llm, use_fast=True),
    #tokenizer=LlamaTokenizer.from_pretrained(model_llm, use_fast=True), # AutoTokenizer
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Define the function to generate the response
def generate_response(financial_goal, risk_tolerance, investment_horizon, monthly_income, monthly_expense, current_savings):
    # Construct the prompt with the six input fields
    prompt = f"""
    I need financial advice based on the following information:
 
    Financial Goal: {financial_goal}
    Risk Tolerance: {risk_tolerance}
    Investment Horizon (years): {investment_horizon}
    Monthly Income (€): {monthly_income}
    Monthly Expense (€): {monthly_expense}
    Current Savings (€): {current_savings}
 
    Please provide a detailed financial plan.
    """
 
    # Generate the response using the AI model
    outputs = pipe(
        prompt,
        max_new_tokens=1000,
    )
    response = outputs[0]["generated_text"]
    return response
 
# Create the Gradio interface
iface = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(label="Financial Goal"),
        gr.Textbox(label="Risk Tolerance"),
        gr.Number(label="Investment Horizon (years)"),
        gr.Number(label="Monthly Income (€)"),
        gr.Number(label="Monthly Expense (€)"),
        gr.Number(label="Current Savings (€)"),
    ],
    outputs=gr.Textbox(label="Financial Advice"),
    title="Financial Advisor",
    description="Enter your financial details to receive personalized advice.",
)
 

def main():
    # Launch the Gradio app
    iface.launch()

if __name__ == "__main__":
    main()