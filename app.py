import torch
from transformers import CodeGenTokenizerFast, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import gradio as gr

# Load tokenizer and model here (your existing code)
model_path = "phi4_finetuned/phi4_finetuned"
base_model_name = "microsoft/Phi-3.5-mini-instruct"

tokenizer = CodeGenTokenizerFast.from_pretrained(model_path)
quant_config = BitsAndBytesConfig(load_in_4bit=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,
    device_map="cuda" if torch.cuda.is_available() else "cpu"
)
model = PeftModel.from_pretrained(base_model, model_path)
model.eval()

generation_args = {
    "max_new_tokens": 1000,
    "temperature": 0.001,
    "do_sample": False,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
}

def generate_response(messages):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_args)

    input_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_len:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

def gradio_generate(text):
    messages = [{"role": "user", "content": text}]
    return generate_response(messages)

demo = gr.Interface(fn=gradio_generate, inputs="text", outputs="text", title="My Model Chat")
demo.launch()
