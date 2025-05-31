from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

# Load model and tokenizer once on startup
model_path = "phi4_finetuned/phi4_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map="auto"
)
model.eval()

generation_args = {
    "max_new_tokens": 1000,
    "temperature": 0.001,
    "do_sample": False,  # deterministic due to low temperature
    "top_p": 0.9,
    "repetition_penalty": 1.1,
}

class Message(BaseModel):
    role: str
    content: str

class GenerateRequest(BaseModel):
    messages: List[Message]

@app.post("/generate")
async def generate(request: GenerateRequest):
    messages = request.messages
    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    formatted_prompt = tokenizer.apply_chat_template(
        [message.dict() for message in messages],
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_args)

    input_length = len(inputs["input_ids"][0])
    generated_tokens = outputs[0][input_length:]
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return {"response": response_text}
