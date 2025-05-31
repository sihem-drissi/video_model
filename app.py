import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
from transformers import CodeGenTokenizerFast, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import uvicorn
import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Model and tokenizer setup
model_path = "/code/phi4_finetuned/phi4_finetuned"  # AML scoring directory
base_model_name = "Salesforce/codegen-350M-mono"  # Confirm from adapter_config.json

try:
    logger.info(f"Model path: {model_path}")
    logger.info(f"Files in model path: {os.listdir(model_path)}")
    logger.info(f"Memory usage before loading: {psutil.Process().memory_info().rss / 1024 / 1024} MB")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    logger.info("Loading tokenizer")
    tokenizer = CodeGenTokenizerFast.from_pretrained(model_path)
    logger.info(f"Memory usage after tokenizer: {psutil.Process().memory_info().rss / 1024 / 1024} MB")
    
    logger.info("Loading base model")
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quantization_config,
        device_map="cuda"  # Use GPU
    )
    logger.info(f"Memory usage after base model: {psutil.Process().memory_info().rss / 1024 / 1024} MB")
    
    logger.info("Loading PEFT model")
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    logger.info(f"Memory usage after PEFT model: {psutil.Process().memory_info().rss / 1024 / 1024} MB")
    logger.info("Model and tokenizer loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model/tokenizer: {str(e)}")
    raise

generation_args = {
    "max_new_tokens": 1000,
    "temperature": 0.001,
    "do_sample": False,
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

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")  # Move to GPU

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_args)

    input_length = len(inputs["input_ids"][0])
    generated_tokens = outputs[0][input_length:]
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return {"response": response_text}
