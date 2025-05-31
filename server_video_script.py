from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

# Load model and tokenizer once on startup
model_path = "./phi4_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
model.eval()

generation_args = {
    "max_new_tokens": 1000,
    "temperature": 0.001,
    "do_sample": True,
    "top_p": 0.9,
    "repetition_penalty": 1.1
}

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    messages = data.get("messages", [])
    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    # Format messages using your tokenizer method
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_args)

    input_length = len(inputs["input_ids"][0])
    generated_tokens = outputs[0][input_length:]
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return jsonify({"response": response_text})

if __name__ == "__main__":
    # Run on 0.0.0.0 so Render or any cloud can access it
    app.run(host="0.0.0.0", port=8080)
