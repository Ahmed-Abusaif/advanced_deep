from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

class DeepSeekHandler:
    def __init__(self):
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        self.local_dir = "./deepseek_model"
        # Download model and tokenizer locally if not already present
        if not os.path.exists(self.local_dir):
            print("Downloading model to local directory...")
            AutoTokenizer.from_pretrained(self.model_name).save_pretrained(self.local_dir)
            AutoModelForCausalLM.from_pretrained(self.model_name).save_pretrained(self.local_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.local_dir,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    def generate_response(self, prompt, context=""):
        full_prompt = f"{context}\n\nQuestion: {prompt}\nAnswer:"
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=512,
            temperature=0.7,
            do_sample=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    handler = DeepSeekHandler()
    prompt = "What is the capital of France?"
    context = ""
    response = handler.generate_response(prompt, context)
    print("Response:", response)
