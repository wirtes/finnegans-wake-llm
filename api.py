#!/usr/bin/env python3
"""FastAPI server for Finnegans Wake style translation."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import uvicorn

app = FastAPI(title="Finnegans Wake Style Translator", version="1.0.0")

class TranslationRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    original: str
    translated: str

class FinnegansTranslator:
    def __init__(self, model_path: str = "./qwen-finnegans-model"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the fine-tuned model."""
        try:
            print("Loading fine-tuned model...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Load base model and apply LoRA weights
            base_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2-1.5B-Instruct",
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.model.eval()
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using base model instead...")
            self.load_base_model()
    
    def load_base_model(self):
        """Fallback to base model if fine-tuned model fails."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2-1.5B-Instruct",
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2-1.5B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    
    def translate(self, text: str) -> str:
        """Translate text to Finnegans Wake style."""
        prompt = f"<|im_start|>user\nTranslate this to Finnegans Wake style: {text}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        assistant_start = response.find("<|im_start|>assistant\n")
        if assistant_start != -1:
            response = response[assistant_start + len("<|im_start|>assistant\n"):].strip()
        
        return response

# Global translator instance
translator = None

@app.on_event("startup")
async def startup_event():
    global translator
    translator = FinnegansTranslator()

@app.get("/")
async def root():
    return {"message": "Finnegans Wake Style Translator API"}

@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """Translate text to Finnegans Wake style."""
    if not translator:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        translated = translator.translate(request.text)
        return TranslationResponse(
            original=request.text,
            translated=translated
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": translator is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)