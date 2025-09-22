#!/usr/bin/env python3
"""Fine-tune Qwen model on Finnegans Wake."""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import os
from data_processor import FinnegansWakeProcessor

class QwenFinnegansTrainer:
    def __init__(self, model_name: str = "Qwen/Qwen2-1.5B-Instruct"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.output_dir = "./qwen-finnegans-model"
    
    def setup_model(self):
        """Initialize tokenizer and model."""
        print(f"Loading {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Setup LoRA for efficient fine-tuning
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        
        self.model = get_peft_model(self.model, lora_config)
        print("Model setup complete!")
    
    def prepare_dataset(self, text_file: str):
        """Prepare training dataset from Finnegans Wake."""
        processor = FinnegansWakeProcessor(text_file)
        training_pairs = processor.create_training_pairs()
        
        print(f"Created {len(training_pairs)} training pairs")
        
        # Tokenize the data
        def tokenize_function(examples):
            inputs = examples["input"]
            targets = examples["target"]
            
            # Format as instruction-following
            formatted_texts = []
            for inp, tgt in zip(inputs, targets):
                text = f"<|im_start|>user\n{inp}<|im_end|>\n<|im_start|>assistant\n{tgt}<|im_end|>"
                formatted_texts.append(text)
            
            tokenized = self.tokenizer(
                formatted_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized
        
        # Create dataset
        dataset_dict = {
            "input": [pair[0] for pair in training_pairs],
            "target": [pair[1] for pair in training_pairs]
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def train(self, dataset):
        """Train the model."""
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            evaluation_strategy="no",
            save_strategy="steps",
            fp16=True,
            remove_unused_columns=False,
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        print("Starting training...")
        trainer.train()
        
        # Save the final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"Model saved to {self.output_dir}")
    
    def run_training(self, text_file: str):
        """Complete training pipeline."""
        self.setup_model()
        dataset = self.prepare_dataset(text_file)
        self.train(dataset)