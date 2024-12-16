import argparse
import tiktoken
import os
import json
import random
import sys
from typing import Optional, Union
from dataclasses import dataclass
from tqdm import tqdm
# Local Llama imports
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# API imports
from openai import OpenAI
from together import Together  # 如果使用 Together API

@dataclass
class LlamaConfig:
    """Configuration for Llama model loading"""
    model_path: str
    model_type: str = "local"  # "local", "anyscale", "together"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class LlamaInterface:
    def __init__(self, config: LlamaConfig):
        self.config = config
        self.client = None
        self.model = None
        self.tokenizer = None
        
        if config.model_type == "local":
            self._init_local_model()
        else:
            self._init_api_client()
    
    def _init_local_model(self):
        """Initialize local Llama model"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.float16,
            device_map=self.config.device
        )
        
    def _init_api_client(self):
        """Initialize API client"""
        if self.config.model_type == "anyscale":
            self.client = OpenAI(
                api_key=self.config.api_key,
                base_url="https://api.endpoints.anyscale.com/v1"
            )
        elif self.config.model_type == "together":
            self.client = Together(api_key=self.config.api_key)
    
    def generate(self, prompt: str, stream: bool = False, **kwargs) -> str:
        """Generate response from either local model or API"""
        if self.config.model_type == "local":
            return self._generate_local(prompt, **kwargs)
        else:
            return self._generate_api(prompt, stream, **kwargs)
    
    def _generate_local(self, prompt: str, **kwargs) -> str:
        """Generate response using local model"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.0),
            top_p=kwargs.get("top_p", 1.0),
            do_sample=False
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _generate_api(self, prompt: str, stream: bool = False, **kwargs) -> str:
        """Generate response using API"""
        messages = [{"role": "user", "content": prompt}]
        
        if self.config.model_type == "anyscale":
            response = self.client.chat.completions.create(
                model=self.config.model_path,
                messages=messages,
                stream=stream,
                temperature=0.0,
                **kwargs
            )
        elif self.config.model_type == "together":
            client = Together()
            response = client.chat.completions.create(
                model=self.config.model_path,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=None,
                temperature=0.7,
                top_p=0.7,
                top_k=50,
                repetition_penalty=1,
                stop=["<|eot_id|>","<|eom_id|>"],
                stream=True
            )
            
        if stream:
            return self._handle_stream(response)
        return response.choices[0].message.content.strip()
    
    def _handle_stream(self, response) -> str:
        """Handle streaming response"""
        full_response = ""
        pbar = tqdm(response, desc="Generating", unit=" tokens")
        for chunk in pbar:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                pbar.set_postfix_str(f"...{content[-50:]}")
        return full_response
