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
        
        # Remove response_format if it exists in kwargs for Together API
        together_kwargs = kwargs.copy()
        together_kwargs.pop('response_format', None)
        
        if self.config.model_type == "anyscale":
            response = self.client.chat.completions.create(
                model=self.config.model_path,
                messages=messages,
                stream=stream,
                temperature=0.0,
                **kwargs
            )
        elif self.config.model_type == "together":
            # Add a system message to encourage JSON output
            messages = [
                {"role": "system", "content": "You are a helpful assistant that always responds with valid JSON."},
                {"role": "user", "content": prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.config.model_path,
                messages=messages,
                stream=stream,
                temperature=0.0,
                **together_kwargs
            )
            
        if stream:
            return self._handle_stream(response)
        return response.choices[0].message.content.strip()
    
    def _handle_stream(self, response) -> str:
        """Handle streaming response"""
        full_response = ""
        
        try:
            for chunk in response:
                # Handle empty choices list at the end
                if not chunk.choices:
                    continue
                
                # Get content from delta if it exists
                if hasattr(chunk.choices[0], 'delta') and chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                # Get content from text if delta doesn't exist
                elif hasattr(chunk.choices[0], 'text') and chunk.choices[0].text:
                    content = chunk.choices[0].text
                    full_response += content
                
            return full_response
        
        except Exception as e:
            print(f"Error processing stream: {str(e)}")
            # Return what we have so far
            return full_response
