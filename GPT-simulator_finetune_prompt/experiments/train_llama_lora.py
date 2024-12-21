import argparse
import os
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
import sys
from math import ceil
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.evaluate import make_game_state, make_game_state_partial
from experiments.quest_gpt import preprocess_obj_desc

def _build_baseline_prompt(data, obj_rules, action_rules=None, score_rules=None, data_type="action"):
    """Build baseline prompt without chain-of-thought or examples"""
    prompt = ""
    
    # Add base instruction based on data type
    if data_type == "score":
        prompt = "You are a simulator of a text game. Read the task description of a text game. Given the current game state in JSON, you need to predict the current game score, whether the game is over, and whether the agent wins the game.\n\n"
    elif data_type == "action":
        prompt = "You are a simulator of a text game. Read the task description of a text game. Given the current game state in JSON, you need to decide the new game state after taking an action.\n\n"
    elif data_type == "tick":
        prompt = "You are a simulator of a text game. Read the task description. Given the current game state in JSON, you need to decide how the game state changes in the next time step (without considering the agent actions). Rules for such changes are described as the tick function of each object.\n\n"
    elif data_type == "full":
        prompt = "You are a simulator of a text game. Read the task description of a text game. Given the current game state in JSON, you need to decide the new game state after taking an action including the game score.\n\n"

    # Add task description
    prompt += f"Task Description:\n{data['current_state']['taskDesc']}\n\n"
    
    # Add rules if available
    if obj_rules:
        prompt += f"Object properties:\n{preprocess_obj_desc(obj_rules)}\n\n"
    if action_rules and data_type in ["action", "full"]:
        prompt += f"Game actions:\n{action_rules}\n\n"
    if score_rules and data_type in ["score", "full"]:
        prompt += f"Score function:\n{score_rules}\n\n"
    
    # Add current state
    prompt += "Current state:\n"
    prompt += json.dumps(data['current_state'], indent=2) + "\n\n"
    
    # Add UUID base
    prompt += f"Current game UUID base: {data['current_state'].get('max_UUID', 0)}\n"
    
    # Add action if applicable
    if data_type in ["action", "score", "full"]:
        action = data['action_state']['lastAction']
        prompt += f"Action to take: {action}\n\n"
    
    # Add response format instruction
    prompt += """Your response must be a valid JSON object that follows these rules:
1. Start with a curly brace {
2. End with a curly brace }
3. Contain only valid JSON
4. Not include any explanatory text before or after the JSON
5. Not include markdown code block markers

Generate the JSON now:"""

    return prompt

class GameStateDataset(Dataset):
    def __init__(self, data_path, tokenizer, rule_folder, games_config, data_type="action", 
                 max_length=2048, chunk_size=1000, max_examples_per_game=500):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        self.data_type = data_type
        self.max_examples_per_game = max_examples_per_game
        self.examples_per_game = {}
        
        # Load game configuration
        with open(games_config) as f:
            config = json.load(f)
            self.allowed_games = set(config["games"])
            self.example_game = config["example"]
        
        print(f"Training on games: {self.allowed_games}")
        print(f"Using example game: {self.example_game}")
        
        # Load rules
        with open(os.path.join(rule_folder, "object_rules.json")) as f:
            self.obj_rules = json.load(f)
        
        if data_type != "tick":
            with open(os.path.join(rule_folder, "action_rules.json")) as f:
                self.action_rules = json.load(f)
        
        if data_type in ["score", "full"]:
            with open(os.path.join(rule_folder, "score_rules.json")) as f:
                self.score_rules = json.load(f)

        # Process data in chunks
        chunk = []
        with open(data_path, 'r') as f:
            for line in f:
                chunk.append(json.loads(line))
                if len(chunk) >= chunk_size:
                    self._process_chunk(chunk)
                    chunk = []
            
            if chunk:
                self._process_chunk(chunk)
                
        print(f"Loaded examples per game: {self.examples_per_game}")

    def _process_chunk(self, chunk):
        for data in chunk:
            game = data["game"]
            
            # Skip if game is not in allowed games
            if game not in self.allowed_games:
                continue
            
            # Skip if we already have enough examples for this game
            if game in self.examples_per_game and self.examples_per_game[game] >= self.max_examples_per_game:
                continue
                
            # Initialize counter for new games
            if game not in self.examples_per_game:
                self.examples_per_game[game] = 0
            
            # Build baseline prompt
            input_text = _build_baseline_prompt(
                data,
                self.obj_rules.get(game),
                self.action_rules.get(game) if hasattr(self, 'action_rules') else None,
                self.score_rules.get(game) if hasattr(self, 'score_rules') else None,
                self.data_type
            )
            
            # Format output text based on data type
            if self.data_type == "score":
                output_text = json.dumps(data["next_score_state"])
            elif self.data_type == "full":
                next_state = make_game_state(data['action_state'])
                next_state["game_state"].append(data["next_score_state"])
                output_text = json.dumps(next_state)
            else:
                next_state = make_game_state(data['action_state'])
                output_text = json.dumps(next_state)
            
            try:
                # Tokenize with timeout
                model_inputs = self.tokenizer(
                    input_text,
                    output_text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                self.examples.append({
                    "input_ids": model_inputs["input_ids"][0],
                    "attention_mask": model_inputs["attention_mask"][0],
                    "labels": model_inputs["input_ids"][0].clone()
                })
                
                # Increment counter for this game
                self.examples_per_game[game] += 1
                
                # Break if we have enough examples for this game
                if self.examples_per_game[game] >= self.max_examples_per_game:
                    continue
                    
            except Exception as e:
                print(f"Error processing example: {e}")
                continue

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def load_or_download_model(model_path: str, save_model_locally: bool = False):
    """
    Load model from local path or download from HuggingFace Hub
    """
    print(f"Loading/Downloading model from {model_path}")
    
    try:
        # Try loading from local path first
        if os.path.exists(model_path):
            print("Loading model from local path...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        # If not local, download from HuggingFace
        else:
            print(f"Downloading model from HuggingFace Hub: {model_path}")
            
            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            print("Tokenizer downloaded successfully")
            
            # Download model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                cache_dir="./models"  # Cache the model locally
            )
            print("Model downloaded successfully")
            
            # Save model locally if specified
            if save_model_locally:
                local_path = os.path.join("models", os.path.basename(model_path))
                print(f"Saving model locally to {local_path}")
                os.makedirs(local_path, exist_ok=True)
                tokenizer.save_pretrained(local_path)
                model.save_pretrained(local_path)
        
        tokenizer.pad_token = tokenizer.eos_token
        print("Model and tokenizer loaded successfully")
        return tokenizer, model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        if "token" in str(e).lower():
            print("\nAuthentication Error:")
            print("1. Make sure you're logged in to Hugging Face:")
            print("   $ huggingface-cli login")
            print("2. Ensure you have access to the model:")
            print(f"   Visit: https://huggingface.co/{model_path}")
        raise e

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune LLaMA with LoRA")
    
    # Model arguments
    parser.add_argument("--base_model_path", type=str, required=True,
                      help="Path to local model or HuggingFace model ID")
    parser.add_argument("--save_model_locally", action="store_true",
                      help="Save downloaded model locally")
    
    # Data arguments
    parser.add_argument("--train_data", type=str, default="data/train.jsonl",
                      help="Path to the training data")
    parser.add_argument("--rule_folder", type=str, default="rules/human_written_rules")
    parser.add_argument("--game_file_names", default="experiments/games.json",
                      help="Path to games configuration file")
    parser.add_argument("--output_dir", type=str, default="results/lora_model")
    
    # Training arguments
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_examples_per_game", type=int, default=500)
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--target_modules", type=str, nargs="+", 
                      default=["q_proj", "k_proj", "v_proj", "o_proj"])
    
    # Task arguments
    parser.add_argument("--data_type", type=str, default="action",
                      choices=["action", "tick", "score", "full"])
    
    return parser.parse_args()

def main():
    args = parse_args()

    print("Loading tokenizer and model...")
    tokenizer, model = load_or_download_model(
        args.base_model_path,
        save_model_locally=args.save_model_locally
    )
    
    # 确保基础模型参数被冻结
    for param in model.parameters():
        param.requires_grad = False  # 冻结基础模型参数
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
        inference_mode=False,
        modules_to_save=None  # 确保不保存完整模块
    )
    
    # 应用 LoRA 并准备训练
    model = get_peft_model(model, peft_config)
    
    # 确保 LoRA 参数可训练
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
    
    model.print_trainable_parameters()
    
    # 确保模型处于训练模式
    model.train()
    
    print("Loading training data...")
    train_dataset = GameStateDataset(
        args.train_data,
        tokenizer,
        args.rule_folder,
        args.game_file_names,  # Pass game configuration file
        args.data_type,
        args.max_length,
        chunk_size=100,
        max_examples_per_game=args.max_examples_per_game
    )
    print(f"Loaded {len(train_dataset)} training examples")
    print("Examples per game:", train_dataset.examples_per_game)

    # Training arguments optimized for large models
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        evaluation_strategy="no",
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="tensorboard",
        load_best_model_at_end=False,
        optim="paged_adamw_32bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}  # 添加这个选项
    )

    # Initialize trainer with data collator
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=lambda data: {
            'input_ids': torch.stack([f["input_ids"] for f in data]),
            'attention_mask': torch.stack([f["attention_mask"] for f in data]),
            'labels': torch.stack([f["labels"] for f in data]),
        }
    )

    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model(args.output_dir)
    
    # Save adapter config
    model.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
