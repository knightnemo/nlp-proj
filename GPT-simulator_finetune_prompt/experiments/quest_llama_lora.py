import argparse
import os
import json
import random
import sys
from math import ceil
import requests
from huggingface_hub import InferenceClient
from typing import Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.evaluate import evaluate, make_game_state, make_game_state_partial, evaluate_score
from experiments.quest_gpt import preprocess_obj_desc, recover_game_state_from_partial

def parse_args():
    parser = argparse.ArgumentParser()
    # Core arguments for local model
    parser.add_argument("--base_model_path", type=str, required=True,
                      help="Path to base model")
    parser.add_argument("--lora_weights_path", type=str, default=None,
                      help="Path to LoRA weights")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Data arguments
    parser.add_argument("--state_data_folder", type=str, default="data")
    parser.add_argument("--test_data", type=str, default="test.jsonl")
    parser.add_argument("--rule_folder", type=str, default="rules/human_written_rules")
    parser.add_argument("--game_file_names", default="experiments/games.json",
                      help="Path to games configuration file")
    parser.add_argument("--output_folder", type=str, default="results")
    parser.add_argument("--output_prefix", type=str, default="")
    parser.add_argument("--output_suffix", type=str, default="")
    
    # Task arguments
    parser.add_argument("--data_type", type=str, default="action",
                      choices=["action", "tick", "score", "full"])
    parser.add_argument("--no_rule", action="store_true")
    
    # Add shard index argument
    parser.add_argument("--shard_idx", type=str, default="0",
                      help="Shard index for output files")
    
    # Add max_new_tokens argument
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                      help="Maximum number of new tokens to generate")
    
    return parser.parse_args()

class HFInferenceAPI:
    def __init__(self, model_id: str, token: str):
        self.client = InferenceClient(model_id, token=token)
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.text_generation(
                prompt,
                max_new_tokens=1024,
                temperature=0.1,
                top_p=0.95,
                repetition_penalty=1.15,
                **kwargs
            )
            return response
        except Exception as e:
            print(f"Error in API call: {str(e)}")
            return None

def _build_baseline_prompt(data: Dict[str, Any], obj_rules: Optional[Dict] = None, 
                         action_rules: Optional[str] = None, 
                         score_rules: Optional[str] = None,
                         data_type: str = "action") -> str:
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
        prompt = "You are a simulator of a text game. Read the task description of a text game. Given the current game state in JSON, you need to decide the new game state after taking an action including the game score. Your response must be a JSON object with two required keys:\n\n"
        prompt += "1. 'game_state': A list containing the new game state objects\n"
        prompt += "2. 'score': An object with three required keys:\n"
        prompt += "   - 'score': Current game score (number)\n"
        prompt += "   - 'gameOver': Whether the game is over (boolean)\n"
        prompt += "   - 'gameWon': Whether the player has won (boolean)\n\n"

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
"""
    
    if data_type == "full":
        prompt += """6. Include both 'game_state' and 'score' keys at the root level
7. The 'score' object must contain 'score', 'gameOver', and 'gameWon' keys

Example format for full type:
{
  "game_state": [ ... game state objects ... ],
  "score": {
    "score": 0,
    "gameOver": false,
    "gameWon": false
  }
}
"""

    prompt += "\nGenerate the JSON now:"
    return prompt

def load_model(base_model_path: str, lora_weights_path: Optional[str] = None, device: str = "cuda"):
    """Load base model with optional LoRA weights"""
    print(f"Loading base model from {base_model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load LoRA weights if provided
    if lora_weights_path:
        print(f"Loading LoRA weights from {lora_weights_path}")
        model = PeftModel.from_pretrained(
            model,
            lora_weights_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    return model, tokenizer

def postProcess(raw_response):
    """更强大的 JSON 响应处理函数"""
    # 清理响应文本
    cleaned = raw_response.strip()
    
    # 1. 如果是 markdown 代码块格式
    if cleaned.startswith("```") and cleaned.endswith("```"):
        # 移除 markdown 标记
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:-3].strip()
        else:
            cleaned = cleaned[3:-3].strip()
            
    # 2. 确保只保留有效的 JSON 部分
    # 找到第一个 { 和最后一个 }
    start_idx = cleaned.find('{')
    end_idx = cleaned.rfind('}')
    
    if (start_idx != -1) and (end_idx != -1):
        cleaned = cleaned[start_idx:end_idx + 1]
    
    # 3. 移除所有换行和多余空格，确保 JSON 是连续的
    cleaned = ' '.join(cleaned.split())
    
    return cleaned

def main():
    args = parse_args()
    random.seed(0)

    # Load model (with or without LoRA)
    model, tokenizer = load_model(
        args.base_model_path,
        args.lora_weights_path,
        args.device
    )
    
    model_type = "LoRA" if args.lora_weights_path else "Base"
    print(f"Testing with {model_type} model")
    
    # Load game configuration
    with open(args.game_file_names) as f:
        config = json.load(f)
        allowed_games = set(config["games"])
        example_game = config["example"]
    
    print(f"Testing on games: {allowed_games}")
    print(f"Using example game: {example_game}")

    # Load rules
    with open(os.path.join(args.rule_folder, "object_rules.json")) as f:
        obj_rules = json.load(f)

    if args.data_type != "tick":
        with open(os.path.join(args.rule_folder, "action_rules.json")) as f:
            action_rules = json.load(f)

    if args.data_type in ["score", "full"]:
        with open(os.path.join(args.rule_folder, "score_rules.json")) as f:
            score_rules = json.load(f)

    # Process test data
    result_outputs = []
    statistics = {}
    
    # Load all test data
    with open(os.path.join(args.state_data_folder, args.test_data)) as f:
        test_data = [json.loads(line) for line in f.readlines()]
    
    # 按游戏分组并采样
    test_examples = []
    for game in allowed_games:
        game_examples = [data for data in test_data if data["game"] == game]
        if len(game_examples) > 50:
            game_examples = random.sample(game_examples, 50)
        test_examples.extend(game_examples)
    
    print(f"Selected total {len(test_examples)} test examples:")
    game_counts = {}
    for example in test_examples:
        game = example["game"]
        game_counts[game] = game_counts.get(game, 0) + 1
    for game, count in game_counts.items():
        print(f"{game}: {count} examples")

    # Initialize accuracy tracking
    accuracy_stats = {
        "overall": {"correct": 0, "total": 0},
        "per_game": {}
    }
    
    for data in test_examples:
        game = data["game"]
        # Initialize game stats if not exists
        if game not in accuracy_stats["per_game"]:
            accuracy_stats["per_game"][game] = {"correct": 0, "total": 0}
            
        print(f"\nProcessing example from game: {game}")
            
        # Build baseline prompt
        prompt = _build_baseline_prompt(
            data,
            obj_rules.get(game),
            action_rules.get(game) if args.data_type != "tick" else None,
            score_rules.get(game) if args.data_type in ["score", "full"] else None,
            args.data_type
        )

        # Generate prediction with improved parameters
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        
        model_outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,  # 使用命令行参数
            temperature=0.1,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.15,
        )
        
        response_text = tokenizer.decode(model_outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # 打印原始输出
        print("\nRaw model output:")
        print("-" * 50)
        print(response_text)
        print("-" * 50)
        
        try:
            # 使用 postProcess 函数处理响应文本
            response_text = postProcess(response_text)
            
            # 检查预期的数据格式
            try:
                prediction = json.loads(response_text)
                
                # 简化格式检查输出
                if args.data_type == "score":
                    required_keys = {"score", "gameOver", "win"}
                    missing_keys = required_keys - set(prediction.keys())
                    if missing_keys:
                        raise ValueError(f"Missing required keys in score prediction: {missing_keys}")
                
                elif args.data_type in ["action", "tick", "full"]:
                    if "game_state" not in prediction:
                        raise ValueError("Missing 'game_state' key in prediction")
                    if not isinstance(prediction["game_state"], list):
                        raise ValueError("'game_state' must be a list")
                
            except json.JSONDecodeError as e:
                print(f"JSON Parsing Error: {str(e)}")
                raise
            except ValueError as e:
                print(f"Format Error: {str(e)}")
                raise
            
            # 评估预测结果
            if args.data_type == "score":
                num_errors, eval_out_str = evaluate_score(prediction, data["next_score_state"])
            else:
                if "game_state" in prediction:
                    formatted_prediction = {"game_state": prediction["game_state"]}
                    if args.data_type == "full":
                        # 对于 full 类型，需要分别评估游戏状态和分数
                        state_errors, state_eval = evaluate(
                            formatted_prediction,
                            data["action_state"],
                            data['action_state']['lastAction']
                        )
                        
                        try:
                            # 从根级别获取score对象并直接评估
                            if "score" not in prediction:
                                raise ValueError("Missing 'score' key in root level")
                            
                            score_errors, score_eval = evaluate_score(prediction["score"], data["next_score_state"])
                            
                            # 合并结果
                            num_errors = state_errors + score_errors  # 现在同时考虑状态错误和分数错误
                            num_score_errors = score_errors
                            eval_out_str = f"Game state evaluation:\n{state_eval}\n\nScore evaluation:\n{score_eval}"
                            
                        except Exception as e:
                            num_errors = state_errors
                            num_score_errors = 1  # 分数评估出错记为1个错误
                            eval_out_str = f"Game state evaluation:\n{state_eval}\n\nScore evaluation error: {str(e)}"
                    else:
                        num_errors, eval_out_str = evaluate(
                            formatted_prediction,
                            data["action_state"],
                            data['action_state']['lastAction']
                        )

            # Print minimal evaluation result
            print(f"Number of errors: {num_errors}")
            if num_errors > 0:
                print(eval_out_str)

            # Update accuracy statistics after each prediction
            accuracy_stats["overall"]["total"] += 1
            accuracy_stats["per_game"][game]["total"] += 1
            if num_errors == 0:
                accuracy_stats["overall"]["correct"] += 1
                accuracy_stats["per_game"][game]["correct"] += 1
            
            # Print current accuracy
            overall_acc = (accuracy_stats["overall"]["correct"] / accuracy_stats["overall"]["total"]) * 100
            game_acc = (accuracy_stats["per_game"][game]["correct"] / accuracy_stats["per_game"][game]["total"]) * 100
            
            print(f"\nCurrent Accuracies:")
            print(f"Overall: {overall_acc:.1f}% ({accuracy_stats['overall']['correct']}/{accuracy_stats['overall']['total']})")
            print(f"{game}: {game_acc:.1f}% ({accuracy_stats['per_game'][game]['correct']}/{accuracy_stats['per_game'][game]['total']})")

            # Record results
            output_record = {
                "game": game,
                "state_id": data.get("state_id", "unknown"),
                "curr_state": data['current_state'],
                "prompt": prompt,
                "predicted_state_raw": response_text,
                "predicted_state": prediction,
                "num_errors": num_errors,
                "eval_out_str": eval_out_str,
                "current_accuracy": {
                    "overall": overall_acc,
                    "game": game_acc
                }
            }
            
            if args.data_type == "full":
                if "num_score_errors" in locals():
                    output_record["num_score_errors"] = num_score_errors
                else:
                    output_record["num_score_errors"] = 0
            
            result_outputs.append(output_record)
            
            # Update statistics
            if game not in statistics:
                statistics[game] = {"total_errors": 0, "total_states": 0}
            statistics[game]["total_states"] += 1
            if num_errors > 0:
                statistics[game]["total_errors"] += 1

            # Print progress
            print(f"Errors: {num_errors}")
            print(eval_out_str)

        except Exception as e:
            print(f"Error processing response: {str(e)}")
            # Update accuracy statistics for errors
            accuracy_stats["overall"]["total"] += 1
            accuracy_stats["per_game"][game]["total"] += 1
            
            # Update statistics for exceptions
            if game not in statistics:
                statistics[game] = {"total_errors": 0, "total_states": 0}
            statistics[game]["total_states"] += 1
            statistics[game]["total_errors"] += 1
            
            # Record error result
            output_record = {
                "game": game,
                "state_id": data.get("state_id", "unknown"),
                "curr_state": data['current_state'],
                "prompt": prompt,
                "error": str(e),
                "num_errors": 1,  # Count exception as an error
                "eval_out_str": f"Exception occurred: {str(e)}",
                "current_accuracy": {
                    "overall": (accuracy_stats["overall"]["correct"] / accuracy_stats["overall"]["total"]) * 100,
                    "game": (accuracy_stats["per_game"][game]["correct"] / accuracy_stats["per_game"][game]["total"]) * 100
                }
            }
            result_outputs.append(output_record)
            continue

    # Update final statistics format
    final_stats = {
        "raw_statistics": statistics,
        "accuracy_statistics": {
            "overall": {
                "total_examples": accuracy_stats["overall"]["total"],
                "correct_examples": accuracy_stats["overall"]["correct"],
                "accuracy": (accuracy_stats["overall"]["correct"] / accuracy_stats["overall"]["total"] * 100) 
                           if accuracy_stats["overall"]["total"] > 0 else 0
            },
            "per_game": {
                game: {
                    "total_examples": stats["total"],
                    "correct_examples": stats["correct"],
                    "accuracy": (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
                }
                for game, stats in accuracy_stats["per_game"].items()
            }
        },
        "test_config": {
            "model_type": model_type,
            "data_type": args.data_type,
            "allowed_games": list(allowed_games),
            "example_game": example_game
        }
    }

    # Save results
    if result_outputs:
        os.makedirs(args.output_folder, exist_ok=True)
        
        # Save complete statistics
        with open(f"{args.output_folder}/results_{args.output_prefix}_{args.shard_idx}{args.output_suffix}.json", "w") as f:
            json.dump(final_stats, f, indent=4)
            
        # Save detailed results including per-example accuracies
        output_file = f"{args.output_folder}/{args.output_prefix}_{args.shard_idx}{args.output_suffix}.jsonl"
        with open(output_file, 'w') as f:
            for line in result_outputs:
                f.write(json.dumps(line) + '\n')

if __name__ == "__main__":
    main()
