import argparse
import os
import json
from tqdm import tqdm
import sys
sys.path.append("/home/featurize/GPT-simulator")
from llama_utils.llama_interface import LlamaConfig, LlamaInterface
from experiments.quest_gpt import (
    preprocess_obj_desc,
    recover_game_state_from_partial
)
from experiments.prompt_strategies import SimpleCoTStrategy, ZeroShotCoTStrategy, FewShotStrategy, CoTSCStrategy
from utils.experiment_utils import extract_confidence, select_majority_prediction
import random
import torch
from math import ceil
from scripts.evaluate import evaluate, evaluate_score
from together import Together

def parse_args():
    parser = argparse.ArgumentParser()
    # Core strategy arguments
    parser.add_argument("--mode", type=str, 
                        choices=["simple_cot", "zero_shot_cot", "cot_sc", "few_shot"],
                        default="simple_cot", help="Prompting method")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples for CoT-SC")
    
    # Data source arguments
    parser.add_argument("--state_data_folder", type=str, default="data")
    parser.add_argument("--test_data", type=str, default="test.jsonl")
    parser.add_argument("--example_data", type=str, default="data/examples.json")
    parser.add_argument("--rule_folder", type=str, default="rules/human_written_rules")
    parser.add_argument("--data_distribution_file", type=str, default="data/dynamic_static_states_per_action.json")
    parser.add_argument("--state_change_file", type=str, default='data/dynamic_states.json')
    parser.add_argument("--game_file_names", default="experiments/games.json")
    
    # Output arguments
    parser.add_argument("--output_folder", type=str, default="results")
    parser.add_argument("--output_prefix", type=str, default="")
    parser.add_argument("--output_suffix", type=str, default="")
    
    # Model configuration
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="local",
                        choices=["local", "anyscale", "together"],
                        help="Type of model loading")
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    # Processing options
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--total_shards", type=int, default=1)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--data_type", type=str, default="action", 
                        choices=["action", "tick", "score", "full"])
    parser.add_argument("--no_rule", action="store_true")
    parser.add_argument("--notlive", action='store_true', default=False)
    parser.add_argument("--random_seed", type=int, default=0)
    
    return parser.parse_args()

def load_game_data(args):
    """Load all necessary game data files"""
    data = {}
    
    # Load games list
    with open(args.game_file_names) as f:
        games_all = json.load(f)
        data['games'] = games_all['games']
        data['example_game'] = games_all['example']
    
    # Load rules
    with open(os.path.join(args.rule_folder, "object_rules.json")) as f:
        data['obj_rules'] = json.load(f)
    if args.data_type != "tick":
        with open(os.path.join(args.rule_folder, "action_rules.json")) as f:
            data['action_rules'] = json.load(f)
    if args.data_type in ["score", "full"]:
        with open(os.path.join(args.rule_folder, "score_rules.json")) as f:
            data['score_rules'] = json.load(f)
            
    # Load examples and state information
    with open(args.example_data) as f:
        data['examples'] = json.load(f)
    with open(args.data_distribution_file) as f:
        data['data_distribution'] = json.load(f)
    with open(args.state_change_file) as f:
        data['state_change_info'] = json.load(f)
        
    return data

def run_experiment(llm, test_data, game_data, args):
    """Run experiment with specified prompting strategy"""
    results = []
    
    for item in tqdm(test_data, desc="Processing items"):
        try:
            print(f"\nProcessing: {item.get('game', 'unknown')} - state_id: {item.get('state_id', '')}")
            
            # Validate required fields
            if 'game' not in item or 'current_state' not in item:
                raise ValueError("Missing required fields in test data item")
                
            game = item["game"]
            if game not in game_data['obj_rules']:
                raise ValueError(f"Game {game} not found in object rules")
                
            data_obj_desc = preprocess_obj_desc(game_data['obj_rules'][game])
            
            # Select strategy based on mode
            if args.mode == "simple_cot":
                strategy = SimpleCoTStrategy(
                    llm=llm,
                    object_desc=data_obj_desc,
                    game_data=game_data,
                    data_type=args.data_type,
                    partial=args.partial
                )
            elif args.mode == "zero_shot_cot":
                strategy = ZeroShotCoTStrategy(
                    llm=llm,
                    object_desc=data_obj_desc,
                    game_data=game_data,
                    data_type=args.data_type,
                    partial=args.partial
                )
            elif args.mode == "cot_sc":
                strategy = CoTSCStrategy(
                    llm=llm,
                    object_desc=data_obj_desc,
                    game_data=game_data,
                    num_samples=args.num_samples,
                    data_type=args.data_type,
                    partial=args.partial
                )
            else:  # few_shot
                strategy = FewShotStrategy(
                    llm=llm,
                    object_desc=data_obj_desc,
                    game_data=game_data,
                    data_type=args.data_type,
                    partial=args.partial
                )
            
            predicted_state = strategy.get_prediction(item)
            
            print("\nModel Response:")
            print("-" * 40)
            print(strategy.last_response)
            print("-" * 40)
            
            print("\nParsed State:")
            if predicted_state is not None:
                print(json.dumps(predicted_state, indent=2))
            else:
                print("Failed to parse response as JSON")
            
            # Format result for evaluation
            result = {
                'game': game,
                'state_id': item.get('state_id', ''),
                'success': predicted_state is not None,
                'predicted_state': predicted_state,
                'curr_state': item['current_state'],
                'action': item.get('action', ''),
                'prompt': strategy.last_prompt,
                'response': strategy.last_response
            }
            
            # Only add evaluation metrics if we have a gold state to compare against
            if 'next_state' in item:
                if args.data_type == "score":
                    num_errors, error_msg = evaluate_score(
                        predicted_state, 
                        item['next_state']
                    )
                else:
                    num_errors, error_msg = evaluate(
                        predicted_state,
                        item['next_state'],
                        item.get('action', ''),
                        evaluate_score=(args.data_type == "full")
                    )
                result['num_errors'] = num_errors
                result['error_msg'] = error_msg
                result['gold_state'] = item['next_state']
                
                print("\nEvaluation:")
                print(f"Number of errors: {num_errors}")
                if num_errors > 0:
                    print(f"Error details: {error_msg}")
            
            print("-" * 80)
            results.append(result)
            
        except Exception as e:
            print(f"\nError processing item: {str(e)}")
            print("\nLast model response:")
            print("-" * 40)
            if hasattr(strategy, 'last_response'):
                print(strategy.last_response)
            print("-" * 40)
            results.append({
                'game': item.get('game', 'unknown'),
                'state_id': item.get('state_id', ''),
                'success': False,
                'error': str(e),
                'response': getattr(strategy, 'last_response', None),
                'item': item
            })
    
    return results

def main():
    args = parse_args()
    random.seed(args.random_seed)
    
    # Initialize Together API if using together model type
    if args.model_type == "together":
        Together.api_key = args.api_key
    
    # Initialize model
    llm = LlamaInterface(LlamaConfig(
        model_path=args.model_path,
        model_type=args.model_type,
        api_key=args.api_key,
        device=args.device
    ))
    
    # Load all necessary data
    game_data = load_game_data(args)
    
    # Load and filter test data
    with open(os.path.join(args.state_data_folder, args.test_data)) as f:
        test_data = [json.loads(line) for line in f]
        if args.total_shards > 1:
            shard_size = ceil(len(test_data) / args.total_shards)
            test_data = test_data[shard_size * args.shard_idx:shard_size * (args.shard_idx + 1)]
    
    if not args.notlive:
        results = run_experiment(llm, test_data, game_data, args)
        
        # Save results
        os.makedirs(args.output_folder, exist_ok=True)
        output_file = f"{args.output_folder}/{args.output_prefix}_{args.mode}_{args.shard_idx}{args.output_suffix}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
    else:
        print("'live' parameter is not set, will not send requests to API.")

if __name__ == "__main__":
    main()