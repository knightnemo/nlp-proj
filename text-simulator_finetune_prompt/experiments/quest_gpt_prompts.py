import argparse
import json
import os
from tqdm import tqdm
from typing import Dict, Any, Optional
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from openai import OpenAI
from experiments.prompt_strategies_gpt import (
    GPTPromptStrategy,
    GPTSimpleCoTStrategy,
    GPTZeroShotCoTStrategy,
    GPTCoTSCStrategy,
    GPTFewShotCoTStrategy,
    GPTBaselineStrategy
)
from experiments.quest_gpt import preprocess_obj_desc
from scripts.evaluate import evaluate, evaluate_score, make_game_state
import time

def load_game_data(args) -> Dict[str, Any]:
    """Load all necessary game data files"""
    data = {}
    
    # Load games list
    with open(args.game_file_names, 'r') as f:
        games_all = json.load(f)
        data['games'] = games_all['games']
        data['example_game'] = games_all.get('example')
    
    # Load test data
    test_data = {}
    with open(os.path.join(args.state_data_folder, args.test_data), 'r') as f:
        all_items = [json.loads(line) for line in f]
        
        # Group items by game
        game_items = {}
        for item in all_items:
            game = item['game']
            if game not in game_items:
                game_items[game] = []
            game_items[game].append(item)
        
        # Sample 50 items per game
        for game, items in game_items.items():
            if len(items) > 50:
                # Use random.sample to get 50 random items
                import random
                test_data[game] = random.sample(items, 50)
            else:
                test_data[game] = items
                print(f"Warning: Game {game} has fewer than 50 examples ({len(items)} available)")
    
    data['test_data'] = test_data
            
    # Load rules
    with open(os.path.join(args.rule_folder, "object_rules.json")) as f:
        data['obj_rules'] = json.load(f)
    if args.data_type != "tick":
        with open(os.path.join(args.rule_folder, "action_rules.json")) as f:
            data['action_rules'] = json.load(f)
    if args.data_type in ["score", "full"]:
        with open(os.path.join(args.rule_folder, "score_rules.json")) as f:
            data['score_rules'] = json.load(f)
            
    return data

def get_strategy(mode: str, llm, object_desc: str, game_data: Dict, args) -> GPTPromptStrategy:
    """Get the appropriate prompt strategy based on mode"""
    # Prepare game data dictionary with required fields
    strategy_data = {
        "object_desc": object_desc,
        "model": args.model,
    }
    
    # Add action rules if needed
    if args.data_type != "tick":
        strategy_data["action_desc"] = game_data.get('action_rules', {})
        
    # Add score rules if needed
    if args.data_type in ["score", "full"]:
        strategy_data["score_desc"] = game_data.get('score_rules', {})

    # Pass no_rule argument to strategies
    if mode == "baseline":
        return GPTBaselineStrategy(llm, object_desc, strategy_data, args.data_type, args.partial, args.no_rule)
    elif mode == "simple":
        return GPTSimpleCoTStrategy(llm, object_desc, strategy_data, args.data_type, args.partial, args.no_rule)
    elif mode == "zero_shot_cot":
        return GPTZeroShotCoTStrategy(llm, object_desc, strategy_data, args.data_type, args.partial, args.no_rule)
    elif mode == "cot_sc":
        return GPTCoTSCStrategy(llm, object_desc, strategy_data, args.data_type, args.partial, args.no_rule)
    elif mode == "few_shot":
        return GPTFewShotCoTStrategy(llm, object_desc, strategy_data, args.data_type, args.partial, args.no_rule)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def stream_llm_gpt(prompt, model=None, response_format=None, client=None):
    """Helper function to call GPT API consistently"""
    if client is None:
        raise ValueError("OpenAI client must be provided")
        
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise game state simulator that only outputs valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format=response_format,
            temperature=0,
            max_tokens=1024
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error in OpenAI API call: {str(e)}")
        return None

def run_experiment(llm, test_data: Dict, game_data: Dict, args) -> list:
    """Run experiment with specified prompting strategy"""
    results = []
    statistics = {game: {
        "total_cases": 0,
        "total_errors": 0,
        "correct_cases": 0,
        "accuracy_rate": 0.0,
        "error_rate": 0.0
    } for game in game_data['games']}
    
    try:
        for game in game_data['games']:
            print(f"\nProcessing game: {game}")
            
            # Get game-specific data
            game_items = test_data.get(game, [])
            if not game_items:
                print(f"Warning: No test data found for game {game}")
                continue
                
            object_desc = preprocess_obj_desc(game_data['obj_rules'].get(game, {}))
            
            # Create game-specific data dictionary
            game_specific_data = {
                'obj_rules': game_data['obj_rules'].get(game, {}),
            }
            if args.data_type != "tick":
                game_specific_data['action_rules'] = game_data.get('action_rules', {}).get(game, {})
            if args.data_type in ["score", "full"]:
                game_specific_data['score_rules'] = game_data.get('score_rules', {}).get(game, {})
            
            # Initialize strategy
            strategy = get_strategy(args.mode, llm, object_desc, game_specific_data, args)
            
            # Process each test case
            for item in tqdm(game_items, desc=f"Processing {game}"):
                try:
                    state_id = item.get("state_id", "unknown")
                    
                    # Determine current and target state keys based on data_type
                    if args.data_type == "action":
                        curr_state_key = "current_state"
                        next_state_key = "action_state"
                    elif args.data_type == "tick":
                        curr_state_key = "action_state"
                        next_state_key = "tick_state"
                    elif args.data_type == "score":
                        curr_state_key = "current_state"
                        next_state_key = "tick_state"
                    elif args.data_type == "full":
                        curr_state_key = "current_state"
                        next_state_key = "tick_state"
                    
                    # Validate required fields
                    if curr_state_key not in item:
                        print(f"Warning: Missing {curr_state_key} in test item {state_id}")
                        print(f"Available fields: {list(item.keys())}")
                        continue
                        
                    if next_state_key not in item:
                        print(f"Warning: Missing {next_state_key} in test item {state_id}")
                        print(f"Available fields: {list(item.keys())}")
                        continue
                    
                    # Handle score states for score and full data types
                    if args.data_type in ["score", "full"]:
                        if not all(k in item for k in ["current_score_state", "next_score_state"]):
                            print(f"Warning: Missing score states in test item {state_id}")
                            print(f"Available fields: {list(item.keys())}")
                            continue
                    
                    # Create game states using make_game_state
                    curr_state = make_game_state(item[curr_state_key])
                    if args.data_type == "full":
                        curr_state["game_state"].append(item["current_score_state"])
                    
                    # Set target state based on data type
                    if args.data_type == "score":
                        item["target_state"] = item["next_score_state"]
                    elif args.data_type == "full":
                        target_state = make_game_state(item[next_state_key])
                        target_state["game_state"].append(item["next_score_state"])
                        item["target_state"] = target_state
                    else:
                        item["target_state"] = make_game_state(item[next_state_key])
                    
                    # Update current state in item
                    item["current_state"] = curr_state
                    
                    # Handle action field for non-tick data types
                    if args.data_type in ["action", "score", "full"]:
                        action = None
                        # Try multiple locations for action
                        for action_source in [
                            lambda x: x.get('action'),
                            lambda x: x.get(next_state_key, {}).get('lastAction'),
                            lambda x: x.get(curr_state_key, {}).get('lastAction'),
                            lambda x: x.get('tick_state', {}).get('lastAction')
                        ]:
                            action = action_source(item)
                            if action:
                                break
                                
                        if action is None:
                            print(f"Warning: Could not find action for {game} - {state_id}")
                            action = ""  # Use empty string as fallback
                        item['action'] = action  # Store found action
                    
                    # Get prediction
                    prediction = strategy.get_prediction(item)
                    
                    if prediction is None:
                        print(f"Failed to get prediction for {game} - {state_id}")
                        print(f"Last prompt: {strategy.last_prompt}")
                        print(f"Last response: {strategy.last_response}")
                        continue
                    
                    # Evaluate prediction
                    if args.data_type == "score":
                        num_errors = evaluate_score(prediction, item["target_state"])
                        eval_out_str = f"Score errors: {num_errors}"
                    else:
                        # Get last_action from the prediction or item
                        last_action = prediction.get('lastAction', '') or item.get('action_state', {}).get('lastAction', '')
                        
                        # Call evaluate with all required arguments
                        eval_result = evaluate(
                            prediction, 
                            item["target_state"],
                            last_action
                        )
                        # Extract just the error count from the tuple
                        num_errors = eval_result[0] if isinstance(eval_result, tuple) else eval_result
                        eval_out_str = f"Object property errors: {num_errors}"
                        if isinstance(eval_result, tuple):
                            eval_out_str += f"\nDetails: {eval_result[1]}"
                    
                    # Record results
                    result = {
                        "game": game,
                        "state_id": state_id,
                        "curr_state": item["current_state"],
                        "action": item.get("action"),
                        "prompt": strategy.last_prompt,
                        "response": strategy.last_response,
                        "gold_state": item["target_state"],
                        "predicted_state": prediction,
                        "num_errors": num_errors,  # Now guaranteed to be an integer
                        "eval_out_str": eval_out_str,
                        "model": args.model
                    }
                    results.append(result)
                    
                    # Update statistics
                    statistics[game]["total_cases"] += 1
                    if num_errors > 0:
                        statistics[game]["total_errors"] += 1
                    else:
                        statistics[game]["correct_cases"] += 1
                        
                    # Calculate rates after each case
                    total = statistics[game]["total_cases"]
                    correct = statistics[game]["correct_cases"]
                    statistics[game]["accuracy_rate"] = (correct / total) * 100 if total > 0 else 0
                    statistics[game]["error_rate"] = ((total - correct) / total) * 100 if total > 0 else 0
                    
                    # Print current statistics for the game
                    print(f"\nCurrent statistics for {game}:")
                    print(f"Total cases: {total}")
                    print(f"Correct cases: {correct}")
                    print(f"Accuracy rate: {statistics[game]['accuracy_rate']:.2f}%")
                    print(f"Error rate: {statistics[game]['error_rate']:.2f}%")
                    
                except Exception as e:
                    print(f"Error processing {game} - {state_id}: {str(e)}")
                    print(f"Error type: {type(e).__name__}")
                    print(f"Error details:")
                    print(f"- Location: {e.__traceback__.tb_frame.f_code.co_name}")
                    print(f"- Line number: {e.__traceback__.tb_lineno}")
                    
                    results.append({
                        "game": game,
                        "state_id": state_id,
                        "error": {
                            "message": str(e),
                            "type": type(e).__name__,
                            "location": e.__traceback__.tb_frame.f_code.co_name,
                            "line": e.__traceback__.tb_lineno
                        },
                        "success": False
                    })
                    statistics[game]["total_cases"] += 1 
                    statistics[game]["total_errors"] += 1
    
    except KeyboardInterrupt:
        print("\n\nExperiment stopped by user.")
        print("Saving partial results...")
    
    # Calculate overall statistics before returning
    total_cases = sum(s["total_cases"] for s in statistics.values())
    total_correct = sum(s["correct_cases"] for s in statistics.values())
    
    statistics["overall"] = {
        "total_cases": total_cases,
        "total_errors": sum(s["total_errors"] for s in statistics.values()),
        "correct_cases": total_correct,
        "accuracy_rate": (total_correct / total_cases) * 100 if total_cases > 0 else 0,
        "error_rate": ((total_cases - total_correct) / total_cases) * 100 if total_cases > 0 else 0
    }
    
    # Add timestamp and experiment info
    statistics["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    statistics["model"] = args.model
    statistics["mode"] = args.mode
    statistics["data_type"] = args.data_type
    
    return results, statistics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_data_folder", type=str, default="data")
    parser.add_argument("--test_data", type=str, default="test.jsonl")
    parser.add_argument("--rule_folder", type=str, default="rules/human_written_rules")
    parser.add_argument("--output_folder", type=str, default="results")
    parser.add_argument("--output_prefix", type=str, default="")
    parser.add_argument("--model", type=str, default="gpt-4-0125-preview")
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--base_url", type=str, default="https://api.chatanywhere.tech/v1")
    parser.add_argument("--mode", type=str, 
                       choices=["simple", "zero_shot_cot", "cot_sc", "few_shot", "baseline"], 
                       default="simple",
                       help="Prompting strategy to use")
    parser.add_argument("--data_type", type=str, choices=["action", "tick", "score", "full"], default="action")
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--game_file_names", type=str, default="experiments/games.json")
    parser.add_argument("--no_rule", action="store_true")
    parser.add_argument("--notlive", action="store_true", default=False)
    parser.add_argument("--sleep_time", type=float, default=None,
                      help="Sleep time between API calls (overrides default)")
    args = parser.parse_args()

    # Initialize OpenAI client with proper configuration
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url
    )
    
    # Load game data
    game_data = load_game_data(args)
    
    # Create output directory
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    if not args.notlive:
        try:
            # Run experiment with the configured client
            results, statistics = run_experiment(client, game_data['test_data'], game_data, args)
            
            # Save results even if interrupted
            if results:  # Only save if we have some results
                # Add timestamp to filenames
                results_file = f"{args.output_folder}/{args.output_prefix}_{args.mode}_{timestamp}.jsonl"
                stats_file = f"{args.output_folder}/{args.output_prefix}_{args.mode}_stats_{timestamp}.json"
                accuracy_file = f"{args.output_folder}/{args.output_prefix}_{args.mode}_accuracy_{timestamp}.txt"
                
                # Save results
                with open(results_file, 'w') as f:
                    for record in results:
                        f.write(json.dumps(record) + '\n')
                
                # Save statistics
                with open(stats_file, 'w') as f:
                    json.dump(statistics, f, indent=2)
                
                # Save accuracy report
                with open(accuracy_file, 'w') as f:
                    f.write(f"Experiment Results\n")
                    f.write(f"Date: {statistics['timestamp']}\n")
                    f.write(f"Model: {statistics['model']}\n")
                    f.write(f"Mode: {statistics['mode']}\n")
                    f.write(f"Data Type: {statistics['data_type']}\n\n")
                    f.write("Per Game Accuracy:\n")
                    
                    # Iterate through all game statistics (excluding metadata keys)
                    for game, stats in ((k, v) for k, v in statistics.items() 
                                       if k not in ['timestamp', 'model', 'mode', 'data_type', 'overall']):
                        f.write(f"{game}: {stats['accuracy_rate']:.2f}% "
                                f"({stats['correct_cases']}/{stats['total_cases']} correct)\n")
                    
                    # Write overall statistics
                    f.write(f"\nOverall Statistics:\n")
                    f.write(f"Total Cases: {statistics['overall']['total_cases']}\n")
                    f.write(f"Correct Cases: {statistics['overall']['correct_cases']}\n")
                    f.write(f"Accuracy Rate: {statistics['overall']['accuracy_rate']:.2f}%\n")
                    f.write(f"Error Rate: {statistics['overall']['error_rate']:.2f}%\n")
                
                print("\nExperiment complete!")
                print(f"Results saved to: {results_file}")
                print(f"Statistics saved to: {stats_file}")
                print(f"Accuracy report saved to: {accuracy_file}")
                
                # Print final accuracy statistics
                print("\nAccuracy Statistics:")
                for game, stats in ((k, v) for k, v in statistics.items() 
                                   if k not in ['timestamp', 'model', 'mode', 'data_type', 'overall']):
                    print(f"{game}: {stats['accuracy_rate']:.2f}% "
                          f"({stats['correct_cases']}/{stats['total_cases']} correct)")
                print(f"\nOverall Accuracy: {statistics['overall']['accuracy_rate']:.2f}%")
                print(f"Overall Error Rate: {statistics['overall']['error_rate']:.2f}%")
        except KeyboardInterrupt:
            print("\nExperiment terminated by user.")
            sys.exit(0)
    else:
        print("'live' parameter is not set, will not send requests to OpenAI API.")

if __name__ == "__main__":
    main() 