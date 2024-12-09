import argparse
import tiktoken
import os
import json
import random
import sys
from math import ceil
from typing import Optional, Union
from dataclasses import dataclass

# Local Llama imports
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# API imports
from together import Together  # 如果使用 Together API

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.evaluate import evaluate, make_game_state, make_game_state_partial, evaluate_score
from requests.exceptions import ChunkedEncodingError
from llama_utils.llama_interface import LlamaConfig, LlamaInterface
from experiments.quest_gpt import (
    preprocess_obj_desc, 
    getTokenLength, 
    loadProgram, 
    postProcess, 
    recover_game_state_from_partial
)

def parse_args():
    parser = argparse.ArgumentParser()
    # 保留原有的参数
    parser.add_argument("--state_data_folder", type=str, default="data")
    parser.add_argument("--test_data", type=str, default="test.jsonl")
    parser.add_argument("--example_data", type=str, default="data/examples.json")
    parser.add_argument("--rule_folder", type=str, default="rules/human_written_rules")
    parser.add_argument("--output_folder", type=str, default="results")
    parser.add_argument("--output_prefix", type=str, default="")
    parser.add_argument("--output_suffix", type=str, default="")
    parser.add_argument("--random_seed", type=int, default=0)
    
    # 添加 Llama 特定的参数
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to local model or model identifier for API")
    parser.add_argument("--model_type", type=str, default="local",
                        choices=["local", "anyscale", "together"],
                        help="Type of model loading")
    parser.add_argument("--api_key", type=str,
                        help="API key for remote services")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run local model on")
    
    parser.add_argument("--data_distribution_file", type=str, default="data/dynamic_static_states_per_action.json")
    parser.add_argument("--state_change_file", type=str, default='data/dynamic_states.json') # fine-grained state change info

    parser.add_argument("--game_file_names", default="experiments/games.json")
    parser.add_argument("--shard_idx", type=int, default=0) # zero-base
    parser.add_argument("--total_shards", type=int, default=1)

    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--data_type", type=str, default="action", choices=["action", "tick", "score", "full"])
    parser.add_argument("--no_rule", action="store_true")

    # Make a boolean parameter, "notlive", which defaults to False
    parser.add_argument("--notlive", dest='notlive', action='store_true', default=False)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    statistics = {}

    random.seed(args.random_seed)
    # 初始化 Llama 配置
    llama_config = LlamaConfig(
        model_path=args.model_path,
        model_type=args.model_type,
        api_key=args.api_key,
        device=args.device
    )
    llm = LlamaInterface(llama_config)
    with open(args.game_file_names) as f:
        games_all = json.load(f)
        games = games_all['games']
        example_game = games_all['example']
        data_games = games[ceil(len(games)/args.total_shards)*args.shard_idx:ceil(len(games)/args.total_shards)*(args.shard_idx+1)]

    print("Number of games: " + str(len(data_games)))
    print("Data_games: " + str(data_games))

    with open(args.data_distribution_file) as f:
        data_distribution = json.load(f)

    # load state change information (the ids of states that are dynamic)
    with open(args.state_change_file) as f:
        state_change_info = json.load(f)

    # load object rules
    with open(os.path.join(args.rule_folder, f"object_rules.json")) as f:
        obj_rules = json.load(f)

    # load action rules
    if args.data_type != "tick":
        with open(os.path.join(args.rule_folder, f"action_rules.json")) as f:
            action_rules = json.load(f)

    # load score rules
    if args.data_type == "score" or args.data_type == "full":
        with open(os.path.join(args.rule_folder, f'score_rules.json')) as f:
            score_rules = json.load(f)

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
    # open example json
    with open(args.example_data) as f:
        example_lut = json.load(f)
    if args.data_type == "score":
        # Load an example that score is changed
        example_score_change = example_lut["score"]
        example_state = make_game_state(example_score_change[curr_state_key])
        example_target_state = make_game_state(example_score_change[next_state_key])
        example_curr_score_state = example_score_change["current_score_state"]
        example_next_score_state = example_score_change["next_score_state"]
        example_task_desc = example_score_change[curr_state_key]["taskDesc"]
        example_action = example_score_change[next_state_key]["lastAction"]

    if args.data_type == "tick" and len(state_change_info[example_game]['time_change']) > 0:
        # Load one example that the game state is changed over time
        # Make sure the example game has tick change states
        example_time_change = example_lut["tick"]
        example_state = make_game_state(example_time_change[curr_state_key])
        example_target_state = make_game_state(example_time_change[next_state_key])
        if args.partial:
            example_target_state_partial = make_game_state_partial(example_state, example_target_state)

        example_task_desc = example_time_change["current_state"]["taskDesc"]
        example_UUID_base = example_time_change[curr_state_key]["max_UUID"]

    elif args.data_type == "action":
        # Load one example that the game state is changed by an action
        example_action_change = example_lut["action"]
        example_state = make_game_state(example_action_change[curr_state_key])
        example_target_state = make_game_state(example_action_change[next_state_key])
        if args.partial:
            example_target_state_partial = make_game_state_partial(example_state, example_target_state)

        example_action = example_action_change[next_state_key]["lastAction"]
        example_task_desc = example_action_change["current_state"]["taskDesc"]
        example_UUID_base = example_action_change[curr_state_key]["max_UUID"]

    elif args.data_type == "full":
        # Load one example that the game state is changed by an action
        example_action_change = example_lut["full"]["action"]
        current_score_state = example_action_change["current_score_state"]
        next_score_state = example_action_change["next_score_state"]
        example_state_a = make_game_state(example_action_change[curr_state_key])
        example_state_a["game_state"].append(current_score_state)
        example_target_state_a = make_game_state(example_action_change[next_state_key])
        example_target_state_a["game_state"].append(next_score_state)
        if args.partial:
            example_target_state_a_partial = make_game_state_partial(example_state_a, example_target_state_a)

        example_action_a = example_action_change[next_state_key]["lastAction"]
        example_task_desc = example_action_change["current_state"]["taskDesc"]
        example_UUID_base_a = example_action_change[curr_state_key]["max_UUID"]

        # Load one example that the game state is changed over time
        # MAKE SURE THE EXAMPLE HAS A TIME CHANGE STATE
        time_change_states = [s for s in state_change_info[example_game]['time_change'] if s not in state_change_info[example_game]['action_change']]
        if len(time_change_states) > 0:
            example_time_change = example_lut["full"]["tick"]

            current_score_state = example_time_change["current_score_state"]
            next_score_state = example_time_change["next_score_state"]
            example_state_t = make_game_state(example_time_change[curr_state_key])
            example_state_t["game_state"].append(current_score_state)
            example_target_state_t = make_game_state(example_time_change[next_state_key])
            example_target_state_t["game_state"].append(next_score_state)
            if args.partial:
                example_target_state_t_partial = make_game_state_partial(example_state_t, example_target_state_t)

            example_action_t = example_time_change[next_state_key]["lastAction"]
            example_UUID_base_t = example_time_change[curr_state_key]["max_UUID"]
        else:
            example_time_change = None


    example_obj_desc = preprocess_obj_desc(obj_rules[example_game])

    if args.data_type != "tick":
        example_action_desc= action_rules[example_game]

    if args.data_type == "score" or args.data_type == "full":
        example_score_desc = score_rules[example_game]

    prompt_tokens = {}



    with open(os.path.join(args.state_data_folder, args.test_data)) as f:
        test_data = f.readlines()

    outputs = []
    for data_str in test_data:
        data = json.loads(data_str)

        game = data["game"]
        if game not in data_games:
            continue

        state_id = data["state_id"]
        if game not in statistics:
            statistics[game] = {"total_errors": 0, "total_states": 0}

        statistics[game]["total_states"] += 1

        is_correct = True
        errors_objects = 0
        errors_format = 0
        num_errors = 0
        total_tokens_prompt = 0

        print('\n===================================================\n')
        print(f"Processing {game}_{state_id}")

        # load game state data
        data_state = make_game_state(data[curr_state_key])
        if args.data_type == "full":
            data_state["game_state"].append(data["current_score_state"])
        if args.data_type == "score":
            data_target = data["next_score_state"]
        elif args.data_type == "full":
            data_target = make_game_state(data[next_state_key])
            score_target = data["next_score_state"]
            data_target["game_state"].append(score_target)
        else:
            data_target = make_game_state(data[next_state_key])
        data_action = data[next_state_key]["lastAction"]
        data_task_desc = data[curr_state_key]["taskDesc"]
        data_UUID_base = data[curr_state_key]["max_UUID"]
        if args.data_type == "score":
            data_curr_score = data["current_score_state"]
            data_target_state = make_game_state(data[next_state_key])

        # load rules
        data_obj_desc = preprocess_obj_desc(obj_rules[game])
        if args.data_type != "tick":
            data_action_desc = action_rules[game]
        if args.data_type == "score" or args.data_type == "full":
            data_score_desc = score_rules[game]

        # decide whether the state is dynamic
        if args.data_type == "action":
            if state_id in data_distribution[game][data_action.split()[0]]['positive']:
                state_change = True # The action to take should change the game state
            else:
                state_change = False
        elif args.data_type == "tick":
            if state_id in data_distribution[game]["tick"]['positive']:
                state_change = True # The output game state should be changed
            else:
                state_change = False
        elif args.data_type == "score":
            if state_id in data_distribution[game]["score"]['positive']:
                state_change = True # The output game score state should be changed
            else:
                state_change = False
        elif args.data_type == "full":
            if state_id in data_distribution[game][data_action.split()[0]]['positive'] or \
                state_id in data_distribution[game]["tick"]['positive'] or \
                state_id in data_distribution[game]["score"]['positive']:
                state_change = True
            else:
                state_change = False

        output_str = ''

        if args.data_type == "score":
            prompt = "You are a simulator of a text game. Read the task description of a text game. Given the current game state in JSON, you need to predict the current game score, whether the game is over, and whether the agent wins the game.\n"
        elif args.data_type == "action":
            prompt = "You are a simulator of a text game. Read the task description of a text game. Given the current game state in JSON, you need to decide the new game state after taking an action.\n"
        elif args.data_type == "tick":
            prompt = "You are a simulator of a text game. Read the task description. Given the current game state in JSON, you need to decide how the game state changes in the next time step (without considering the agent actions). Rules for such changes are described as the tick function of each object.\n"
        elif args.data_type == "full":
            prompt = "You are a simulator of a text game. Read the task description of a text game. Given the current game state in JSON, you need to decide the new game state after taking an action including the game score.\n"

        if args.data_type != "score":
            prompt += "You may need to create new objects when you predict the new game state. You should assign the uuid of new objects starting from the UUID base given in the instructions."

        if args.partial and args.data_type in ("action", "tick"):
            prompt += "Your response should be in the JSON format. It should have two keys: 'modified' and 'removed'. The 'modified' key stores a list of all the object states that are added or changed after taking the action. Keep it an empty list if no object is added or modified. The 'removed' key stores a list of uuids of the objects that are removed. Keep it an empty list if no object is removed.\n"
        elif args.partial and args.data_type in ("full"):
            prompt += "Your response should be in the JSON format. It should have three keys: 'modified', 'removed', and 'score'. The 'modified' key stores a list of all the object states that are added or changed after taking the action. Keep it an empty list if no object is added or modified. The 'removed' key stores a list of uuids of the objects that are removed. Keep it an empty list if no object is removed. The 'score' key stores a JSON with three keys: 'score', 'gameOver', and 'gameWon'. 'score' stores the current game score, 'gameOver' stores a bool value on whether the game is over, and 'gameWon' stores a bool value on whether the game is won. \n"
        elif args.data_type == "score":
            prompt += "Your response should be a JSON with three keys: 'score', 'gameOver', and 'gameWon'. 'score' stores the current game score, 'gameOver' stores a bool value on whether the game is over, and 'gameWon' stores a bool value on whether the game is won.\n"
        else:
            prompt += "Your response should be in the same JSON format as the given game state.\n"

        if not args.data_type == "full":
            prompt += "Here is an example:\n"

            prompt += "Example game task description:\n"
            prompt += f"{example_task_desc}\n"
            if not args.no_rule:
                prompt += "Here are the descriptions of all game objects properties in the example game:\n"
                prompt += example_obj_desc.strip()
                prompt += "\n"
                if args.data_type == "action":
                    prompt += "Here are the descriptions of all game actions in the example game:\n"
                    prompt += example_action_desc.strip()
                    prompt += '\n'

                if args.data_type == "score":
                    prompt += "Here is a description of the game score function:\n"
                    prompt += example_score_desc.strip()
                    prompt += '\n'

            if args.data_type == "score":
                prompt += "Here is the previous game state:\n"
            else:
                prompt += "Here is the game state:\n"
            prompt += f'{example_state}\n'
            prompt += '\n'

            if args.data_type == "score":
                prompt += f"The game score of the preivous state is:\n{example_curr_score_state}\n"
            else:
                prompt += f"The current game UUID base is {example_UUID_base}\n"

            if args.data_type == "action" or args.data_type == "score":
                prompt += f"The action to take is: {example_action}\n"


            if args.data_type == "score":
                prompt += f"Here is the current game state after taking the action:"
                prompt += f"{example_target_state}\n"

            prompt += "The expected response is:\n"
            if args.data_type == "score":
                prompt += f'{example_next_score_state}\n'
            else:
                if args.partial:
                    prompt += f'{example_target_state_partial}\n'
                else:
                    prompt += f'{example_target_state}\n'
            prompt += '\n'

        else:

            prompt += "Note that while game states can be changed by actions, some game states may change over the time, which is described in the tick function of each object class. \n"
            if example_time_change is not None:
                prompt += "Here are two examples of both cases. Both examples are from the same example game.\n"

            prompt += "Example game task description:\n"
            prompt += f"{example_task_desc}\n"
            if not args.no_rule:
                prompt += "Here are the descriptions of all game objects properties in the example game:\n"
                prompt += example_obj_desc.strip()
                prompt += "\n"
                prompt += "Here are the descriptions of all game actions in the example game:\n"
                prompt += example_action_desc.strip()
                prompt += '\n'
                prompt += "Here is a description of the score function of the example game:\n"
                prompt += example_score_desc.strip()
                prompt += '\n'

            if example_time_change is not None:
                # Example 1: a game state is changed by an action
                prompt += "In the first example, the game state is changed by an action:\n"

            prompt += "Here is the game state:\n"
            prompt += f'{example_state_a}\n'
            prompt += '\n'

            prompt += f"The current game UUID base is {example_UUID_base_a}\n"

            prompt += f"The action to take is: {example_action_a}\n"
            prompt += "The expected response is:\n"
            if args.partial:
                prompt += f'{example_target_state_a_partial}\n'
            else:
                prompt += f'{example_target_state_a}\n'
            prompt += '\n'

            # Example 2: a game state is changed over time
            if example_time_change is not None:
                prompt += "In the second example from the same example game, the game state is changed over the time. Note that while in this example the game state is changed by time only, it is possible that a game state is changed by both an action and time.\n"

                prompt += "Here is the game state:\n"
                prompt += f'{example_state_t}\n'
                prompt += '\n'

                prompt += f"The current game UUID base is {example_UUID_base_t}\n"
                prompt += f"The action to take is: {example_action_t}\n"
                prompt += "The expected response is:\n"
                if args.partial:
                    prompt += f'{example_target_state_t_partial}\n'
                else:
                    prompt += f'{example_target_state_t}\n'
                prompt += '\n'

        # Task
        prompt += "Here is the game that you need to simulate:\n"
        prompt += "Task Description:\n"
        prompt += f"{data_task_desc}\n"
        if not args.no_rule:
            prompt += "Here are the descriptions of all game objects properties:\n"
            prompt += data_obj_desc.strip()
            prompt += "\n"
            if args.data_type == "action" or args.data_type == "full":
                prompt += "Here are the descriptions of all game actions:\n"
                prompt += data_action_desc.strip()
                prompt += '\n'
            if args.data_type == "score" or args.data_type == "full":
                prompt += "Here is a description of the game score function:\n"
                prompt += data_score_desc.strip()
                prompt += '\n'


        if args.data_type == "score":
            prompt += "Here is the previous game state:\n"
        else:
            prompt += "Here is the game state:\n"
        prompt += f'{data_state}\n'
        prompt += '\n'

        if args.data_type == "score":
            prompt += f"The game score of the preivous state is:\n{data_curr_score}\n"
        else:
            prompt += f"The current game UUID base is {data_UUID_base}\n"

        if args.data_type == "action" or args.data_type == "score" or args.data_type == "full":
            prompt += f"The action to take is:\n{data_action}\n"

        if args.data_type == "score":
            prompt += f"Here is the current game state after taking the action:\n"
            prompt += f"{data_target_state}\n"

        output_str += 'Prompt:\n'
        output_str += prompt
        output_str += '\n===================================================\n'

        print(prompt)
        numTokens_prompt = getTokenLength(prompt)
        total_tokens_prompt += numTokens_prompt

        if not args.notlive:

            response = llm.generate(
                prompt=prompt, 
                stream=True,
                response_format={"type": "json_object"}
            )
            print(response)

            numTokens_response = getTokenLength(response)
            print("")
            print("Responded with " + str(numTokens_response) + " tokens.")
            print("")

            output_str += 'Response:\n'
            output_str += response
            output_str += '\n===================================================\n'
            output_str += 'Target:\n'
            output_str += json.dumps(data_target)
            output_str += '\n===================================================\n'

            try:
                prediction = json.loads(response)
                if args.partial and args.data_type != "score":
                    if args.data_type == "full":
                        has_score = True
                    else:
                        has_score = False
                    prediction = recover_game_state_from_partial(data_state, prediction, has_score=has_score)
            except Exception as e:
                # evaluate() will handle this format error
                print(e)
                prediction = response

            if args.data_type == "score":
                num_errors, eval_out_str = evaluate_score(prediction, data_target)
            elif args.data_type == "full":
                num_errors, num_score_errors, eval_out_str = evaluate(prediction, data_target, data_action, evaluate_score=True)
            else:
                num_errors, eval_out_str = evaluate(prediction, data_target, data_action)

            output_str += "Evaluation:\n"
            output_str += eval_out_str
            output_str += '\n===================================================\n'

            if args.data_type == "score":
                output_record = {"game": game,
                            "state_id":state_id,
                            "curr_state": data_state,
                            "prompt": prompt,
                            "gold_state": data_target,
                            "predicted_state_raw": response,
                            "predicted_state": prediction,
                            "num_errors": num_errors,
                            "eval_out_str": eval_out_str,
                            "model": args.model
                            }
            elif args.data_type == "full":
                output_record = {"game": game,
                                "state_id":state_id,
                                "curr_state": data_state,
                                "action": data_action,
                                "prompt": prompt,
                                "gold_state": data_target,
                                "predicted_state_raw": response,
                                "predicted_state": prediction,
                                "num_errors": num_errors,
                                "num_score_errors": num_score_errors,
                                "eval_out_str": eval_out_str,
                                "model": args.model
                                }
            else:
                output_record = {"game": game,
                                "state_id":state_id,
                                "curr_state": data_state,
                                "action": data_action,
                                "prompt": prompt,
                                "gold_state": data_target,
                                "predicted_state_raw": response,
                                "predicted_state": prediction,
                                "num_errors": num_errors,
                                "eval_out_str": eval_out_str,
                                "model": args.model
                                }
            # incorrect format
            if num_errors < 0:
                is_correct = False
                errors_format += 1

            if num_errors > 0:
                is_correct = False
                errors_objects += num_errors

            if args.data_type == "full" and num_score_errors > 0:
                is_correct = False

            if args.data_type == "score":
                statistics[game][state_id] = {
                    "prompt_tokens": numTokens_prompt,
                    "response_tokens": numTokens_response,
                    "state_change": state_change,
                    "objprop_errors":num_errors}
            elif args.data_type == "full":
                statistics[game][state_id] = {
                    "prompt_tokens": numTokens_prompt,
                    "response_tokens": numTokens_response,
                    "action": data_action,
                    "state_change": state_change,
                    "objprop_errors":num_errors,
                    "score_errors": num_score_errors}
            else:
                statistics[game][state_id] = {
                    "prompt_tokens": numTokens_prompt,
                    "response_tokens": numTokens_response,
                    "action": data_action,
                    "state_change": state_change,
                    "objprop_errors":num_errors}



            outputs.append(output_record)
        else:
            print("'live' parameter is not set, will not send requests to OpenAI API.")



        print(f"Total tokens: {total_tokens_prompt}")
        print(f"Game {game}, State {state_id}, Num_errors: {num_errors}")
        if not is_correct:
            statistics[game]["total_errors"] += 1
        if game not in prompt_tokens:
            prompt_tokens[game] = total_tokens_prompt
        else:
            prompt_tokens[game] += total_tokens_prompt

    #
    # results saving
    #

    os.makedirs(args.output_folder, exist_ok=True)

    if outputs:
        # save statics
        with open(f"{args.output_folder}/results_{args.output_prefix}_{args.shard_idx}{args.output_suffix}.json", "w") as f:
            json.dump(statistics, f, indent=4)

        # save predicts states
        output_file = f"{args.output_folder}/{args.output_prefix}_{args.shard_idx}{args.output_suffix}.jsonl"
        with open(output_file, 'w') as f:
            for line in outputs:
                f.write(json.dumps(line)+ '\n')

    print(statistics)
    print(prompt_tokens)
    print(f"Total prompt tokens: {sum([prompt_tokens[game] for game in prompt_tokens])}")

if __name__ == "__main__":
    main()