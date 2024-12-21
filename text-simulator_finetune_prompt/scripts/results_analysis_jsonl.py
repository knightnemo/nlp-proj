import argparse
import json
import os
import pandas as pd
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.evaluate import get_state_diff_detail_v2, load_jsonl_as_dict

def parse_args():
    parser = argparse.ArgumentParser()
    # Prefix of experiment output files
    parser.add_argument("--prefix", type=str)
    # Suffix of the experiment output files
    parser.add_argument("--suffix", type=str, default="")
    # Number of experiment shards
    parser.add_argument("--n_shards", type=int, default=4)
    # Output folder
    parser.add_argument("--output_folder", type=str, default="results")
    # Experiment type
    parser.add_argument("--exp_type", choices=["action", "tick", "score", "full"])
    # Folder that saves the results
    parser.add_argument("--results_folder", type=str, default="results")
    parser.add_argument("--state_change_file", type=str, default="data/dynamic_states.json")
    args = parser.parse_args()
    return args

def score_statistics(args):
    prefix = args.prefix
    suffix = args.suffix
    general_results = {"game": [], "total states": [], "total_errors": []}

    for i in range(args.n_shards):
        with open(f"{args.results_folder}/results_{prefix}_{i}{suffix}.jsonl") as f:
            data = [json.loads(line) for line in f]

        for game in data:
            if game == 'time':
                continue

            general_results["game"].append(game)
            general_results["total states"].append(data[game]['total_states'])
            general_results["total_errors"].append(data[game]['total_errors'])

    # Number of object property errors, score errors, and total states
    df_general = pd.DataFrame(general_results)
    df_general.to_csv(f'{args.output_folder}/{prefix}_general.csv', index=False)

def per_state_statistics(args):
    prefix = args.prefix
    suffix = args.suffix
    general_results = {"game": [], "total states": [], "total_errors": []}
    action_results = {}
    state_change_results = {}
    wrong_output_formats = {}

    for i in range(args.n_shards):
        try:
            with open(f"{args.results_folder}/{prefix}_{i}{suffix}.jsonl") as f:
                data = [json.loads(line) for line in f]
                
            prediction_dict = load_jsonl_as_dict(f"{args.results_folder}/{prefix}_{i}{suffix}.jsonl")

            for entry in data:
                if isinstance(entry, dict) and 'game' in entry:
                    game = entry['game']
                    general_results["game"].append(game)
                    # Use get() with default values for potentially missing keys
                    general_results["total states"].append(entry.get('total_states', 0))
                    general_results["total_errors"].append(entry.get('total_errors', 0))

                    state_change_results[game] = {
                        "correct_unchanged": 0,
                        "total_unchanged_states": 0,
                        "correct_changed": 0,
                        "total_changed_states": 0
                    }

                    wrong_output_formats[game] = {
                        "wrong_state_format": 0,
                        "total_states": 0
                    }

                    # Access states safely
                    states = entry.get('states', [])
                    for state in states:
                        if not isinstance(state, dict):
                            continue
                            
                        action = state.get('action', '')
                        action_verb = action.split()[0] if action else ''

                        is_correct = (state.get('objprop_errors', 0) == 0)
                        wrong_output_format = (state.get('objprop_errors', 0) < 0)
                        is_changed = state.get('state_change', False)

                        # Update statistics
                        if is_changed:
                            if is_correct:
                                state_change_results[game]["correct_changed"] += 1
                            state_change_results[game]["total_changed_states"] += 1
                        else:
                            if is_correct:
                                state_change_results[game]["correct_unchanged"] += 1
                            state_change_results[game]["total_unchanged_states"] += 1

                        # Initialize action results if needed
                        if action_verb:
                            if action_verb not in action_results:
                                action_results[action_verb] = {
                                    "correct_unchanged": 0,
                                    "total_unchanged_states": 0,
                                    "correct_changed": 0,
                                    "total_changed_states": 0
                                }
                            
                            # Update action statistics
                            if is_changed:
                                if is_correct:
                                    action_results[action_verb]["correct_changed"] += 1
                                action_results[action_verb]["total_changed_states"] += 1
                            else:
                                if is_correct:
                                    action_results[action_verb]["correct_unchanged"] += 1
                                action_results[action_verb]["total_unchanged_states"] += 1

                        if wrong_output_format:
                            wrong_output_formats[game]["wrong_state_format"] += 1
                        wrong_output_formats[game]["total_states"] += 1

        except Exception as e:
            print(f"Error processing shard {i}: {str(e)}")
            continue

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # Prepare and save results
    def prepare_results_for_df(data, index_name):
        output = {index_name: []}
        if data:  # Check if data is not empty
            for key in data[list(data.keys())[0]]:
                output[key] = []

            for key in data:
                output[index_name].append(key)
                for sub_key in data[key]:
                    output[sub_key].append(data[key][sub_key])
        return output

    df_state = pd.DataFrame(prepare_results_for_df(state_change_results, 'game'))
    df_state.to_csv(f'{args.output_folder}/{prefix}_state.csv', index=False)

def per_property_statistics(args):
    prefix = args.prefix
    suffix = args.suffix
    n_shards = args.n_shards

    general_results = {"game": [], "state_id": [], "format": [], "action": [], 'uuid': [], "name": [], 'type': [], "obj_stat": [], 'gold_stat': [], 'prop_key': [], "prop": [], "is_modified": [], "gold_prop": []}

    for i in range(n_shards):
        with open(f"{args.results_folder}/{prefix}_{i}{suffix}.jsonl") as f:
            data = [json.loads(line) for line in f]

        prediction_dict = load_jsonl_as_dict(f"{args.results_folder}/{prefix}_{i}{suffix}.jsonl")

        for entry in data:
            if isinstance(entry, dict) and 'game' in entry:
                game = entry['game']
                for state in entry.get('states', []):
                    if state in ['total_states', 'total_errors', "time"]:
                        continue

                    state_id = int(state)
                    state_data = entry['states'][state]
                    action_verb = state_data["action"].split()[0]
                    wrong_output_format = (state_data["objprop_errors"] < 0)

                    results_data = prediction_dict[game][state_id]
                    curr_state = results_data['curr_state']
                    gold_state = results_data['gold_state']
                    predicted_state = results_data['predicted_state']

                    if args.exp_type == "full":
                        curr_state["game_state"] = curr_state['game_state'][:-1]
                        gold_state["game_state"] = gold_state["game_state"][:-1]
                        if results_data["num_errors"] != -1 and type(predicted_state["game_state"]) == list and "score" in predicted_state["game_state"][-1]:
                            predicted_state["game_state"] = predicted_state["game_state"][:-1]

                    if not wrong_output_format:
                        diffs_gold = get_state_diff_detail_v2(curr_state, gold_state)

                        if len(diffs_gold["added"]) == 0 and len(diffs_gold["removed"]) == 0 and len([x for x in diffs_gold["modified"] if x[-1] != 1]) == 0:
                            positive = 0
                        else:
                            positive = 1

                        gold_stat = {}
                        for _, obj2 in diffs_gold["added"]:
                            gold_stat[obj2["uuid"]] = {'contains': 1}
                            for key in obj2["properties"]:
                                gold_stat[obj2["uuid"]][key] = 'na'

                        for obj1, _ in diffs_gold["removed"]:
                            gold_stat[obj1["uuid"]] = {'contains': 1}
                            for key in obj1["properties"]:
                                gold_stat[obj1["uuid"]][key] = 'na'

                        for key, state_1, state_2, state_code in diffs_gold["modified"]:
                            if state_2 is not None:
                                uuid = state_2["uuid"]
                            else:
                                uuid = state_1["uuid"]

                            if uuid not in gold_stat:
                                gold_stat[uuid] = {}

                            gold_stat[uuid][key] = 0 if state_code == 1 else 1

                        try:
                            diffs = get_state_diff_detail_v2(gold_state, predicted_state)
                            diffs_modification = get_state_diff_detail_v2(curr_state, predicted_state)
                        except Exception as e:
                            wrong_output_format = True
                        else:
                            unmodified_prop = {}
                            for key, state_1, state_2, state_code in diffs_modification["modified"]:
                                if state_2 is not None:
                                    uuid = state_2["uuid"]
                                else:
                                    uuid = state_1["uuid"]
                                if state_code == 1:
                                    if uuid not in unmodified_prop:
                                        unmodified_prop[uuid] = set()
                                    unmodified_prop[uuid].add(key)

                            for _, obj2 in diffs["added"]:
                                if type(obj2) == dict and "uuid" in obj2 and "name" in obj2 and "type" in obj2:
                                    general_results["game"].append(game)
                                    general_results["state_id"].append(state_id)
                                    general_results["action"].append(action_verb)
                                    general_results["uuid"].append(obj2["uuid"])
                                    general_results["name"].append(obj2["name"])
                                    general_results["type"].append(obj2["type"])
                                    general_results["obj_stat"].append(2)
                                    general_results["gold_stat"].append(positive)
                                    general_results["prop_key"].append("na")
                                    general_results["prop"].append("na")
                                    general_results["gold_prop"].append("na")
                                    general_results["is_modified"].append("na")
                                    general_results["format"].append(1)

                            for obj1, _ in diffs["removed"]:
                                general_results["game"].append(game)
                                general_results["state_id"].append(state_id)
                                general_results["action"].append(action_verb)
                                general_results["uuid"].append(obj1["uuid"])
                                general_results["name"].append(obj1["name"])
                                general_results["type"].append(obj1["type"])
                                general_results["obj_stat"].append(0)
                                general_results["gold_stat"].append(positive)
                                general_results["prop_key"].append("na")
                                general_results["prop"].append("na")
                                general_results["gold_prop"].append("na")
                                general_results["is_modified"].append("na")
                                general_results["format"].append(1)

                            for key, state_1, state_2, state_code in diffs["modified"]:
                                general_results["game"].append(game)
                                general_results["state_id"].append(state_id)
                                general_results["action"].append(action_verb)

                                if state_1 is not None:
                                    uuid = state_1['uuid']
                                    general_results["uuid"].append(state_1['uuid'])
                                    general_results["name"].append(state_1["name"])
                                    general_results["type"].append(state_1["type"])
                                else:
                                    uuid = state_2['uuid']
                                    general_results["uuid"].append(state_2['uuid'])
                                    general_results["name"].append(state_2["name"])
                                    general_results["type"].append(state_2["type"])

                                if uuid in diffs["same"]:
                                    obj_code = 3
                                else:
                                    obj_code = 1

                                general_results["obj_stat"].append(obj_code)
                                general_results["gold_stat"].append(positive)
                                general_results["prop_key"].append(key)
                                general_results["prop"].append(state_code)

                                if state_code == 3:
                                    general_results["is_modified"].append("na")
                                else:
                                    if uuid in unmodified_prop and key in unmodified_prop[uuid]:
                                        general_results["is_modified"].append(0)
                                    else:
                                        general_results["is_modified"].append(1)

                                if state_code == 3:
                                    general_results["gold_prop"].append("na")
                                else:
                                    general_results["gold_prop"].append(gold_stat[uuid][key])

                                general_results["format"].append(1)

    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    df_detail = pd.DataFrame(general_results)
    df_detail.to_csv(f'{args.output_folder}/{prefix}_detail.csv', index=False)

def main():
    args = parse_args()
    if args.exp_type == "score":
        score_statistics(args)
    else:
        per_state_statistics(args)
        per_property_statistics(args)

if __name__ == "__main__":
    main()