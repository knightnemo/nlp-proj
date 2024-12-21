import json
import re
from utils.experiment_utils import extract_confidence, select_majority_prediction
from scripts.evaluate import evaluate, make_game_state, make_game_state_partial
from experiments.quest_gpt import (
    preprocess_obj_desc,
    recover_game_state_from_partial
)
import random
from together import Together

class PromptStrategy:
    def __init__(self, llm, object_desc, game_data, data_type="action", partial=False):
        self.llm = llm
        self.object_desc = object_desc
        self.game_data = game_data
        self.data_type = data_type
        self.partial = partial
        self.last_prompt = None
        self.last_response = None
        self.no_rule = False

    def _get_base_instruction(self):
        if self.data_type == "score":
            return "You are a simulator of a text game. Read the task description of a text game. Given the current game state in JSON, you need to predict the current game score, whether the game is over, and whether the agent wins the game."
        elif self.data_type == "action":
            return "You are a simulator of a text game. Read the task description of a text game. Given the current game state in JSON, you need to decide the new game state after taking an action."
        elif self.data_type == "tick":
            return "You are a simulator of a text game. Read the task description. Given the current game state in JSON, you need to decide how the game state changes in the next time step (without considering the agent actions). Rules for such changes are described as the tick function of each object."
        elif self.data_type == "full":
            return "You are a simulator of a text game. Read the task description of a text game. Given the current game state in JSON, you need to decide the new game state after taking an action including the game score."

    def get_prediction(self, item):
        prompt = self._build_prompt(item)
        self.last_prompt = prompt
        
        try:
            # Don't use response_format for Together API
            if isinstance(self.llm.client, Together):
                try:
                    raw_response = self.llm.generate(
                        prompt=prompt,
                        stream=True
                    )
                    print("\nRaw API Response:")
                    print("-" * 40)
                    print("Type:", type(raw_response))
                    print("Content:", raw_response)
                    if hasattr(raw_response, 'choices'):
                        print("Choices:", raw_response.choices)
                    if hasattr(raw_response, 'delta'):
                        print("Delta:", raw_response.delta)
                    print("-" * 40)
                    
                    # Store the raw response and convert to string if needed
                    self.last_response = raw_response
                    if not isinstance(raw_response, str):
                        raw_response = str(raw_response)
                    
                    print("\nRaw Response Content:")
                    print("-" * 40)
                    print(raw_response)
                    print("-" * 40)
                    
                    try:
                        # Clean and parse the response
                        response = raw_response.strip()
                        # Find the first { and last } to extract JSON
                        start_idx = response.find('{')
                        end_idx = response.rfind('}')
                        if start_idx != -1 and end_idx != -1:
                            json_str = response[start_idx:end_idx + 1]
                            print("\nExtracted JSON:")
                            print("-" * 40)
                            print(json_str)
                            print("-" * 40)
                            prediction = json.loads(json_str)
                        else:
                            print("\nNo JSON structure found in response")
                            return None
                        
                        if self.data_type == "score":
                            if not all(k in prediction for k in ["score", "gameOver", "gameWon"]):
                                prediction.update({
                                    "score": prediction.get("score", 0),
                                    "gameOver": prediction.get("gameOver", False),
                                    "gameWon": prediction.get("gameWon", False)
                                })
                        else:
                            if "game_state" not in prediction:
                                if isinstance(prediction, dict):
                                    prediction = {"game_state": [prediction]}
                                else:
                                    print(f"Invalid prediction structure: {prediction}")
                                    return None
                        
                        if self.partial and self.data_type != "score":
                            prediction = recover_game_state_from_partial(
                                item['current_state'],
                                prediction,
                                has_score=(self.data_type == "full")
                            )
                        
                        return prediction
                        
                    except json.JSONDecodeError as e:
                        print(f"JSON parsing error: {e}")
                        print(f"Attempted to parse: {json_str if 'json_str' in locals() else 'No JSON found'}")
                        return None
                    
                except Exception as e:
                    print(f"\nAPI Call Error Details:")
                    print("-" * 40)
                    print(f"Error type: {type(e)}")
                    print(f"Error message: {str(e)}")
                    if hasattr(e, '__dict__'):
                        print("Error attributes:", e.__dict__)
                    print("-" * 40)
                    raise
                    
            else:
                raw_response = self.llm.generate(
                    prompt=prompt,
                    stream=True,
                    response_format={"type": "json_object"}
                )
                
                # Store the raw response
                self.last_response = raw_response
                
                print("\nRaw Response Content:")
                print("-" * 40)
                print(raw_response)
                print("-" * 40)
                
                try:
                    # Clean and parse the response
                    response = raw_response.strip()
                    # Find the first { and last } to extract JSON
                    start_idx = response.find('{')
                    end_idx = response.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_str = response[start_idx:end_idx + 1]
                        print("\nExtracted JSON:")
                        print("-" * 40)
                        print(json_str)
                        print("-" * 40)
                        prediction = json.loads(json_str)
                    else:
                        print("\nNo JSON structure found in response")
                        return None
                    
                    if self.data_type == "score":
                        if not all(k in prediction for k in ["score", "gameOver", "gameWon"]):
                            prediction.update({
                                "score": prediction.get("score", 0),
                                "gameOver": prediction.get("gameOver", False),
                                "gameWon": prediction.get("gameWon", False)
                            })
                    else:
                        if "game_state" not in prediction:
                            if isinstance(prediction, dict):
                                prediction = {"game_state": [prediction]}
                            else:
                                print(f"Invalid prediction structure: {prediction}")
                                return None
                    
                    if self.partial and self.data_type != "score":
                        prediction = recover_game_state_from_partial(
                            item['current_state'],
                            prediction,
                            has_score=(self.data_type == "full")
                        )
                    
                    return prediction
                    
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {e}")
                    print(f"Attempted to parse: {json_str if 'json_str' in locals() else 'No JSON found'}")
                    return None
                
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return None

    def _build_prompt(self, item):
        raise NotImplementedError("Each strategy must implement _build_prompt")

class SimpleCoTStrategy(PromptStrategy):
    def _build_prompt(self, item):
        prompt = self._get_base_instruction() + "\n\n"
        
        if 'current_state' in item and 'taskDesc' in item['current_state']:
            prompt += f"Task Description:\n{item['current_state']['taskDesc']}\n\n"
        
        if not self.no_rule:
            prompt += f"Here are the descriptions of all game objects properties:\n{self.object_desc}\n\n"
            if self.data_type in ["action", "full"]:
                prompt += f"Here are the descriptions of all game actions:\n{self.game_data['action_rules'][item['game']]}\n\n"
            if self.data_type in ["score", "full"]:
                prompt += f"Here is a description of the game score function:\n{self.game_data['score_rules'][item['game']]}\n\n"
        
        prompt += "Here is the game state:\n"
        prompt += f"{json.dumps(item['current_state'])}\n\n"
            
        prompt += f"The current game UUID base is {item['current_state'].get('max_UUID', 0)}\n"
        if self.data_type in ["action", "score", "full"]:
            action = item.get('action') or item['current_state'].get('lastAction', '')
            prompt += f"The action to take is: {action}\n\n"

        if 'explanation' in item:
            prompt += f"\nExplanation of state changes:\n{item['explanation']}\n\n"
        
        return prompt

class ZeroShotCoTStrategy(PromptStrategy):
    def get_prediction(self, item):
        reasoning_prompt = self._build_prompt(item)
        self.last_prompt = reasoning_prompt
        
        reasoning = self.llm.generate(
            prompt=reasoning_prompt + "\nStart your reasoning here:\n",
            stream=True
        )
        
        extraction_prompt = f"""Based on the reasoning below, generate a game state JSON object.
Your response must:
1. Start with a curly brace {{
2. End with a curly brace }}
3. Contain only valid JSON
4. Not include any explanatory text before or after the JSON
5. Not include markdown code block markers

Reasoning:
{reasoning}

Generate the JSON now:"""
        
        response = self.llm.generate(
            prompt=extraction_prompt,
            stream=True,
            response_format={"type": "json_object"}
        )
        self.last_response = response
        
        try:
            response = response.strip()
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            
            if start_idx == -1 or end_idx == -1:
                print(f"No JSON found in response: {response[:100]}...")
                return None
                
            json_str = response[start_idx:end_idx + 1]
            prediction = json.loads(json_str)
            
            if self.data_type == "score":
                prediction = {
                    "score": prediction.get("score", 0),
                    "gameOver": prediction.get("gameOver", False),
                    "gameWon": prediction.get("gameWon", False)
                }
            else:
                if not isinstance(prediction.get("game_state"), list):
                    if isinstance(prediction, dict):
                        prediction = {"game_state": [prediction]}
                    else:
                        print(f"Invalid prediction structure: {prediction}")
                        return None
            
            if self.partial and self.data_type != "score":
                prediction = recover_game_state_from_partial(
                    item['current_state'],
                    prediction,
                    has_score=(self.data_type == "full")
                )
                
            return prediction
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response was: {response[:200]}...")
            return None
        except Exception as e:
            print(f"Error processing prediction: {e}")
            print(f"Response was: {response[:200]}...")
            return None

    def _build_prompt(self, item):
        prompt = self._get_base_instruction() + "\n\n"
        
        if 'current_state' in item and 'taskDesc' in item['current_state']:
            prompt += f"Task Description:\n{item['current_state']['taskDesc']}\n\n"
        
        if not self.no_rule:
            prompt += f"Here are the descriptions of all game objects properties:\n{self.object_desc}\n\n"
            if self.data_type in ["action", "full"]:
                prompt += f"Here are the descriptions of all game actions:\n{self.game_data['action_rules'][item['game']]}\n\n"
            if self.data_type in ["score", "full"]:
                prompt += f"Here is a description of the game score function:\n{self.game_data['score_rules'][item['game']]}\n\n"
        
        prompt += "Here is the game state:\n"
        prompt += f"{json.dumps(item['current_state'])}\n\n"
        
        prompt += f"The current game UUID base is {item['current_state'].get('max_UUID', 0)}\n"
        if self.data_type in ["action", "score", "full"]:
            action = item.get('action') or item['current_state'].get('lastAction', '')
            prompt += f"The action to take is: {action}\n\n"
        
        prompt += "Let's solve this step by step. Think through each step carefully:\n"
        prompt += "1. First, analyze the current state and understand what objects are present\n"
        prompt += "2. Consider the rules and how they apply to the current situation\n"
        prompt += "3. Determine what changes will occur based on the rules\n"
        prompt += "4. Explain how the state will change\n\n"

        if 'explanation' in item:
            prompt += f"\nExplanation of state changes:\n{item['explanation']}\n\n"
            
        prompt += "Reasoning:\n"
        return prompt

class FewShotStrategy(PromptStrategy):
    def __init__(self, llm, object_desc, game_data, num_examples=3, **kwargs):
        super().__init__(llm, object_desc, game_data, **kwargs)
        self.num_examples = num_examples
        
        with open("data/few_shot_examples.json") as f:
            self.few_shot_examples = json.load(f)

    def _get_examples(self, item):
        examples = []
        
        game_examples = self.few_shot_examples.get(item['game'], [])
        if not game_examples:
            return examples
            
        num_to_sample = min(len(game_examples), self.num_examples)
        selected_examples = random.sample(game_examples, num_to_sample)
        
        for example in selected_examples:
            example_data = {
                'task_desc': example['current_state']['taskDesc'],
                'state': example['current_state'],
                'target': example['action_state'] if self.data_type == 'action' else example['tick_state'],
                'action': example['action_state']['lastAction'] if self.data_type in ['action', 'full'] else None,
                'explanation': example.get('explanation', 'No explanation provided for this example.')
            }
            examples.append(example_data)
            
        return examples

    def _build_prompt(self, item):
        prompt = self._get_base_instruction() + "\n\n"
        
        examples = self._get_examples(item)
        for i, example_data in enumerate(examples, 1):
            prompt += f"Example {i}:\n\n"
            prompt += f"Task Description:\n{example_data['task_desc']}\n\n"
            prompt += f"Initial state:\n{json.dumps(example_data['state'])}\n\n"
            if example_data.get('action'):
                prompt += f"Action: {example_data['action']}\n\n"
            prompt += f"Resulting state:\n{json.dumps(example_data['target'])}\n\n"
            prompt += f"Explanation of state changes:\n{example_data['explanation']}\n\n"
        
        prompt += "Now for our current case:\n\n"
        
        if 'current_state' in item and 'taskDesc' in item['current_state']:
            prompt += f"Task Description:\n{item['current_state']['taskDesc']}\n\n"
        
        if not self.no_rule:
            prompt += f"Here are the descriptions of all game objects properties:\n{self.object_desc}\n\n"
            if self.data_type in ["action", "full"]:
                prompt += f"Here are the descriptions of all game actions:\n{self.game_data['action_rules'][item['game']]}\n\n"
            if self.data_type in ["score", "full"]:
                prompt += f"Here is a description of the game score function:\n{self.game_data['score_rules'][item['game']]}\n\n"
        
        prompt += "Here is the game state:\n"
        prompt += f"{json.dumps(item['current_state'])}\n\n"
        
        prompt += f"The current game UUID base is {item['current_state'].get('max_UUID', 0)}\n"
        if self.data_type in ["action", "score", "full"]:
            action = item.get('action') or item['current_state'].get('lastAction', '')
            prompt += f"The action to take is: {action}\n\n"

        if 'explanation' in item:
            prompt += f"\nExplanation of state changes:\n{item['explanation']}\n\n"
            
        prompt += "Generate the next game state as a JSON object.\n"
        return prompt

class CoTSCStrategy(PromptStrategy):
    def __init__(self, llm, object_desc, game_data, num_samples=5, **kwargs):
        super().__init__(llm, object_desc, game_data, **kwargs)
        self.num_samples = num_samples
    
    def get_prediction(self, item):
        predictions = []
        
        for _ in range(self.num_samples):
            reasoning_prompt = self._build_prompt(item)
            self.last_prompt = reasoning_prompt
            
            reasoning = self.llm.generate(
                prompt=reasoning_prompt + "\nStart your reasoning here:\n",
                stream=True
            )
            
            extraction_prompt = f"""Based on the reasoning below, generate a game state JSON object.
Your response must:
1. Start with a curly brace {{
2. End with a curly brace }}
3. Contain only valid JSON
4. Not include any explanatory text before or after the JSON
5. Not include markdown code block markers

Reasoning:
{reasoning}

Generate the JSON now:"""
            
            response = self.llm.generate(
                prompt=extraction_prompt,
                stream=True,
                response_format={"type": "json_object"}
            )
            self.last_response = response
            
            try:
                response = response.strip()
                start_idx = response.find('{')
                end_idx = response.rfind('}')
                
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx + 1]
                    prediction = json.loads(json_str)
                    predictions.append(prediction)
            except Exception as e:
                print(f"Failed to parse sample: {e}")
                continue
                
        if not predictions:
            return None

        return select_majority_prediction(predictions)

    def _build_prompt(self, item):
        prompt = self._get_base_instruction() + "\n\n"
        
        if 'current_state' in item and 'taskDesc' in item['current_state']:
            prompt += f"Task Description:\n{item['current_state']['taskDesc']}\n\n"
        
        if not self.no_rule:
            prompt += f"Here are the descriptions of all game objects properties:\n{self.object_desc}\n\n"
            if self.data_type in ["action", "full"]:
                prompt += f"Here are the descriptions of all game actions:\n{self.game_data['action_rules'][item['game']]}\n\n"
            if self.data_type in ["score", "full"]:
                prompt += f"Here is a description of the game score function:\n{self.game_data['score_rules'][item['game']]}\n\n"
        
        prompt += "Here is the game state:\n"
        prompt += f"{json.dumps(item['current_state'])}\n\n"
        
        prompt += f"The current game UUID base is {item['current_state'].get('max_UUID', 0)}\n"
        if self.data_type in ["action", "score", "full"]:
            action = item.get('action') or item['current_state'].get('lastAction', '')
            prompt += f"The action to take is: {action}\n\n"
        
        prompt += "Let's solve this step by step and assign a confidence score:\n"
        prompt += "1. Analyze the current state\n"
        prompt += "2. Consider the action and rules\n"
        prompt += "3. Determine state changes\n"
        prompt += "4. Generate the final state\n"
        prompt += "5. Assign a confidence score (0-1)\n\n"

        if 'explanation' in item:
            prompt += f"\nExplanation of state changes:\n{item['explanation']}\n\n"
            
        prompt += "Let's think through this carefully:\n"
        return prompt