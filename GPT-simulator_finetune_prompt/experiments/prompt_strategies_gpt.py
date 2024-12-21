import json
from typing import Optional, Dict, Any

# Add base PromptStrategy class definition
class PromptStrategy:
    """Base class for prompt strategies"""
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
        raise NotImplementedError("Each strategy must implement get_prediction")

    def _build_prompt(self, item):
        raise NotImplementedError("Each strategy must implement _build_prompt")

# Update GPTPromptStrategy to inherit from local PromptStrategy
class GPTPromptStrategy(PromptStrategy):
    """Base GPT strategy that inherits from PromptStrategy"""
    def __init__(self, llm, object_desc, game_data, data_type="action", partial=False, no_rule=False):
        super().__init__(llm, object_desc, game_data, data_type, partial)
        self.model = game_data.get("model", "gpt-4o")
        self.no_rule = no_rule
        self.client = llm
        
        # Initialize few_shot_examples as a dictionary
        self.few_shot_examples = {}
        try:
            # Load examples from JSON file
            with open("data/few_shot_examples.json") as f:
                examples_data = json.load(f)
                # Ensure examples_data is a dictionary mapping game names to example lists
                if isinstance(examples_data, dict):
                    self.few_shot_examples = examples_data
                else:
                    print("Warning: few_shot_examples.json should contain a dictionary mapping game names to example lists")
        except Exception as e:
            print(f"Warning: Could not load few-shot examples: {str(e)}")

    def _get_base_instruction(self) -> str:
        return """You are a simulator of a text game. Read the task description of a 
text game. Given the current game state in JSON, you need to 
decide the new game state after taking an action.
Your response should be in the same JSON format as the given 
game state."""

    def _build_prompt(self, item: Dict[str, Any]) -> str:
        prompt = self._get_base_instruction() + "\n\n"
        
        if isinstance(self, GPTFewShotCoTStrategy):
            # Only keep example code for GPTFewShotCoTStrategy
            game_examples = self.few_shot_examples.get(item['game'], [])
            if game_examples:
                example = game_examples[0]  # Use first example
                prompt += """Here is an example:
Example game task description:
""" + example['current_state']['taskDesc'] + """

Here are the descriptions of all game objects properties in the 
example game:
""" + self.object_desc + """

Here are the descriptions of all game actions in the example game:
""" + self.game_data['action_desc'] + """

Here is the game state:
""" + json.dumps(example['current_state'], indent=2) + """

The action to take is """ + example['action_state']['lastAction'] + """

The expected response is:
""" + json.dumps(example['action_state'], indent=2) + """

"""
                if 'explanation' in example:
                    prompt += f"Explanation of state changes:\n{example['explanation']}\n\n"
                    
                prompt += "Here is the game that you need to simulate:\n"
        
        # Rest of the method remains the same...
        if "taskDesc" in item["current_state"]:
            prompt += f"Task Description:\n{item['current_state']['taskDesc']}\n\n"
            
        if not self.no_rule:
            prompt += f"Here are the descriptions of all game objects properties:\n{self.object_desc}\n\n"
            
            if self.data_type in ["action", "full"]:
                prompt += f"Here are the descriptions of all game actions:\n{self.game_data['action_desc']}\n\n"
                
            if self.data_type in ["score", "full"]:
                prompt += f"Here is a description of the game score function:\n{self.game_data['score_desc']}\n\n"
        
        prompt += "Here is the game state:\n"
        prompt += json.dumps(item['current_state'], indent=2) + "\n\n"
        
        prompt += f"The current game UUID base is {item['current_state'].get('max_UUID', 0)}\n"
        
        if self.data_type in ["action", "score", "full"]:
            # Get action from multiple possible sources with fallback chain
            action = ''
            
            # First try to get action directly from item
            if isinstance(item, dict):
                # Check each possible location in order of priority
                if 'action' in item:
                    action = item['action']
                elif 'action_state' in item and isinstance(item['action_state'], dict):
                    action = item['action_state'].get('lastAction', '')
                elif 'current_state' in item and isinstance(item['current_state'], dict):
                    action = item['current_state'].get('lastAction', '')
                elif 'tick_state' in item and isinstance(item['tick_state'], dict):
                    action = item['tick_state'].get('lastAction', '')

                # If still no action and current_state has game_state list
                if not action and 'current_state' in item:
                    current_state = item['current_state']
                    if isinstance(current_state, dict) and 'game_state' in current_state:
                        game_state = current_state['game_state']
                        if isinstance(game_state, list) and len(game_state) > 0:
                            first_state = game_state[0]
                            if isinstance(first_state, dict) and 'lastAction' in first_state:
                                action = first_state['lastAction']
            
            prompt += f"The action to take is: {action}\n\n"
        
        prompt += """Your response must be a valid JSON object that follows these rules:
1. Start with a curly brace {
2. End with a curly brace }
3. Contain only valid JSON
4. Not include any explanatory text before or after the JSON
5. Not include markdown code block markers

Generate the JSON now:"""
        
        return prompt

    def get_prediction(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            prompt = self._build_prompt(item)
            self.last_prompt = prompt
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise game state simulator that only outputs valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=4096
            )
            
            response_text = completion.choices[0].message.content
            self.last_response = response_text
            
            try:
                # Parse the raw response into JSON
                prediction = json.loads(response_text)
                
                # Handle partial responses if needed
                if self.partial and self.data_type != "score":
                    if self.data_type == "full":
                        has_score = True
                    else:
                        has_score = False
                    from experiments.quest_gpt import recover_game_state_from_partial
                    prediction = recover_game_state_from_partial(item['current_state'], prediction, has_score=has_score)
                
                # Handle different data types
                if self.data_type == "score":
                    # Ensure required score fields exist
                    if not all(k in prediction for k in ["score", "gameOver", "gameWon"]):
                        prediction.update({
                            "score": prediction.get("score", 0),
                            "gameOver": prediction.get("gameOver", False),
                            "gameWon": prediction.get("gameWon", False)
                        })
                else:
                    # For non-score predictions, ensure game_state format
                    if "game_state" not in prediction:
                        if isinstance(prediction, dict):
                            prediction = {"game_state": [prediction]}
                        else:
                            return None
                
                # Create output record with both game_state and lastAction
                output_record = prediction.copy()  # Copy the full prediction
                output_record['lastAction'] = item['action_state']['lastAction']  # Add lastAction from action_state
                
                return output_record
                
            except json.JSONDecodeError:
                return None
                
        except Exception as e:
            print(f"Error in API call: {str(e)}")
            return None

class GPTSimpleCoTStrategy(GPTPromptStrategy):
    """Simple Chain of Thought strategy that just adds 'Let's think step by step'"""
    def _build_prompt(self, item: Dict[str, Any]) -> str:
        prompt = super()._build_prompt(item)
        # Add step by step thinking prompt before the response rules
        prompt = prompt.replace("Generate the JSON now:", 
            "Let's think step by step about how the game state should change.\n\nGenerate the JSON now:")
        return prompt

class GPTZeroShotCoTStrategy(GPTPromptStrategy):
    """Zero-shot Chain of Thought that generates reasoning first, then extracts answer"""
    def get_prediction(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # First generate reasoning
        reasoning_prompt = self._build_prompt(item)
        reasoning_prompt += "\nLet's solve this step by step. Think through each step carefully:\n"
        reasoning = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a precise game state simulator that thinks step by step."},
                {"role": "user", "content": reasoning_prompt}
            ],
            temperature=0
        ).choices[0].message.content

        # Then use reasoning to generate final answer
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

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a precise game state simulator that only outputs valid JSON."},
                {"role": "user", "content": extraction_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0
        ).choices[0].message.content

        try:
            prediction = json.loads(response)
            return prediction
        except json.JSONDecodeError:
            return response

class GPTCoTSCStrategy(GPTZeroShotCoTStrategy):
    """Self-Consistency Chain of Thought that generates multiple answers and takes majority"""
    def __init__(self, llm, object_desc, game_data, data_type="action", partial=False, no_rule=False, num_samples=3):
        super().__init__(llm, object_desc, game_data, data_type, partial, no_rule)
        self.num_samples = num_samples

    def get_prediction(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        predictions = []
        for _ in range(self.num_samples):
            prediction = super().get_prediction(item)
            if isinstance(prediction, dict):
                predictions.append(prediction)

        if not predictions:
            return None

        # Return majority prediction
        from utils.experiment_utils import select_majority_prediction
        return select_majority_prediction(predictions)

class GPTFewShotCoTStrategy(GPTPromptStrategy):
    """Few-shot Chain of Thought that adds examples before generating reasoning"""
    def __init__(self, llm, object_desc, game_data, data_type="action", partial=False, no_rule=False, num_examples=3):
        super().__init__(llm, object_desc, game_data, data_type, partial, no_rule)
        self.num_examples = num_examples

    def _build_prompt(self, item: Dict[str, Any]) -> str:
        prompt = self._get_base_instruction() + "\n\n"
        
        # Safely get examples for the current game
        game_name = item.get('game', '')
        game_examples = self.few_shot_examples.get(game_name, [])
        
        if game_examples:
            # Use up to num_examples examples
            for i, example in enumerate(game_examples[:self.num_examples]):
                prompt += f"Example {i+1}:\n"
                prompt += f"Task Description:\n{example['current_state']['taskDesc']}\n\n"
                
                # Only show rules for first example to keep prompt concise
                if i == 0 and not self.no_rule:
                    prompt += f"Object properties:\n{self.object_desc}\n\n"
                    if self.data_type in ["action", "full"]:
                        prompt += f"Game actions:\n{self.game_data['action_desc']}\n\n"
                    if self.data_type in ["score", "full"]:
                        prompt += f"Score function:\n{self.game_data['score_desc']}\n\n"
                
                prompt += f"Initial state:\n{json.dumps(example['current_state'], indent=2)}\n\n"
                
                if self.data_type in ["action", "score", "full"]:
                    prompt += f"Action taken: {example['action_state']['lastAction']}\n\n"
                
                prompt += f"Resulting state:\n{json.dumps(example['action_state'], indent=2)}\n\n"
                
                if 'explanation' in example:
                    prompt += f"Explanation:\n{example['explanation']}\n\n"
                
                prompt += "-" * 50 + "\n\n"
        
        # Add current case
        prompt += "Now for the current case:\n\n"
        if "taskDesc" in item["current_state"]:
            prompt += f"Task Description:\n{item['current_state']['taskDesc']}\n\n"
            
        if not self.no_rule:
            prompt += f"Object properties:\n{self.object_desc}\n\n"
            if self.data_type in ["action", "full"]:
                prompt += f"Game actions:\n{self.game_data['action_desc']}\n\n"
            if self.data_type in ["score", "full"]:
                prompt += f"Score function:\n{self.game_data['score_desc']}\n\n"
        
        prompt += "Current state:\n"
        prompt += json.dumps(item['current_state'], indent=2) + "\n\n"
        
        prompt += f"Current game UUID base: {item['current_state'].get('max_UUID', 0)}\n"
        
        if self.data_type in ["action", "score", "full"]:
            # Get action using the parent class's robust action lookup
            action = self._get_action_from_item(item)
            prompt += f"Action to take: {action}\n\n"
        
        prompt += "Let's solve this step by step like in the examples above.\n\n"
        prompt += """Your response must be a valid JSON object that follows these rules:
1. Start with a curly brace {
2. End with a curly brace }
3. Contain only valid JSON
4. Not include any explanatory text before or after the JSON
5. Not include markdown code block markers

Generate the JSON now:"""
        
        return prompt

    def _get_action_from_item(self, item: Dict[str, Any]) -> str:
        """Helper method to get action from item using robust lookup logic"""
        action = ''
        if isinstance(item, dict):
            # Check each possible location in order of priority
            if 'action' in item:
                action = item['action']
            elif 'action_state' in item and isinstance(item['action_state'], dict):
                action = item['action_state'].get('lastAction', '')
            elif 'current_state' in item and isinstance(item['current_state'], dict):
                action = item['current_state'].get('lastAction', '')
            elif 'tick_state' in item and isinstance(item['tick_state'], dict):
                action = item['tick_state'].get('lastAction', '')

            # If still no action and current_state has game_state list
            if not action and 'current_state' in item:
                current_state = item['current_state']
                if isinstance(current_state, dict) and 'game_state' in current_state:
                    game_state = current_state['game_state']
                    if isinstance(game_state, list) and len(game_state) > 0:
                        first_state = game_state[0]
                        if isinstance(first_state, dict) and 'lastAction' in first_state:
                            action = first_state['lastAction']
        return action

class GPTBaselineStrategy(GPTPromptStrategy):
    """Baseline strategy without chain-of-thought or explanations"""
    def _build_prompt(self, item: Dict[str, Any]) -> str:
        prompt = self._get_base_instruction() + "\n\n"
        
        if "taskDesc" in item["current_state"]:
            prompt += f"Task Description:\n{item['current_state']['taskDesc']}\n\n"
            
        if not self.no_rule:
            prompt += f"Here are the descriptions of all game objects properties:\n{self.object_desc}\n\n"
            
            if self.data_type in ["action", "full"]:
                prompt += f"Here are the descriptions of all game actions:\n{self.game_data['action_desc']}\n\n"
                
            if self.data_type in ["score", "full"]:
                prompt += f"Here is a description of the game score function:\n{self.game_data['score_desc']}\n\n"
        
        prompt += "Here is the game state:\n"
        prompt += json.dumps(item['current_state'], indent=2) + "\n\n"
        
        prompt += f"The current game UUID base is {item['current_state'].get('max_UUID', 0)}\n"
        
        if self.data_type in ["action", "score", "full"]:
            action = self._get_action_from_item(item)
            prompt += f"The action to take is: {action}\n\n"
        
        prompt += """Your response must be a valid JSON object that follows these rules:
1. Start with a curly brace {
2. End with a curly brace }
3. Contain only valid JSON
4. Not include any explanatory text before or after the JSON
5. Not include markdown code block markers

Generate the JSON now:"""
        
        return prompt

    def _get_action_from_item(self, item: Dict[str, Any]) -> str:
        """Helper method to get action from item using robust lookup logic"""
        action = ''
        if isinstance(item, dict):
            # Check each possible location in order of priority
            if 'action' in item:
                action = item['action']
            elif 'action_state' in item and isinstance(item['action_state'], dict):
                action = item['action_state'].get('lastAction', '')
            elif 'current_state' in item and isinstance(item['current_state'], dict):
                action = item['current_state'].get('lastAction', '')
            elif 'tick_state' in item and isinstance(item['tick_state'], dict):
                action = item['tick_state'].get('lastAction', '')

            # If still no action and current_state has game_state list
            if not action and 'current_state' in item:
                current_state = item['current_state']
                if isinstance(current_state, dict) and 'game_state' in current_state:
                    game_state = current_state['game_state']
                    if isinstance(game_state, list) and len(game_state) > 0:
                        first_state = game_state[0]
                        if isinstance(first_state, dict) and 'lastAction' in first_state:
                            action = first_state['lastAction']
        return action