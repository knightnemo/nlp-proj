import json
import random

def sample_examples_from_json(examples_file='data/examples.json', num_examples=2):
    """Sample examples from examples.json"""
    with open(examples_file) as f:
        data = json.load(f)
    
    # Get examples from different types (action, tick)
    examples = []
    
    # Sample from action examples
    if 'action' in data['full']:
        action_example = {
            'current_state': data['full']['action']['current_state'],
            'action': data['full']['action']['lastAction'],
            'final_state': data['full']['action']['action_state'],
            'reasoning': """1. Current state analysis:
- Analyze current objects and their states
- Identify relevant properties

2. Action effects:
- Determine how the action changes objects
- Consider container relationships

3. Changes needed:
- Update object locations
- Modify relevant properties
- Keep other states unchanged"""
        }
        examples.append(action_example)
    
    # Sample from tick examples
    if 'tick' in data['full']:
        tick_example = {
            'current_state': data['full']['tick']['current_state'],
            'action': data['full']['tick']['lastAction'],
            'final_state': data['full']['tick']['tick_state'],
            'reasoning': """1. Current state analysis:
- Check device states and cycles
- Review object properties

2. Time effects:
- Process ongoing actions
- Update cycle stages

3. Changes needed:
- Update device states
- Modify object properties
- Apply time-based changes"""
        }
        examples.append(tick_example)
    
    return {"examples": examples}

def main():
    examples = sample_examples_from_json()
    
    with open('data/few_shot_examples.json', 'w') as f:
        json.dump(examples, f, indent=2)
    
    print("Created few-shot examples in data/few_shot_examples.json")

if __name__ == "__main__":
    main()