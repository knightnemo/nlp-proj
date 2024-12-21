import json
import random

def sample_jsonl(input_file, output_file, num_samples):
    # Read JSONL file and parse each line as JSON
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    
    # Randomly sample entries
    sampled_data = random.sample(data, min(num_samples, len(data)))
    
    # Write to new file as JSON
    with open(output_file, 'w') as f:
        json.dump(sampled_data, f, indent=2)

if __name__ == "__main__":
    input_file = "train.jsonl"
    output_file = "few_shot_examples.json"
    num_samples = 20
    
    sample_jsonl(input_file, output_file, num_samples)