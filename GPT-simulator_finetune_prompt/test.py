import os
from dotenv import load_dotenv
from llama_utils.llama_interface import LlamaInterface, LlamaConfig
import json
from tqdm import tqdm
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load environment variables
load_dotenv()

def load_test_data(test_file="human_annotation/data/test.jsonl"):
    """Load test data from jsonl file"""
    data = []
    with open(test_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def main():
    # Get token from environment
    hf_token = os.getenv('HUGGING_FACE_HUB_TOKEN')
    if not hf_token:
        raise ValueError("Please set HUGGING_FACE_HUB_TOKEN environment variable")
    
    # Configure for Llama
    config = LlamaConfig(
        model_path="meta-llama/Llama-3.3-70B-Instruct",
        model_type="local",
        device="cuda",
        api_key=hf_token  # Pass token to config
    )
    
    # Initialize interface
    print("Initializing Llama model...")
    llama = LlamaInterface(config)
    
    # Load test data
    print("Loading test data...")
    test_data = load_test_data()
    
    # Run predictions
    results = []
    print("Running predictions...")
    for item in tqdm(test_data):
        prompt = f"""Given the current state:
{item['current_state']}

And the action:
{item['action']}

Predict the next state:"""
        
        try:
            response = llama.generate(
                prompt=prompt,
                max_tokens=1024,
                temperature=0.0
            )
            
            results.append({
                'id': item.get('id', ''),
                'current_state': item['current_state'],
                'action': item['action'],
                'predicted_state': response,
                'actual_state': item.get('next_state', ''),
                'success': True
            })
            
        except Exception as e:
            print(f"Error processing item: {e}")
            results.append({
                'id': item.get('id', ''),
                'error': str(e),
                'success': False
            })
    
    # Save results
    output_file = "llama_results.jsonl"
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

if __name__ == "__main__":
    main()