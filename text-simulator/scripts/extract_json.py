import re
import json
from collections import defaultdict
import difflib
from evaluate import evaluate
def recover_game_state_from_partial(curr_state, partial_change, has_score=False):
    recovered_state = {"game_state":[]}
    modified_uuids = {state['uuid']:state for state in partial_change["modified"]}
    if has_score:
        obj_states = curr_state["game_state"][:-1]
    else:
        obj_states = curr_state["game_state"]
    for state in obj_states:
        if state['uuid'] in partial_change["removed"]:
            continue
        if state['uuid'] in modified_uuids:
            recovered_state["game_state"].append(modified_uuids[state['uuid']])
        else:
            recovered_state["game_state"].append(state)

    if has_score:
        if len(partial_change['score']) > 0:
            recovered_state["game_state"].append(partial_change['score'])
        else:
            recovered_state["game_state"].append(curr_state["game_state"][-1])

    return recovered_state

def clean_json_string(json_str):
    # Clean and fix common JSON formatting issues
    json_str = json_str.strip()
    
    # Handle nested JSON structures better
    # First preserve escaped quotes
    json_str = re.sub(r'\\\"', '__ESCAPED_QUOTE__', json_str)
    
    # Fix single quotes to double quotes
    json_str = re.sub(r"(?<!\\)'", '"', json_str)
    
    # Fix unquoted property names
    json_str = re.sub(r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1 "\2":', json_str)
    
    # Restore escaped quotes
    json_str = json_str.replace('__ESCAPED_QUOTE__', '\\"')
    
    # Remove any control characters except newlines
    json_str = ''.join(char for char in json_str if ord(char) >= 32 or char == '\n')
    
    return json_str

def find_matching_bracket(s, start):
    """Find the matching closing bracket for an opening bracket."""
    count = 1
    i = start + 1
    while i < len(s):
        if s[i] == '{':
            count += 1
        elif s[i] == '}':
            count -= 1
            if count == 0:
                return i
        i += 1
    return -1

def extract_json_objects(text):
    """Extract all JSON objects from text, handling nested structures."""
    objects = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            end = find_matching_bracket(text, i)
            if end != -1:
                objects.append(text[i:end + 1])
                i = end
        i += 1
    return objects

def extract_json_from_file(file_path):
    """Extract JSON objects from file and return list"""
    json_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Find all JSON objects
    start = 0
    while True:
        start = content.find('{', start)
        if start == -1:
            break
            
        end = find_matching_bracket(content, start)
        if end == -1:
            break
            
        json_str = content[start:end + 1]
        try:
            json_obj = json.loads(json_str)
            json_list.append(json_obj)
        except json.JSONDecodeError:
            pass
            
        start = end + 1
        
    return json_list

def compare_with_target(extracted_jsons, target_file):
    """Compare extracted JSONs with target JSONL file using evaluation functions"""
    # Load target file
    target_jsons = []
    with open(target_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                target_jsons.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    
    total_lines = len(target_jsons)
    total_errors = 0
    total_score_errors = 0
    differences = []
    
    for i in range(total_lines):
        if i >= len(extracted_jsons):
            differences.append(f"Line {i+1}: Missing in extracted data")
            total_errors += 1
            continue
            
        # Recover game state
        prediction = extracted_jsons[i]
        data_target = target_jsons[i]
        data_state = data_target.get('current_state', {})
        data_action = data_target.get('lastAction', '')
        has_score = 'score' in data_target
        
        try:
            # Recover full game state
            prediction = recover_game_state_from_partial(data_state, prediction, has_score=has_score)
            
            # Evaluate the prediction
            num_errors, num_score_errors, eval_out_str = evaluate(
                prediction, 
                data_target, 
                data_action, 
                evaluate_score=True
            )
            
            total_errors += num_errors
            total_score_errors += num_score_errors
            
            if num_errors > 0 or num_score_errors > 0:
                differences.append(f"Line {i+1}: {eval_out_str}")
                
        except Exception as e:
            differences.append(f"Line {i+1}: Error during evaluation - {str(e)}")
            total_errors += 1
    
    accuracy = ((total_lines - total_errors) / total_lines * 100) if total_lines > 0 else 0
    score_accuracy = ((total_lines - total_score_errors) / total_lines * 100) if total_lines > 0 else 0
    
    return {
        'total_lines': total_lines,
        'total_errors': total_errors,
        'total_score_errors': total_score_errors,
        'accuracy': accuracy,
        'score_accuracy': score_accuracy,
        'differences': differences
    }

def save_comparison_results(results, output_file='comparison_results.txt'):
    """Save comparison results to a file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("JSON Extraction Validation Results\n")
        f.write("================================\n\n")
        f.write(f"Total lines analyzed: {results['total_lines']}\n")
        f.write(f"Total errors: {results['total_errors']}\n")
        f.write(f"Total score errors: {results['total_score_errors']}\n")
        f.write(f"State accuracy: {results['accuracy']:.2f}%\n")
        f.write(f"Score accuracy: {results['score_accuracy']:.2f}%\n\n")
        
        if results['differences']:
            f.write("Detailed Differences:\n")
            f.write("-------------------\n")
            for diff in results['differences']:
                f.write(f"{diff}\n")

# 主执行代码
if __name__ == "__main__":
    source_file = 'llama3_70b.log'  # 源文件
    target_file = 'data/test.jsonl'  # 目标文件
    
    # 提取JSON对象
    extracted_jsons = extract_json_from_file(source_file)
    
    # 与目标文件比对
    results = compare_with_target(extracted_jsons, target_file)
    save_comparison_results(results)
    
    print(f"提取和评估完成！")
    print(f"状态准确率: {results['accuracy']:.2f}%")
    print(f"分数准确率: {results['score_accuracy']:.2f}%")
    print(f"总行数: {results['total_lines']}")
    print(f"状态错误数: {results['total_errors']}")
    print(f"分数错误数: {results['total_score_errors']}")
