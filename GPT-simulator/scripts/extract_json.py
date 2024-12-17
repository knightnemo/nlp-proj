import re
import json

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
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = r"=== Model Response ===(.*?)===================="
    matches = re.findall(pattern, content, re.DOTALL)
    
    json_objects = []
    successful_parses = 0
    
    for i, match in enumerate(matches, 1):
        match = match.strip()
        
        # Extract JSON objects
        json_matches = extract_json_objects(match)
        
        if json_matches:
            try:
                json_str = json_matches[-1]  # Get the last JSON object
                cleaned_json = clean_json_string(json_str)
                
                # Debug output
                print(f"\n处理第 {i} 个响应块:")
                print(f"找到 {len(json_matches)} 个JSON结构，使用最后一个")
                print(f"JSON内容: {cleaned_json[:200]}...")
                
                json_object = json.loads(cleaned_json)
                successful_parses += 1
                json_objects.append((i, json_object))
                print(f"成功解析JSON! (序号 {i})")
                
            except json.JSONDecodeError as e:
                print(f"\nJSON解析错误 在响应块 {i}:")
                print(f"错误信息: {str(e)}")
                print(f"错误位置: {e.pos}")
                error_context = cleaned_json[max(0, e.pos-30):min(len(cleaned_json), e.pos+30)]
                print(f"错误上下文: ...{error_context}...")
                continue

    print(f"\n总共找到 {len(matches)} 个响应块")
    print(f"成功解析 {successful_parses} 个JSON对象")
    return json_objects

def save_results(json_data, output_file='output.txt'):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"总共解析出 {len(json_data)} 个JSON对象\n\n")
        for idx, data in json_data:  # Unpack tuple of (index, data)
            f.write(f"=== JSON对象 {idx} ===\n")
            f.write(json.dumps(data, indent=4, ensure_ascii=False))
            f.write('\n\n')

# 主执行代码
if __name__ == "__main__":
    file_path = 'llama3_70b.log'  # 替换为你的文件路径
    json_data = extract_json_from_file(file_path)
    save_results(json_data)
