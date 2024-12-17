import re
import json

def extract_json_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 使用正则表达式匹配=== Model Response ===和====================之间的内容
    pattern = r"=== Model Response ===(.*?)===================="
    matches = re.findall(pattern, content, re.DOTALL)

    json_objects = []
    for match in matches:
        match = match.strip()  # 去除两端的空白字符
        
        # 查找其中的JSON部分（通常是以 `{` 开头，`}` 结尾的结构）
        json_pattern = r'\{.*\}'
        json_matches = re.findall(json_pattern, match)

        for json_str in json_matches:
            try:
                # 尝试解析JSON格式的字符串
                json_object = json.loads(json_str)
                json_objects.append(json_object)
            except json.JSONDecodeError as e:
                print(f"无法解析JSON: {e} 内容：{json_str[:100]}...")  # 打印部分内容帮助调试
                continue

    return json_objects

# 调用示例
file_path = 'llama3_70b.log'  # 请替换为你的文件路径
json_data = extract_json_from_file(file_path)

# 打印提取的JSON内容
for idx, data in enumerate(json_data, start=1):
    print(f"提取的JSON对象 {idx}:")
    print(json.dumps(data, indent=4, ensure_ascii=False))
