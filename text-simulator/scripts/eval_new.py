import argparse
import json
import os
import pandas as pd
import re
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--exp_type", type=str, required=True, choices=["action", "tick", "score", "full"])
    parser.add_argument("--output_folder", type=str, default="analysis")
    parser.add_argument("--state_data_folder", type=str, default="data")
    parser.add_argument("--test_data", type=str, default="test.jsonl")
    args = parser.parse_args()
    
    # 自动构建日志文件路径
    args.log_file = f"{args.prefix}.log"
    return args

def load_test_data(args):
    """加载测试数据作为对照"""
    test_data = {}
    with open(os.path.join(args.state_data_folder, args.test_data)) as f:
        for line in f:
            data = json.loads(line)
            game = data["game"]
            state_id = data["state_id"]
            if game not in test_data:
                test_data[game] = {}
            test_data[game][state_id] = data
    return test_data


def extract_json_from_response(response_text):
    """从模型响应中提取 JSON"""
    try:
        # 尝试直接解析整个响应
        return json.loads(response_text)
    except:
        # 如果失败,尝试提取```之间的内容
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                return None
    return None

def extract_game_state(text):
    """从文本中提取游戏状态JSON"""
    try:
        state_match = re.search(r'{\s*"game_state":\s*\[.*?\]}', text, re.DOTALL)
        if state_match:
            return json.loads(state_match.group(0))
    except:
        pass
    return None

def extract_action(text):
    """从文本中提取动作信息"""
    action_match = re.search(r'The action to take is:\s*"([^"]*)"', text)
    if action_match:
        return action_match.group(1)
    return None

def analyze_log_file(log_file):
    """分析日志文件中的每个交互"""
    interactions = []
    current_interaction = {}
    
    with open(log_file, 'r') as f:
        content = f.read()
        
    # 按照 "=== Model Response ===" 分割交互
    segments = content.split("=== Model Response ===")
    
    for i in range(len(segments)-1):  # 最后一个片段可能不完整
        prompt = segments[i]
        response = segments[i+1].split("=====================")[0].strip()
        
        # 提取当前状态
        current_state = extract_game_state(prompt)
        # 提取目标动作
        action = extract_action(prompt)
        # 提取模型响应
        model_response = extract_json_from_response(response)
        
        if current_state and model_response:
            interactions.append({
                "current_state": current_state,
                "action": action,
                "model_response": model_response,
                "raw_response": response
            })
            
    return interactions

def evaluate_response(interaction):
    """评估单个交互的正确性"""
    current_state = interaction["current_state"]
    model_response = interaction["model_response"]
    
    # 检查响应格式
    if not isinstance(model_response, dict):
        return -1, -1  # 格式错误
        
    # 检查必要的字段
    required_fields = ["modified", "removed", "score"]
    if not all(field in model_response for field in required_fields):
        return -1, -1
        
    # 计算对象属性错误
    objprop_errors = 0
    # TODO: 实现具体的对象属性评估逻辑
    
    # 计算分数错误
    score_errors = 0
    if "score" in model_response:
        gold_score = current_state["game_state"][-1]["score"]
        pred_score = model_response["score"].get("score", None)
        if pred_score != gold_score:
            score_errors += 1
            
    return objprop_errors, score_errors

def main():
    args = parse_args()
    
    # 加载测试数据
    test_data = load_test_data(args)
    
    # 分析日志文件
    interactions = analyze_log_file(args.log_file)
    
    statistics = defaultdict(lambda: {"total_states": 0, "total_errors": 0})
    
    for interaction in interactions:
        # 从交互中提取游戏和状态ID
        game_match = re.search(r'Processing\s+(\w+)_(\d+)', interaction["raw_response"])
        if game_match:
            game = game_match.group(1)
            state_id = int(game_match.group(2))
            
            # 获取对应的正确答案
            if game in test_data and state_id in test_data[game]:
                gold_data = test_data[game][state_id]
                objprop_errors, score_errors = evaluate_response(interaction, gold_data)
                
                statistics[game]["total_states"] += 1
                if objprop_errors > 0 or score_errors > 0:
                    statistics[game]["total_errors"] += 1

if __name__ == "__main__":
    main()