import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import json
import argparse
import random
import os
from datetime import datetime
from openai import OpenAI
import copy

@dataclass
class OpenAIConfig:
    """OpenAI配置"""
    api_key: str = ""
    base_url: str = "https://api.chatanywhere.tech/v1"
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 1024

@dataclass
class KitchenState:
    """厨房状态"""
    pot_location: str  # "table", "sink", "stove"
    has_water: bool
    water_state: str  # "ice", "liquid", "steam"
    water_temp: float
    stove_on: bool
    sink_on: bool

class OpenAIInterface:
    def __init__(self, config: OpenAIConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
    
    def generate(self, prompt: str, stream: bool = False) -> str:
        """生成回答"""
        messages = [{'role': 'user', 'content': prompt}]
        
        try:
            if stream:
                response_text = ""
                stream = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    stream=True
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        response_text += chunk.choices[0].delta.content
                        print(chunk.choices[0].delta.content, end="")
                return response_text
            else:
                completion = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                return completion.choices[0].message.content
                
        except Exception as e:
            print(f"Error in OpenAI API call: {str(e)}")
            return ""

class BoilWaterEvaluator:
    def __init__(self, predictor_config: OpenAIConfig, judge_config: OpenAIConfig):
        self.predictor = OpenAIInterface(predictor_config)
        self.judge = OpenAIInterface(judge_config)
        self.current_actions = None
        self.initial_state = KitchenState(
            pot_location="table",
            has_water=False,
            water_state="none",
            water_temp=20.0,
            stove_on=False,
            sink_on=False
        )

    def generate_predictor_prompt(self, actions: List[str]) -> str:
        """生成预测器的 prompt"""
        prompt = ("You are a kitchen state predictor. Given the initial state and a sequence of actions, "
                 "predict the state after EACH action, including the initial state.You just need to follow the rules and you don't need to verify whether the answer is right.\n\n"
                 
                 "Kitchen Configuration:\n"
                 "- Objects: Stove, Pot, Water, Sink\n"
                 "- Initial state:\n"
                 "  * Empty pot on table\n"
                 "  * Stove is off\n"
                 "  * Sink is off\n"
                 "  * Room temperature: 20°C\n\n"
                 
                 f"Action sequence to execute: {actions}\n\n"
                 
                 "State Rules:\n"
                 "1. Water states:\n"
                 "   - Below 0°C: ice\n"
                 "   - 0-100°C: liquid water\n"
                 "   - Above 100°C: steam\n"
                 "2. Temperature changes:\n"
                 "   - Stove increases temperature by 25°C per step when on\n"
                 "   - Maximum stove temperature: 500°C\n"
                 "3. Container rules:\n"
                 "   - Pot can hold water\n"
                 "   - Sink adds water when on\n\n"
                 
                 "COMMAND: Output the state after each action, including initial state.\n"
                 "For each state, specify:\n"
                 "1. Location of pot\n"
                 "2. Water presence and state\n"
                 "3. Water temperature\n"
                 "4. Device states (on/off)\n\n"
                 
                 "Example format:\n"
                 "State 0: pot(table), no water, stove(off), sink(off)\n"
                 "State 1: pot(sink), water(liquid,20°C), stove(off), sink(on)\n")
        
        return prompt

    def generate_judge_prompt(self, actions: List[str], predictor_responses: List[str]) -> str:
        """生成判断器的 prompt"""
        prompt = ("You are a kitchen state judge. Given multiple predictions for the same action sequence, "
                 "analyze them and output the correct state sequence.\n\n"
                 
                 "Initial state:\n"
                 "- Empty pot on table\n"
                 "- Stove is off\n"
                 "- Sink is off\n"
                 "- Room temperature: 20°C\n\n"
                 
                 f"Action sequence: {', '.join(actions)}\n\n"
                 
                 "Physical rules:\n"
                 "1. Water states: ice (<0°C), liquid (0-100°C), steam (>100°C)\n"
                 "2. Temperature changes: +25°C per step when stove is on\n"
                 "3. Device rules:\n"
                 "   - Sink adds water when on\n"
                 "   - Stove heats only when pot is on it\n\n"
                 
                 "Predictions provided:\n")
        
        # 添加所有预测
        for i, pred in enumerate(predictor_responses, 1):
            prompt += f"\nPrediction {i}:\n{pred}\n"
        
        prompt += ("\nCOMMAND: Output the state sequence using exactly this format:\n"
                  "State 0: pot location: table, water: no water, temperature: N/A, stove: off, sink: off\n"
                  "State 1: pot location: sink, water: liquid 20°C, stove: off, sink: on\n"
                  "...\n\n"
                  "Output ONLY the state sequence, no explanations.\n")
        
        return prompt

    def parse_state(self, state_str: str) -> Optional[KitchenState]:
        """解析状态字符串"""
        try:
            # 解析位置
            if "pot location: table" in state_str.lower():
                location = "table"
            elif "pot location: sink" in state_str.lower():
                location = "sink"
            elif "pot location: stove" in state_str.lower():
                location = "stove"
            else:
                return None

            # 解析水的状态
            has_water = "no water" not in state_str.lower()
            if has_water:
                if "ice" in state_str.lower():
                    water_state = "ice"
                elif "steam" in state_str.lower():
                    water_state = "steam"
                else:
                    water_state = "liquid"
            else:
                water_state = "none"

            # 解析温度
            import re
            temp_match = re.search(r'(\d+)°C', state_str)
            water_temp = float(temp_match.group(1)) if temp_match and has_water else 20.0

            # 解析设备状态
            stove_on = "stove: on" in state_str.lower()
            sink_on = "sink: on" in state_str.lower()

            return KitchenState(
                pot_location=location,
                has_water=has_water,
                water_state=water_state,
                water_temp=water_temp,
                stove_on=stove_on,
                sink_on=sink_on
            )
        except Exception as e:
            print(f"Error parsing state: {e}")
            return None

    def parse_trajectory(self, response: str) -> List[KitchenState]:
        """解析完整的状态轨迹"""
        states = []
        for line in response.strip().split('\n'):
            if line.startswith('State'):
                try:
                    # 使用更严格的格式解析
                    parts = line.split(', ')
                    location = parts[0].split(': ')[2]  # "State X: pot location: table" -> "table"
                    
                    water_part = parts[1].split(': ')[1]  # "water: no water" or "water: liquid 20°C"
                    has_water = "no water" not in water_part
                    water_temp = 20.0
                    water_state = "none"
                    
                    if has_water:
                        if "liquid" in water_part:
                            water_state = "liquid"
                            temp = float(water_part.split()[1].replace('°C', ''))
                            water_temp = temp
                        elif "ice" in water_part:
                            water_state = "ice"
                        elif "steam" in water_part:
                            water_state = "steam"
                    
                    stove_on = "on" in parts[3].split(': ')[1]  # "stove: on/off"
                    sink_on = "on" in parts[4].split(': ')[1]   # "sink: on/off"
                    
                    state = KitchenState(
                        pot_location=location,
                        has_water=has_water,
                        water_state=water_state,
                        water_temp=water_temp,
                        stove_on=stove_on,
                        sink_on=sink_on
                    )
                    states.append(state)
                except Exception as e:
                    print(f"Error parsing state line: {line}\nError: {e}")
                    continue
        
        return states

    def calculate_accuracy(self, predictions: List[List[KitchenState]], 
                         judge_prediction: List[KitchenState], 
                         true_trajectory: List[KitchenState]) -> Dict:
        """计算预测准确率"""
        def states_match(s1: KitchenState, s2: KitchenState) -> bool:
            return (s1.pot_location == s2.pot_location and
                   s1.has_water == s2.has_water and
                   s1.water_state == s2.water_state and
                   abs(s1.water_temp - s2.water_temp) < 1.0 and
                   s1.stove_on == s2.stove_on and
                   s1.sink_on == s2.sink_on)

        predictor_step_correct = 0
        predictor_total_steps = 0
        
        for pred in predictions:
            for pred_state, true_state in zip(pred, true_trajectory):
                if states_match(pred_state, true_state):
                    predictor_step_correct += 1
                predictor_total_steps += 1
        
        judge_step_correct = 0
        judge_total_steps = len(true_trajectory)
        
        for judge_state, true_state in zip(judge_prediction, true_trajectory):
            if states_match(judge_state, true_state):
                judge_step_correct += 1
        
        return {
            "predictor_accuracy": predictor_step_correct / predictor_total_steps if predictor_total_steps > 0 else 0,
            "judge_accuracy": judge_step_correct / judge_total_steps if judge_total_steps > 0 else 0
        }

    def parse_judge_selection(self, judge_response: str) -> int:
        """解析判别器选择的预测编号"""
        try:
            # 查找 "Best prediction: X" 格式的文本
            import re
            match = re.search(r'Best prediction:\s*(\d+)', judge_response)
            if match:
                return int(match.group(1))
        except Exception as e:
            print(f"Error parsing judge selection: {e}")
        return 0  # 如果解析失败返回0

    def calculate_predictor_accuracy(self, predictor_responses: List[str]) -> float:
        """计算所有预测的平均准确率"""
        if not predictor_responses:
            return 0.0
        
        true_trajectory = self.get_true_trajectory(self.current_actions)
        total_correct = 0
        total_steps = 0
        
        for response in predictor_responses:
            predicted_states = self.parse_trajectory(response)
            if not predicted_states:
                continue
            
            # 确保预测和真实轨迹长度匹配
            if len(predicted_states) != len(true_trajectory):
                print(f"Length mismatch: predicted={len(predicted_states)}, true={len(true_trajectory)}")
                continue
            
            for pred_state, true_state in zip(predicted_states, true_trajectory):
                if self.states_match(pred_state, true_state):
                    total_correct += 1
                total_steps += 1
            
            # 打印详细的匹配信息
            print("\n预测状态匹配详情:")
            for i, (pred, true) in enumerate(zip(predicted_states, true_trajectory)):
                match = self.states_match(pred, true)
                print(f"Step {i}: {'✓' if match else '✗'}")
                if not match:
                    print(f"Predicted: {pred}")
                    print(f"True: {true}")
        
        return total_correct / total_steps if total_steps > 0 else 0.0

    def calculate_judge_accuracy(self, selected_prediction: List[KitchenState]) -> float:
        """计算判别器选择的预测的准确率"""
        if not selected_prediction:
            return 0.0
            
        true_trajectory = self.get_true_trajectory(self.current_actions)
        if len(selected_prediction) != len(true_trajectory):
            return 0.0
            
        correct_steps = sum(1 for pred, true in zip(selected_prediction, true_trajectory)
                          if self.states_match(pred, true))
        
        return correct_steps / len(true_trajectory)

    def run_test(self, actions: List[str], num_samples: int = 3) -> Dict:
        """运行测试并返回结果"""
        self.current_actions = actions
        predictor_prompt = self.generate_predictor_prompt(actions)
        
        predictor_responses = []
        predictions = []
        for _ in range(num_samples):
            response = self.predictor.generate(predictor_prompt)
            if response:
                predictor_responses.append(response)
                predicted_states = self.parse_trajectory(response)
                if predicted_states:
                    predictions.append(predicted_states)
                print("\n预测器响应:")
                print(response)
        
        if not predictor_responses:
            return self._empty_result(actions)
        
        # 修改这里，传入完整的预测响应列表
        judge_prompt = self.generate_judge_prompt(actions, predictor_responses)
        judge_response = self.judge.generate(judge_prompt)
        print("\n判别器响应:")
        print(judge_response)
        
        # 直接解析判别器的状态序列
        judge_states = self.parse_trajectory(judge_response)
        
        predictor_accuracy = self.calculate_predictor_accuracy(predictor_responses)
        judge_accuracy = self.calculate_judge_accuracy(judge_states)
        
        return {
            "actions": actions,
            "predictor_responses": predictor_responses,
            "predictions": [[str(state) for state in pred] for pred in predictions],
            "judge_response": judge_response,
            "judge_prediction": [str(state) for state in judge_states],
            "true_trajectory": [str(state) for state in self.get_true_trajectory(actions)],
            "accuracy_stats": {
                "predictor_accuracy": predictor_accuracy,
                "judge_accuracy": judge_accuracy
            }
        }

    def get_true_trajectory(self, actions: List[str]) -> List[KitchenState]:
        """获取真实轨迹"""
        trajectory = [self.initial_state]
        current_state = copy.deepcopy(self.initial_state)
        
        for action in actions:
            new_state = self.apply_action(current_state, action)
            trajectory.append(new_state)
            current_state = copy.deepcopy(new_state)
            
        return trajectory

    def apply_action(self, state: KitchenState, action: str) -> KitchenState:
        """应用动作并返回新状态"""
        new_state = KitchenState(
            pot_location=state.pot_location,
            has_water=state.has_water,
            water_state=state.water_state,
            water_temp=state.water_temp,
            stove_on=state.stove_on,
            sink_on=state.sink_on
        )
        
        if action == "put pot in sink":
            new_state.pot_location = "sink"
        elif action == "put pot on stove":
            new_state.pot_location = "stove"
        elif action == "turn on sink":
            new_state.sink_on = True
            if new_state.pot_location == "sink":
                new_state.has_water = True
                new_state.water_state = "liquid"
                new_state.water_temp = 20.0
        elif action == "turn off sink":
            new_state.sink_on = False
        elif action == "turn on stove":
            new_state.stove_on = True
        elif action == "turn off stove":
            new_state.stove_on = False
        
        # 更新温度和状态
        if new_state.has_water and new_state.pot_location == "stove" and new_state.stove_on:
            new_state.water_temp = min(new_state.water_temp + 25.0, 500.0)
            if new_state.water_temp > 100.0:
                new_state.water_state = "steam"
            elif new_state.water_temp < 0.0:
                new_state.water_state = "ice"
            else:
                new_state.water_state = "liquid"
        
        return new_state

    def states_match(self, state1: KitchenState, state2: KitchenState) -> bool:
        """比较两个状态是否匹配"""
        if not state1 or not state2:
            return False
        
        # 位置匹配
        if state1.pot_location != state2.pot_location:
            return False
        
        # 水的存在状态匹配
        if state1.has_water != state2.has_water:
            return False
        
        # 如果都有水，检查水的状态和温度
        if state1.has_water and state2.has_water:
            if state1.water_state != state2.water_state:
                return False
            # 温度允许有1度的误差
            if abs(state1.water_temp - state2.water_temp) > 1.0:
                return False
            
        # 设备状态匹配
        if state1.stove_on != state2.stove_on or state1.sink_on != state2.sink_on:
            return False
        
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--base_url", type=str, default="https://api.chatanywhere.tech/v1")
    parser.add_argument("--num_tests", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    # 配置
    config = OpenAIConfig(
        model=args.model,
        base_url=args.base_url
    )

    # 创建评估器
    evaluator = BoilWaterEvaluator(config, config)

    # 测试用例
    test_cases = [
        ["put pot in sink", "turn on sink", "put pot on stove", "turn on stove"],
        ["put pot on stove", "turn on stove"],
        ["put pot in sink", "turn on sink", "turn off sink", "put pot on stove"],
        ["turn on stove", "put pot on stove"],
        ["put pot in sink", "put pot on stove", "turn on stove"],
    ]

    results = []
    total_predictor_correct = 0
    total_judge_correct = 0
    total_steps = 0

    print(f"开始测试 {args.num_tests} 组用例...")

    for i in range(args.num_tests):
        print(f"\n测试进度: {i+1}/{args.num_tests}")
        actions = random.choice(test_cases)
        print(f"动作序列: {actions}")
        
        try:
            result = evaluator.run_test(actions, num_samples=args.num_samples)
            results.append(result)
            
            # 累计正确数和总步数
            accuracy = result["accuracy_stats"]
            total_predictor_correct += accuracy["predictor_accuracy"]
            total_judge_correct += accuracy["judge_accuracy"]
            total_steps += 1
            
        except Exception as e:
            print(f"Error in evaluation: {str(e)}")
            continue

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"boil_water_eval_results_{args.model}_{timestamp}.json"
    
    result_path = os.path.join(args.output_dir, result_filename)
    with open(result_path, "w") as f:
        json.dump({
            "model_info": {
                "name": args.model,
                "base_url": args.base_url,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens
            },
            "results": results,
            "metrics": {
                "avg_predictor_accuracy": total_predictor_correct / total_steps if total_steps > 0 else 0,
                "avg_judge_accuracy": total_judge_correct / total_steps if total_steps > 0 else 0,
                "total_tests": args.num_tests
            }
        }, f, indent=4, default=str)
    
    print(f"\n结果已保存到: {result_path}")

if __name__ == "__main__":
    main()