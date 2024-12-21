import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import json
import argparse
import random
import os
from datetime import datetime
from openai import OpenAI

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

class SingleBoilWaterEvaluator:
    def __init__(self, config: OpenAIConfig):
        self.llm = OpenAIInterface(config)

    def generate_prompt(self, actions: List[str]) -> str:
        """生成单个LLM的prompt"""
        prompt = ("You are a kitchen state simulator. Given the initial state and a sequence of actions, "
                 "predict the state after EACH action, including the initial state.\n\n"
                 
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
                 
                 "For each state transition, you must verify:\n"
                 "1. Is the object movement possible?\n"
                 "2. Are temperature changes correct?\n"
                 "3. Is water state consistent with temperature?\n"
                 "4. Do device states match their rules?\n\n"
                 
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

    def parse_state(self, state_str: str) -> Optional[KitchenState]:
        """解析状态字符串"""
        try:
            # 解析位置
            if "pot(table)" in state_str:
                location = "table"
            elif "pot(sink)" in state_str:
                location = "sink"
            elif "pot(stove)" in state_str:
                location = "stove"
            else:
                return None

            # 解析水的状态
            has_water = "no water" not in state_str
            if has_water:
                if "ice" in state_str:
                    water_state = "ice"
                elif "steam" in state_str:
                    water_state = "steam"
                else:
                    water_state = "liquid"
            else:
                water_state = "none"

            # 解析温度
            import re
            temp_match = re.search(r'(\d+)°C', state_str)
            water_temp = float(temp_match.group(1)) if temp_match else 20.0

            # 解析设备状态
            stove_on = "stove(on)" in state_str
            sink_on = "sink(on)" in state_str

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
            if "State" in line:
                state = self.parse_state(line)
                if state:
                    states.append(state)
        return states

    def calculate_accuracy(self, response: str) -> float:
        """计算预测准确率"""
        predicted_states = self.parse_trajectory(response)
        true_states = self.get_true_trajectory(self.actions)  # 需要保存actions
        
        if not predicted_states or len(predicted_states) != len(true_states):
            return 0.0
        
        correct_steps = sum(1 for pred, true in zip(predicted_states, true_states)
                          if self.states_match(pred, true))
        
        return correct_steps / len(true_states)

    def states_match(self, state1: KitchenState, state2: KitchenState) -> bool:
        """比较两个状态是否匹配"""
        if not state1 or not state2:
            return False
        return (state1.pot_location == state2.pot_location and
                state1.has_water == state2.has_water and
                state1.water_state == state2.water_state and
                abs(state1.water_temp - state2.water_temp) < 1.0 and
                state1.stove_on == state2.stove_on and
                state1.sink_on == state2.sink_on)

    def run_test(self, actions: List[str]) -> Dict:
        """运行测试并返回结果"""
        prompt = self.generate_prompt(actions)
        
        # 获取模型响应并输出
        response = self.llm.generate(prompt)
        print("\n模型响应:")
        print(response)  # 添加这行来输出模型的响应

        # 解析响应并计算准确率
        accuracy = self.calculate_accuracy(response)

        return {
            "prompt": prompt,
            "response": response,  # 添加模型响应
            "accuracy": accuracy
        }

    def get_true_trajectory(self, actions: List[str]) -> List[KitchenState]:
        """计算真实的状态轨迹"""
        trajectory = [self.initial_state]
        current_state = self.initial_state
        
        for action in actions:
            new_state = self.apply_action(current_state, action)
            trajectory.append(new_state)
            current_state = new_state
            
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--base_url", type=str, default="https://api.chatanywhere.tech/v1")
    parser.add_argument("--num_tests", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    # 配置
    config = OpenAIConfig(
        model=args.model,
        base_url=args.base_url
    )

    # 创建评估器
    evaluator = SingleBoilWaterEvaluator(config)

    # 测试用例
    test_cases = [
        ["put pot in sink", "turn on sink", "put pot on stove", "turn on stove"],
        ["put pot on stove", "turn on stove"],
        ["put pot in sink", "turn on sink", "turn off sink", "put pot on stove"],
        ["turn on stove", "put pot on stove"],
        ["put pot in sink", "put pot on stove", "turn on stove"],
    ]

    results = []
    total_correct = 0
    total_steps = 0

    print(f"开始测试 {args.num_tests} 组用例...")

    for i in range(args.num_tests):
        print(f"\n测试进度: {i+1}/{args.num_tests}")
        actions = random.choice(test_cases)
        print(f"动作序列: {actions}")
        
        try:
            result = evaluator.run_test(actions)
            results.append(result)
            
            # 累计正确数和总步数
            total_correct += result["accuracy"]
            total_steps += 1
            
        except Exception as e:
            print(f"Error in evaluation: {str(e)}")
            continue

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"boil_water_single_eval_results_{args.model}_{timestamp}.json"
    
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
                "avg_accuracy": total_correct / total_steps if total_steps > 0 else 0,
                "total_tests": args.num_tests
            }
        }, f, indent=4, default=str)
    
    print(f"\n结果已保存到: {result_path}")

if __name__ == "__main__":
    main() 