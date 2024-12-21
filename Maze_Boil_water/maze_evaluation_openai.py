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
    api_key: str = "sk-OndcygfG5hh4mmoSFzQKefFLrnPKxyI653aYKYC8QzK3Y8ZR"
    
    base_url: str = "https://api.chatanywhere.tech/v1"
    model: str = "gpt-4o"
    temperature: float = 1.5
    max_tokens: int = 512

class OpenAIInterface:
    def __init__(self, config: OpenAIConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
    
    def generate(self, prompt: str, stream: bool = False) -> str:
        """生成回答，支持流式和非流式两种方式"""
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

@dataclass
class MazeConfig:
    """迷宫配置"""
    width: int
    height: int
    walls: List[Tuple[int, int]]
    start_pos: Tuple[int, int]

class MazeEnvironment:
    """迷宫环境"""
    def __init__(self, config: MazeConfig):
        self.width = config.width
        self.height = config.height
        self.walls = set(config.walls)
        self.start_pos = config.start_pos
        self.current_pos = self.start_pos
        
        self.action_map = {
            "UP": (0, -1),
            "DOWN": (0, 1),
            "LEFT": (-1, 0),
            "RIGHT": (1, 0)
        }
    
    def reset(self) -> Tuple[int, int]:
        """重置环境"""
        self.current_pos = self.start_pos
        return self.current_pos
    
    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """检查位置是否有效"""
        x, y = pos
        return (0 <= x < self.width and 
                0 <= y < self.height and 
                pos not in self.walls)
    
    def execute_single_action(self, action: str) -> Tuple[Tuple[int, int], bool]:
        """执行单个动作"""
        if action not in self.action_map:
            raise ValueError(f"无效动作: {action}")
            
        dx, dy = self.action_map[action]
        new_x = self.current_pos[0] + dx
        new_y = self.current_pos[1] + dy
        new_pos = (new_x, new_y)
        
        if not self.is_valid_position(new_pos):
            return self.current_pos, True
            
        self.current_pos = new_pos
        return new_pos, False

class TestCaseGenerator:
    """测试用例生成器"""
    def __init__(self, maze_config: MazeConfig):
        self.maze_config = maze_config
        self.actions = ["UP", "DOWN", "LEFT", "RIGHT"]
    
    def generate_random_sequence(self, min_length: int = 2, max_length: int = 5) -> List[str]:
        """生成随机动作序列"""
        length = random.randint(min_length, max_length)
        return random.choices(self.actions, k=length)
    
    def generate_test_cases(self, num_cases: int) -> List[List[str]]:
        """生成测试用例"""
        test_cases = []
        for _ in range(num_cases):
            test_cases.append(self.generate_random_sequence())
        return test_cases

class MazeLLMEvaluator:
    """迷宫LLM评估器"""
    def __init__(self, maze_config: MazeConfig, llm: OpenAIInterface):
        self.env = MazeEnvironment(maze_config)
        self.llm = llm
    
    def get_true_trajectory(self, actions: List[str]) -> List[Tuple[int, int]]:
        """获取真实轨迹"""
        self.env.reset()
        trajectory = [self.env.current_pos]
        
        for action in actions:
            new_pos, hit_wall = self.env.execute_single_action(action)
            trajectory.append(new_pos)
        
        return trajectory
    
    def generate_prompt(self, actions: List[str]) -> str:
        prompt = ("You are a maze position predictor. Given a maze configuration and a sequence of actions, "
                 "predict the position after EACH action, including the starting position.\n\n")
        
        prompt += "Maze Configuration:\n"
        prompt += f"- Size: {self.env.width}x{self.env.height} grid\n"
        prompt += f"- Starting position: {self.env.start_pos}\n"
        prompt += f"- Wall positions: {list(self.env.walls)}\n\n"
        
        prompt += f"Action sequence to execute: {actions}\n\n"
        
        prompt += ( "Movement Rules:\n"
                 "- RIGHT: Move (x+1, y)\n"
                 "- LEFT: Move (x-1, y)\n"
                 "- UP: Move (x, y+1)\n"
                 "- DOWN: Move (x, y-1)\n"
                 "- IMPORTANT: Before each move, you must verify:\n"
                 "  1. The new position is within grid bounds (0 to 4)\n"
                 "  2. The new position does not contain a wall\n"
                 "- If the new position is invalid (out of bounds or hits a wall):\n"
                 "  * Stay at the current position\n"
                 "  * Continue with the next action\n\n")
        
        prompt += "COMMAND: Output a sequence of positions, one per line, including the start position.\n"
        prompt += "Each position must be in the format: (x,y)\n"
        prompt += "Example for actions [RIGHT, DOWN]:\n"
        prompt += "(0,0)\n(1,0)\n(1,1)\n\n"
        prompt += "after predict one position please do not forget the role of this maze :"
        prompt += "Movement Rules:\n"
        prompt += " - RIGHT: Move (x+1, y)\n"
        prompt += "- LEFT: Move (x-1, y)\n"
        prompt += "- UP: Move (x, y+1)\n"
        prompt += "- DOWN: Move (x, y-1)\n "
        prompt += "REMEMBER that position is not valid if it is out of bounds (x and y can not be less than 0 or greater than 4) or hits a wall!!\n\n"
        prompt += "here is the action sequence you need to predict"
        prompt += f"Action sequence to execute: {actions}\n\n"
        prompt += " Your response (one position per line),just output the predicted trajectory do not include any other text like why this is correct or anything else!!"
        return prompt

    def parse_response(self, response: str) -> List[Optional[Tuple[int, int]]]:
        lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
        
        positions = []
        for line in lines:
            if not line.startswith('('):
                continue
                
            try:
                coords = line.strip('()').split(',')
                if len(coords) == 2:
                    x, y = map(int, coords)
                    if 0 <= x < self.env.width and 0 <= y < self.env.height:
                        positions.append((x, y))
                    else:
                        positions.append(None)
            except:
                continue
        
        return positions

    def evaluate_prediction(self, actions: List[str]) -> Dict:
        """评估预测"""
        prompt = self.generate_prompt(actions)
        response = self.llm.generate(prompt, stream=True)
        
        predicted_positions = self.parse_response(response)
        true_trajectory = self.get_true_trajectory(actions)
        
        if not predicted_positions:
            return {
                "actions": actions,
                "prompt": prompt,
                "response": response,
                "predicted_trajectory": [],
                "true_trajectory": [list(pos) for pos in true_trajectory],
                "correct": False,
                "error_message": "Failed to parse model response",
                "correct_steps": 0,
                "total_steps": len(true_trajectory),
                "accuracy": 0.0
            }
        
        # 调整预测序列长度
        if len(predicted_positions) < len(true_trajectory):
            predicted_positions.extend([None] * (len(true_trajectory) - len(predicted_positions)))
        elif len(predicted_positions) > len(true_trajectory):
            predicted_positions = predicted_positions[:len(true_trajectory)]
        
        # 检查每个位置是否正确
        correct_steps = sum(1 for pred_pos, true_pos in zip(predicted_positions, true_trajectory) 
                           if pred_pos == true_pos)
        
        accuracy = correct_steps / len(true_trajectory)
        
        return {
            "actions": actions,
            "prompt": prompt,
            "response": response,
            "predicted_trajectory": [list(pos) if pos else None for pos in predicted_positions],
            "true_trajectory": [list(pos) for pos in true_trajectory],
            "correct_steps": correct_steps,
            "total_steps": len(true_trajectory),
            "accuracy": accuracy
        }

def parse_args():
    parser = argparse.ArgumentParser(description='Maze Evaluation with OpenAI')
    parser.add_argument("--base_url", type=str, 
                    default="https://api.chatanywhere.tech/v1",
                    help="API base URL")
    parser.add_argument("--model", type=str, 
                    default="gpt-4o",
                    help="OpenAI model to use")
    parser.add_argument("--maze_width", type=int, default=5,
                    help="Maze width")
    parser.add_argument("--maze_height", type=int, default=5,
                    help="Maze height")
    parser.add_argument("--num_tests", type=int, default=40,
                    help="Number of test cases")
    parser.add_argument("--output_dir", type=str, default="results",
                    help="Output directory")
    parser.add_argument("--random_seed", type=int, default=42,
                    help="Random seed")
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.random_seed)
    
    # 使用默认配置初始化OpenAI
    openai_config = OpenAIConfig()
    llm = OpenAIInterface(openai_config)
    
    # 创建迷宫配置
    maze_config = MazeConfig(
        width=args.maze_width,
        height=args.maze_height,
        walls=[(1, 1), (2, 2), (3, 3)],  # 示例墙壁
        start_pos=(0, 0)
    )
    
    # 创建测试用例生成器
    test_generator = TestCaseGenerator(maze_config)
    test_cases = test_generator.generate_test_cases(args.num_tests)
    
    # 创建评估器
    evaluator = MazeLLMEvaluator(maze_config, llm)
    
    # 运行评估
    results = []
    total_correct_steps = 0
    total_steps = 0
    
    print(f"\n开始评估 {args.num_tests} 组测试...")
    
    for i, actions in enumerate(test_cases, 1):
        try:
            result = evaluator.evaluate_prediction(actions)
            results.append(result)
            
            # 添加错误处理
            correct_steps = result.get("correct_steps", 0)
            total_steps_current = result.get("total_steps", 0)
            
            total_correct_steps += correct_steps
            total_steps += total_steps_current
            
            print(f"\n评估进度: {i}/{args.num_tests}")
            print(f"动作序列: {actions}")
            print(f"预测准确率: {result.get('accuracy', 0):.2%}")
            if total_steps > 0:
                print(f"总体准确率: {(total_correct_steps/total_steps):.2%}")
        except Exception as e:
            print(f"Error in evaluation: {str(e)}")
            continue
    
    # 计算总体评估指标
    metrics = {
        "total_tests": args.num_tests,
        "total_correct_steps": total_correct_steps,
        "total_steps": total_steps,
        "overall_accuracy": total_correct_steps / total_steps if total_steps > 0 else 0,
        "best_accuracy": max(r["accuracy"] for r in results),
        "worst_accuracy": min(r["accuracy"] for r in results)
    }
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"maze_eval_results_openai_{args.model}_{timestamp}.json"
    
    result_path = os.path.join(args.output_dir, result_filename)
    with open(result_path, "w") as f:
        json.dump({
            "model_info": {
                "name": args.model,
                "base_url": args.base_url,
                "temperature": openai_config.temperature,
                "max_tokens": openai_config.max_tokens
            },
            "maze_config": {
                "width": maze_config.width,
                "height": maze_config.height,
                "walls": list(maze_config.walls),
                "start_pos": maze_config.start_pos
            },
            "results": results,
            "metrics": metrics
        }, f, indent=4, default=str)
    
    print(f"\n结果已保存到: {result_path}")
    
    # 打印总体评估结果
    print("\n=== 评估总结 ===")
    print(f"模型: {args.model}")
    print(f"总测试数: {metrics['total_tests']}")
    print(f"总步数: {metrics['total_steps']}")
    print(f"正确步数: {metrics['total_correct_steps']}")
    print(f"总体准确率: {metrics['overall_accuracy']:.2%}")
    print(f"最佳准确率: {metrics['best_accuracy']:.2%}")
    print(f"最差准确率: {metrics['worst_accuracy']:.2%}")

if __name__ == "__main__":
    main()