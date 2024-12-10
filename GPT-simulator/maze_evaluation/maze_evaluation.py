import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import os
from together import Together

@dataclass
class LlamaConfig:
    """LLaMA模型配置"""
    model_path: str
    model_type: str = "local"
    api_key: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class LlamaInterface:
    """LLaMA接口类"""
    def __init__(self, config: LlamaConfig):
        self.config = config
        
        if config.model_type == "local":
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_path
            ).to(config.device)
        elif config.model_type == "together":
            Together.api_key = config.api_key
        elif config.model_type == "anyscale":
            # 如果需要添加其他API支持
            pass
            
    def generate(self, prompt: str, stream: bool = True, response_format: dict = None) -> str:
        if self.config.model_type == "local":
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response[len(prompt):]
        elif self.config.model_type == "together":
            response = Together.complete(
                prompt=prompt,
                model=self.config.model_path,
                max_tokens=512,
                temperature=0.7,
                stream=stream
            )
            if stream:
                full_response = ""
                for chunk in response:
                    if chunk.output:
                        full_response += chunk.output
                return full_response
            return response.output
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
        self.current_pos = self.start_pos
        return self.current_pos
    
    def execute_actions(self, actions: List[str]) -> Tuple[Tuple[int, int], bool]:
        """执行一系列动作并返回最终位置"""
        self.reset()
        
        for action in actions:
            if action not in self.action_map:
                raise ValueError(f"无效动作: {action}")
                
            dx, dy = self.action_map[action]
            new_x = self.current_pos[0] + dx
            new_y = self.current_pos[1] + dy
            new_pos = (new_x, new_y)
            
            # 检查是否撞墙
            if (new_x < 0 or new_x >= self.width or 
                new_y < 0 or new_y >= self.height or
                new_pos in self.walls):
                return self.current_pos, True
                
            self.current_pos = new_pos
            
        return self.current_pos, False

    def get_state_description(self) -> str:
        """生成当前状态描述"""
        desc = f"在一个 {self.width}x{self.height} 的迷宫中:\n"
        desc += f"1. 起始位置在 {self.start_pos}\n"
        desc += f"2. 墙壁位置：{list(self.walls)}\n"
        return desc
    def execute_single_action(self, action: str) -> Tuple[Tuple[int, int], bool]:
        """执行单个动作并返回新位置"""
        if action not in self.action_map:
            raise ValueError(f"无效动作: {action}")
            
        dx, dy = self.action_map[action]
        new_x = self.current_pos[0] + dx
        new_y = self.current_pos[1] + dy
        new_pos = (new_x, new_y)
        
        # 检查是否撞墙
        if (new_x < 0 or new_x >= self.width or 
            new_y < 0 or new_y >= self.height or
            new_pos in self.walls):
            return self.current_pos, True
            
        self.current_pos = new_pos
        return new_pos, False
class MazeLLMEvaluator:
    """Maze LLM Evaluator"""
    def __init__(self, maze_config: MazeConfig, llm: LlamaInterface, decay_factor: float = 0.9):
        self.env = MazeEnvironment(maze_config)
        self.llm = llm
        self.decay_factor = decay_factor  
    
    def generate_prompt(self, actions: List[str]) -> str:
        """Generate evaluation prompt"""
        prompt = ("You are a maze position predictor. Given a maze configuration and a sequence of actions, "
                 "predict the position after EACH action, including the starting position.\n\n")
        
        prompt += "Maze Configuration:\n"
        prompt += f"- Size: {self.env.width}x{self.env.height} grid\n"
        prompt += f"- Starting position: {self.env.start_pos}\n"
        prompt += f"- Wall positions: {list(self.env.walls)}\n\n"
        
        prompt += f"Action sequence to execute: {actions}\n\n"
        
        prompt += ("Movement rules:\n"
                  "- RIGHT: Move (x+1, y)\n"
                  "- LEFT: Move (x-1, y)\n"
                  "- UP: Move (x, y-1)\n"
                  "- DOWN: Move (x, y+1)\n"
                  "- If you hit a wall or grid boundary, stop at the previous position\n\n")
        
        prompt += "COMMAND: Output a sequence of positions, one per line, including the start position.\n"
        prompt += "Each position must be in the format: (x,y)\n"
        prompt += "Example for actions [RIGHT, DOWN]:\n"
        prompt += "(0,0)\n(1,0)\n(1,1)\n\n"
        prompt += "Your response (one position per line):"
        
        return prompt

    def parse_response(self, response: str) -> List[Optional[Tuple[int, int]]]:
        """Parse LLM response into list of coordinates"""
        try:
            # Split response into lines and clean
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            
            positions = []
            import re
            pattern = r'^\((\d+),(\d+)\)$'
            
            for line in lines:
                match = re.match(pattern, line)
                if match:
                    x, y = map(int, match.groups())
                    positions.append((x, y))
                else:
                    positions.append(None)
            
            return positions
        except Exception as e:
            print(f"Error parsing response: {e}")
            return [None] * (len(actions) + 1)

    def calculate_step_reward(self, 
                            predicted_pos: Tuple[int, int], 
                            true_pos: Tuple[int, int],
                            step: int) -> float:
        """计算单步奖励"""
        if predicted_pos is None:
            return float('-inf')
            
        # 计算曼哈顿距离
        distance = abs(predicted_pos[0] - true_pos[0]) + abs(predicted_pos[1] - true_pos[1])
        
        # 计算时序衰减
        time_decay = self.decay_factor ** step
        
        if distance == 0:
            return time_decay  # 完全正确，应用时序衰减
        else:
            return time_decay / (1.0 + distance)  # 距离越远，奖励越小

    def get_true_trajectory(self, actions: List[str]) -> List[Tuple[int, int]]:
        """获取真实轨迹"""
        trajectory = [self.env.start_pos]
        self.env.reset()
        
        for action in actions:
            pos, hit_wall = self.env.execute_single_action(action)
            trajectory.append(pos)
            if hit_wall:
                # 如果撞墙，后续位置都保持不变
                trajectory.extend([pos] * (len(actions) - len(trajectory) + 1))
                break
        
        # 确保轨迹长度正确
        while len(trajectory) < len(actions) + 1:
            trajectory.append(trajectory[-1])
            
        return trajectory

    def evaluate_prediction(self, actions: List[str]) -> Dict:
        """评估预测"""
        prompt = self.generate_prompt(actions)
        
        system_message = "Output only coordinate sequences, one per line in (x,y) format."
        
        response = self.llm.generate(
            prompt=f"{system_message}\n\n{prompt}",
            stream=True,
            response_format={"type": "text"}
        )
        
        predicted_positions = self.parse_response(response)
        true_trajectory = self.get_true_trajectory(actions)
        
        # 计算每一步的奖励
        step_rewards = []
        for step, (pred_pos, true_pos) in enumerate(zip(predicted_positions, true_trajectory)):
            reward = self.calculate_step_reward(pred_pos, true_pos, step)
            step_rewards.append(reward)
        
        # 计算总奖励
        total_reward = sum(step_rewards)
        
        return {
            "actions": actions,
            "prompt": prompt,
            "response": response,
            "predicted_trajectory": predicted_positions,
            "true_trajectory": true_trajectory,
            "step_rewards": step_rewards,
            "total_reward": total_reward
        }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                    help="模型路径或标识符")
    parser.add_argument("--model_type", type=str, default="local",
                    choices=["local", "anyscale", "together"],
                    help="模型类型")
    parser.add_argument("--api_key", type=str,
                    help="API密钥")
    parser.add_argument("--device", type=str, 
                    default="cuda" if torch.cuda.is_available() else "cpu",
                    help="设备类型")
    parser.add_argument("--maze_width", type=int, default=5,
                    help="迷宫宽度")
    parser.add_argument("--maze_height", type=int, default=5,
                    help="迷宫高度")
    parser.add_argument("--output_dir", type=str, default="results",
                    help="输出目录")
    parser.add_argument("--random_seed", type=int, default=42,
                    help="随机种子")
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.random_seed)
    
    # 初始化LLaMA
    llama_config = LlamaConfig(
        model_path=args.model_path,
        model_type=args.model_type,
        api_key=args.api_key,
        device=args.device
    )
    llm = LlamaInterface(llama_config)
    
    # 创建迷宫配置
    maze_config = MazeConfig(
        width=args.maze_width,
        height=args.maze_height,
        walls=[(1, 1), (2, 2), (3, 3)],  # 示例墙壁
        start_pos=(0, 0)
    )
    
    # 创建评估器
    evaluator = MazeLLMEvaluator(maze_config, llm)
    
    # 测试动作序列
    test_sequences = [
        ["RIGHT", "RIGHT", "DOWN"],
        ["UP", "RIGHT", "DOWN", "LEFT"],
        ["RIGHT", "DOWN", "RIGHT", "UP"]
    ]
    
    # 运行评估
    results = []
    for actions in test_sequences:
        result = evaluator.evaluate_prediction(actions)
        results.append(result)
        print(f"\n评估动作序列: {actions}")
        print(f"预测轨迹: {result['predicted_trajectory']}")
        print(f"真实轨迹: {result['true_trajectory']}")
        print(f"总奖励值: {result['total_reward']}")
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()