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
            "UP": (0, 1),
            "DOWN": (0, -1),
            "LEFT": (-1, 0),
            "RIGHT": (1, 0)
        }
    
    def reset(self) -> Tuple[int, int]:
        self.current_pos = self.start_pos
        return self.current_pos
    
    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        return (0 <= x < self.width and 
                0 <= y < self.height and 
                pos not in self.walls)
    
    def execute_single_action(self, action: str) -> Tuple[Tuple[int, int], bool]:
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
    
    def get_true_trajectory(self, actions: List[str]) -> List[Tuple[int, int]]:
        """获取真实轨迹"""
        self.reset()
        trajectory = [self.current_pos]
        
        for action in actions:
            new_pos, _ = self.execute_single_action(action)
            trajectory.append(new_pos)
        
        return trajectory

class MazeGPTSystem:
    """迷宫GPT系统(包含预测和判断功能)"""
    def __init__(self, maze_config: MazeConfig, predictor_gpt: OpenAIInterface, judge_gpt: OpenAIInterface):
        self.env = MazeEnvironment(maze_config)
        self.predictor_gpt = predictor_gpt
        self.judge_gpt = judge_gpt
    
    def generate_predictor_prompt(self, actions: List[str]) -> str:
        """Generate predictor prompt for a single trajectory"""
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
        prompt += " Your response (one position per line) just output the predicted trajectory do not include any other text like why this is correct or anything else!!"
        
        return prompt
    
    def generate_judge_prompt(self, actions: List[str], predictor_response: str) -> str:
        """Generate judge prompt"""
        prompt = ("You are a maze trajectory judge. Your task is to analyze the predicted trajectories and verify if the position changes between each step match the given actions.\n\n"
             
             "Maze Configuration:\n"
             f"- Size: {self.env.width}x{self.env.height} grid\n"
             f"- Starting position: {self.env.start_pos}\n"
             f"- Wall positions: {list(self.env.walls)}\n\n"
             
             f"Action sequence: {actions}\n\n"
             
             "Movement Rules:\n"
             "- RIGHT: Must change from (x,y) to (x+1,y)\n"
             "- LEFT: Must change from (x,y) to (x-1,y)\n"
             "- UP: Must change from (x,y) to (x,y+1)\n"
             "- DOWN: Must change from (x,y) to (x,y-1)\n"
             "- If a move would hit a wall or go out of bounds, position should remain unchanged\n\n"
             
             "Validation Process:\n"
             "For each trajectory, check between every two consecutive positions:\n"
             "1. Does the position change match the corresponding action?\n"
             "2. If position didn't change, was it because of a wall or boundary?\n"
             "3. Are all positions within the grid (0-4)?\n"
             "4. Are all positions avoiding walls?\n\n"
             
             "Examples:\n"
             "- If action is RIGHT at (0,0): valid next position is (1,0)\n"
             "- If action is UP at (0,0): valid next position is (0,1)\n"
             "- If action is LEFT at (0,0): must stay at (0,0) due to boundary\n\n"
             " REMEMBER that position is not valid if it is out of bounds or hits a wall!!\n\n"
             "Output Format:\n"
             "- Use (x,y) format for positions\n"
             "- One position per line\n"
                 f"- Must output exactly {len(actions) + 1} positions\n"
                 "- No additional text\n"
                 "Output only the single correct trajectory DO NOT include any other text like why this is correct or anything else.Do not predict the position by yourself just judge the predicted trajectories!!")
        prompt += "tell you again the problem set up: Maze Configuration:\n"
        prompt += f"- Size: {self.env.width}x{self.env.height} grid\n"
        prompt += f"- Starting position: {self.env.start_pos}\n"
        prompt += f"- Wall positions: {list(self.env.walls)}\n\n"
        prompt += f"here is the action sequence and the predicted trajectories you need to judge: Action sequence: {actions}\n\n"
        prompt += f"Predicted trajectories:\n{predictor_response}\n\n"
        return prompt

    def parse_trajectories(self, response: str) -> List[List[Optional[Tuple[int, int]]]]:
        """解析多个轨迹预测结果"""
        trajectories = []
        current_trajectory = []
        
        for line in response.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line == "---":
                if current_trajectory:
                    trajectories.append(current_trajectory)
                    current_trajectory = []
                continue
                
            if line.startswith('('):
                try:
                    coords = line.strip('()').split(',')
                    if len(coords) == 2:
                        x, y = map(int, coords)
                        if 0 <= x < self.env.width and 0 <= y < self.env.height:
                            current_trajectory.append((x, y))
                        else:
                            current_trajectory.append(None)
                except:
                    continue
        
        if current_trajectory:
            trajectories.append(current_trajectory)
            
        return trajectories

    def parse_trajectory(self, response: str) -> List[Optional[Tuple[int, int]]]:
        """解析单个轨迹预测结果"""
        positions = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if not line or not line.startswith('('):
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
    
    def compare_trajectories(self, pred_traj: List[Optional[Tuple[int, int]]], true_traj: List[Tuple[int, int]]) -> bool:
        """比较两个轨迹是否相同"""
        if len(pred_traj) != len(true_traj):
            return False
        return all(p == t for p, t in zip(pred_traj, true_traj))
    
    def calculate_accuracy(self, predictions: List[List[Optional[Tuple[int, int]]]], 
                         judge_prediction: List[Optional[Tuple[int, int]]], 
                         true_trajectory: List[Tuple[int, int]]) -> Dict:
        """计算预测正确率"""
        # 计算预测器的正确率
        correct_predictions = sum(1 for pred in predictions 
                                if self.compare_trajectories(pred, true_trajectory))
        predictor_accuracy = correct_predictions / len(predictions) if predictions else 0
        
        # 计算判断器的正确率
        judge_correct = self.compare_trajectories(judge_prediction, true_trajectory)
        
        return {
            "predictor_accuracy": predictor_accuracy,
            "predictor_correct_count": correct_predictions,
            "predictor_total_count": len(predictions),
            "judge_accuracy": 1.0 if judge_correct else 0.0,
            "judge_correct": judge_correct
        }
    
    def run_test(self, actions: List[str], num_samples: int = 5) -> Dict:
        """运行完整的测试流程"""
        # 第一步: 生成多个预测（通过多次调用）
        predictions = []
        predictor_responses = []
        
        print("\nGenerating predictions...")
        for i in range(num_samples):
            print(f"\nPrediction {i+1}/{num_samples}")
            self.env.reset()
            
            predictor_prompt = self.generate_predictor_prompt(actions)
            predictor_response = self.predictor_gpt.generate(predictor_prompt, stream=True)
            
            prediction = self.parse_trajectory(predictor_response)
            if prediction:  # 只添加有效的预测
                predictions.append(prediction)
                predictor_responses.append(predictor_response)
        
        # 合并所有预测结果为一个字符串，添加分隔符
        combined_predictions = "\n---\n".join(predictor_responses)
        
        # 第二步: 让judge给出正确答案
        print("\nGenerating judge response...")
        self.env.reset()
        judge_prompt = self.generate_judge_prompt(actions, combined_predictions)
        judge_response = self.judge_gpt.generate(judge_prompt, stream=True)
        judge_prediction = self.parse_trajectory(judge_response)
        
        # 第三步: 获取真实轨迹进行比较
        self.env.reset()
        true_trajectory = self.env.get_true_trajectory(actions)
        
        # 第四步: 计算正确率
        accuracy_stats = self.calculate_accuracy(predictions, judge_prediction, true_trajectory)
        
        return {
            "actions": actions,
            "predictor_responses": predictor_responses,
            "predictions": [
                [list(pos) if pos else None for pos in pred]
                for pred in predictions
            ],
            "judge_response": judge_response,
            "judge_prediction": [list(pos) if pos else None for pos in judge_prediction],
            "true_trajectory": [list(pos) for pos in true_trajectory],
            "accuracy_stats": accuracy_stats
        }

class TestCaseGenerator:
    """测试用例生成器"""
    def __init__(self, maze_config: MazeConfig):
        self.maze_config = maze_config
        self.actions = ["UP", "DOWN", "LEFT", "RIGHT"]
    
    def generate_random_sequence(self, min_length: int = 2, max_length: int = 5) -> List[str]:
        length = random.randint(min_length, max_length)
        return random.choices(self.actions, k=length)
    
    def generate_test_cases(self, num_cases: int) -> List[List[str]]:
        test_cases = []
        for _ in range(num_cases):
            test_cases.append(self.generate_random_sequence())
        return test_cases

def parse_args():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--num_samples", type=int, default=4,
                    help="Number of predictions to generate per test case")
    parser.add_argument("--num_tests", type=int, default=10,
                    help="Number of test cases")
    parser.add_argument("--output_dir", type=str, default="results",
                    help="Output directory")
    parser.add_argument("--random_seed", type=int, default=42,
                    help="Random seed")
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.random_seed)
    
    # 初始化OpenAI配置
    openai_config = OpenAIConfig(
        model=args.model,
        base_url=args.base_url
    )
    
    # 创建两个GPT实例
    predictor_gpt = OpenAIInterface(openai_config)
    judge_gpt = OpenAIInterface(openai_config)
    
    # 创建迷宫配置
    maze_config = MazeConfig(
        width=args.maze_width,
        height=args.maze_height,
        walls=[(1, 1), (2, 2), (3, 3)],  # 示例墙壁
        start_pos=(0, 0)
    )
    
    # 创建系统
    system = MazeGPTSystem(maze_config, predictor_gpt, judge_gpt)
    
    # 生成测试用例
    test_generator = TestCaseGenerator(maze_config)
    test_cases = test_generator.generate_test_cases(args.num_tests)
    
    # 运行测试
    results = []
    total_predictor_accuracy = 0
    total_judge_accuracy = 0
    
    print(f"\n开始测试 {args.num_tests} 组用例...")
    
    for i, actions in enumerate(test_cases, 1):
        print(f"\n测试进度: {i}/{args.num_tests}")
        print(f"动作序列: {actions}")
        
        # 运行完整的测试流程
        result = system.run_test(actions, num_samples=args.num_samples)
        results.append(result)
        
        # 累计正确率
        total_predictor_accuracy += result["accuracy_stats"]["predictor_accuracy"]
        total_judge_accuracy += result["accuracy_stats"]["judge_accuracy"]
        
        # 打印当前测试用例的正确率
        print(f"预测器正确率: {result['accuracy_stats']['predictor_accuracy']:.2%} ({result['accuracy_stats']['predictor_correct_count']}/{result['accuracy_stats']['predictor_total_count']})")
        print(f"判断器正确: {'是' if result['accuracy_stats']['judge_correct'] else '否'}")
    
    # 计算平均正确率
    avg_predictor_accuracy = total_predictor_accuracy / len(test_cases)
    avg_judge_accuracy = total_judge_accuracy / len(test_cases)
    
    print(f"\n测试完成！总体统计：")
    print(f"预测器平均正确率: {avg_predictor_accuracy:.2%}")
    print(f"判断器平均正确率: {avg_judge_accuracy:.2%}")
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"maze_gpt_results_{timestamp}.json"
    
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
            "overall_stats": {
                "avg_predictor_accuracy": avg_predictor_accuracy,
                "avg_judge_accuracy": avg_judge_accuracy,
                "total_test_cases": len(test_cases)
            }
        }, f, indent=4)
    
    print(f"\n测试结果已保存到: {result_path}")

if __name__ == "__main__":
    main() 