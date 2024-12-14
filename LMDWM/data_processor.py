# data_processor.py
import json
import numpy as np
from typing import List, Tuple, Dict

class GameStateProcessor:
    def __init__(self):
        self.state_dims = None  # 将在处理第一个状态时确定
        self.action_vocab = {}  # 动作词汇表
        self.next_action_id = 0
    
    def _state_to_vector(self, state: Dict) -> np.ndarray:
        """将游戏状态转换为向量"""
        # 这里需要根据你的游戏状态结构来实现
        # 示例：提取关键特征并转换为向量
        state_vector = []
        game_state = state["game_state"]
        
        for obj in game_state:
            # 提取对象属性
            if "properties" in obj:
                for prop, value in obj["properties"].items():
                    if isinstance(value, bool):
                        state_vector.append(float(value))
                    elif isinstance(value, (int, float)):
                        state_vector.append(float(value))
                    
        return np.array(state_vector)
    
    def _action_to_vector(self, action: str) -> np.ndarray:
        """将动作转换为one-hot向量"""
        if action not in self.action_vocab:
            self.action_vocab[action] = self.next_action_id
            self.next_action_id += 1
        
        action_vector = np.zeros(len(self.action_vocab))
        action_vector[self.action_vocab[action]] = 1
        return action_vector
    
    def process_game_data(self, data_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """处理游戏数据，返回(X, y)训练数据"""
        X_data = []  # 输入：(s, a, r)
        y_data = []  # 输出：(s', r')
        
        with open(data_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                
                # 获取当前状态向量
                curr_state_vec = self._state_to_vector(data["curr_state"])
                if self.state_dims is None:
                    self.state_dims = len(curr_state_vec)
                
                # 获取动作向量
                action_vec = self._action_to_vector(data["action"])
                
                # 获取当前奖励
                curr_reward = float(data["curr_state"]["game_state"][-1].get("score", 0))
                
                # 获取下一个状态和奖励
                next_state_vec = self._state_to_vector(data["gold_state"])
                next_reward = float(data["gold_state"]["game_state"][-1].get("score", 0))
                
                # 组合输入和输出
                X = np.concatenate([curr_state_vec, action_vec, [curr_reward]])
                y = np.concatenate([next_state_vec, [next_reward]])
                
                X_data.append(X)
                y_data.append(y)
        
        return np.array(X_data), np.array(y_data)