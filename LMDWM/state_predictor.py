import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
from wm_predict import GameStateMLP
import json

class StatePredictor:
    def __init__(self, model_path: str):
        checkpoint = torch.load(model_path)
        self.model = self._load_model(checkpoint)
        self.state_dims = checkpoint['state_dims']
        self.hidden_dims = checkpoint['hidden_dims']
        self.action_vocab = checkpoint['action_vocab']
        
    def _load_model(self, checkpoint):
        # 加载模型（与你的MLP结构保持一致）
        model = GameStateMLP(input_dim=self.state_dims, hidden_dims=self.hidden_dims, output_dim=self.state_dims)  
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    def predict(self, state: Dict, action: str, reward: float) -> Tuple[np.ndarray, float]:
        """单次预测"""
        with torch.no_grad():
            x = self._prepare_input(state, action, reward)
            prediction = self.model(x)
            return self._decode_prediction(prediction)

class EnsemblePredictor:
    def __init__(self, predictor: StatePredictor, llm: LlamaInterface):
        self.predictor = predictor
        self.llm = llm
    
    def predict_with_ensemble(self, 
                            state: Dict, 
                            action: str, 
                            reward: float, 
                            k: int = 5,
                            temperature: float = 0.1) -> Dict:
        """进行k次预测并使用LLM综合结果"""
        # 收集k次预测结果
        predictions = []
        for _ in range(k):
            next_state, next_reward = self.predictor.predict(state, action, reward)
            predictions.append({
                "state": next_state,
                "reward": next_reward
            })
        
        # 构建提示词
        prompt = self._build_ensemble_prompt(state, action, reward, predictions)
        
        # 使用LLM判断
        response = self.llm.generate(
            prompt=prompt,
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        
        return self._parse_llm_response(response)
    
    def _build_ensemble_prompt(self, 
                             state: Dict, 
                             action: str, 
                             reward: float, 
                             predictions: List[Dict]) -> str:
        """构建LLM提示词"""
        prompt = f"""Given the current game state and {len(predictions)} different predictions for the next state after action '{action}', 
please analyze these predictions and provide the most likely next state. You can either choose one of the predictions or generate a new one.

Current State:
{json.dumps(state, indent=2)}

Action: {action}
Current Reward: {reward}

Predictions:
"""
        
        for i, pred in enumerate(predictions, 1):
            prompt += f"\nPrediction {i}:\n{json.dumps(pred, indent=2)}"
            
        prompt += """\n
Please analyze these predictions and provide:
1. The most likely next state and reward
2. A brief explanation of your choice

Output your response in JSON format with keys: "chosen_state", "chosen_reward", "explanation"
"""
        return prompt
    
    def _parse_llm_response(self, response: str) -> Dict:
        """解析LLM的响应"""
        try:
            result = json.loads(response)
            return {
                "next_state": result["chosen_state"],
                "next_reward": result["chosen_reward"],
                "explanation": result["explanation"]
            }
        except:
            # 如果解析失败，返回第一个预测结果
            return {
                "next_state": predictions[0]["state"],
                "next_reward": predictions[0]["reward"],
                "explanation": "Failed to parse LLM response, using first prediction."
            }