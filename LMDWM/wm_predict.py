# predict.py
import torch
import numpy as np
from model import GameStateMLP
from data_processor import DataProcessor
from argparse import ArgumentParser

def predict_next_state(model, processor, current_state, action, current_reward):
    # 准备输入数据
    state_vec = processor._state_to_vector(current_state)
    action_vec = processor._action_to_vector(action)
    X = torch.FloatTensor(np.concatenate([state_vec, action_vec, [current_reward]]))
    
    # 预测
    with torch.no_grad():
        prediction = model(X.unsqueeze(0))
    
    # 解码预测结果
    next_state_vec = prediction[0, :-1].numpy()
    next_reward = prediction[0, -1].item()
    
    return next_state_vec, next_reward

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--input_dim", type=int, required=True)
    parser.add_argument("--hidden_dims", type=list, required=True)
    parser.add_argument("--output_dim", type=int, required=True)
    args = parser.parse_args()
    
    # 加载模型
    checkpoint = torch.load(args.model)
    model = GameStateMLP(args.input_dim, args.hidden_dims, args.output_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    # 预测
    next_state, next_reward = predict_next_state(model, processor, current_state, action, current_reward)