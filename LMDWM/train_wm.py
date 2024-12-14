# train.py
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import GameStateMLP
from data_processor import GameStateProcessor
import torch.nn as nn

def train_model(data_file: str, epochs: int = 100, batch_size: int = 32):
    # 处理数据
    processor = GameStateProcessor()
    X, y = processor.process_game_data(data_file)
    
    # 转换为 PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    # 创建数据加载器
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    input_dim = X.shape[1]
    hidden_dims = [256, 128, 64]  # 可以调整
    output_dim = y.shape[1]
    
    model = GameStateMLP(input_dim, hidden_dims, output_dim)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    # 训练循环
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            
            # 前向传播
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
    
    return model, processor

if __name__ == "__main__":
    model, processor = train_model("data/games/balance-scale-heaviest.jsonl")
    
    # 保存模型和处理器
    torch.save({
        'model_state_dict': model.state_dict(),
        'state_dims': processor.state_dims,
        'action_vocab': processor.action_vocab
    }, 'game_state_model.pth')