import torch
import torch.nn as nn

class EmotionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(EmotionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x có dạng: (batch_size, sequence_length, input_size)
        out, _ = self.lstm(x)
        # Chỉ lấy đầu ra của bước thời gian (time-step) cuối cùng để phân loại
        out = self.fc(out[:, -1, :]) 
        return out