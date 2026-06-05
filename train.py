import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from lstm_model import EmotionLSTM # Tải kiến trúc mô hình từ file bạn đã tạo

# 1. Thiết lập siêu tham số (Hyperparameters)
SEQ_LENGTH = 15     # Số khung hình liên tiếp để tạo thành 1 chuỗi
BATCH_SIZE = 32     # Số lượng mẫu đưa vào huấn luyện mỗi lần
EPOCHS = 50         # Số vòng lặp huấn luyện toàn bộ dữ liệu
LEARNING_RATE = 0.001

# 2. Xử lý dữ liệu (Data Preprocessing) - ĐÃ ĐƯỢC CHUẨN HÓA
def load_and_preprocess_data(csv_file, seq_length):
    print("Đang đọc và chuẩn hóa dữ liệu từ CSV...")
    df = pd.read_csv(csv_file)
    
    labels = df['label'].values
    raw_features = df.drop('label', axis=1).values 
    
    normalized_features = []
    
    # Duyệt qua từng khung hình để chuẩn hóa
    for row in raw_features:
        # Reshape thành mảng 2D: 17 điểm, mỗi điểm 2 tọa độ (x, y)
        pts = row.reshape(-1, 2)
        
        # 1. Lấy tọa độ mũi làm gốc (Trong YOLO-pose, mũi thường là index 0)
        nose_x, nose_y = pts[0][0], pts[0][1]
        
        # 2. Dời gốc tọa độ: Trừ tất cả các điểm cho tọa độ của mũi
        pts_shifted = pts - [nose_x, nose_y]
        
        # 3. Tính chiều rộng khuôn mặt để làm tỷ lệ (Scale)
        # Cộng thêm 1e-6 để tránh lỗi chia cho 0
        face_width = np.max(pts[:, 0]) - np.min(pts[:, 0]) + 1e-6
        
        # 4. Tỉ lệ hóa: Chia tất cả tọa độ cho chiều rộng
        pts_normalized = pts_shifted / face_width
        
        # Duỗi phẳng lại thành mảng 1D và lưu lại
        normalized_features.append(pts_normalized.flatten())
        
    normalized_features = np.array(normalized_features)
    
    sequences = []
    seq_labels = []
    
    # Dùng cửa sổ trượt (sliding window) để gom các frame thành chuỗi
    for i in range(len(normalized_features) - seq_length):
        # Chỉ tạo chuỗi nếu 15 frames liên tiếp này thuộc cùng 1 trạng thái cảm xúc
        if len(set(labels[i : i + seq_length])) == 1:
            sequences.append(normalized_features[i : i + seq_length])
            seq_labels.append(labels[i])
            
    return np.array(sequences), np.array(seq_labels)

# Khởi tạo class Dataset cho PyTorch
class EmotionDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# 3. Chuẩn bị luồng huấn luyện
X, y = load_and_preprocess_data('emotion_dataset.csv', SEQ_LENGTH)
print(f"Tổng số chuỗi (sequences) hợp lệ thu được: {len(X)}")

# Chia tập dữ liệu: 80% để Train, 20% để Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_loader = DataLoader(EmotionDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(EmotionDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

# 4. Khởi tạo Mô hình, Hàm mất mát và Tối ưu hóa
# Input size = 34 (vì YOLO11-pose có 17 điểm, mỗi điểm có x và y -> 17 * 2 = 34)
model = EmotionLSTM(input_size=34, hidden_size=64, num_layers=1, num_classes=4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 5. Vòng lặp Huấn luyện (Training Loop)
print("Bắt đầu huấn luyện mô hình...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    
    for sequences, labels in train_loader:
        optimizer.zero_grad() # Xóa gradient cũ
        outputs = model(sequences) # Dự đoán
        loss = criterion(outputs, labels) # Tính sai số
        
        loss.backward() # Lan truyền ngược
        optimizer.step() # Cập nhật trọng số
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        
    train_acc = 100 * correct / len(y_train)
    
    # In kết quả mỗi 10 vòng
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {train_acc:.2f}%')

# 6. Lưu mô hình
torch.save(model.state_dict(), 'lstm_emotion.pth')
print("Đã lưu trọng số mô hình vào file 'lstm_emotion.pth' thành công!")