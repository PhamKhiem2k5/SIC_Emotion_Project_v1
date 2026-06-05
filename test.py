import cv2
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO
import psutil
import time

# Tải cấu trúc mạng LSTM mà bạn đã định nghĩa
from lstm_model import EmotionLSTM 

# 1. Khởi tạo Mô hình
print("Đang tải mô hình YOLO và LSTM...")
yolo_model = YOLO('yolo11n-pose.pt')

# Khởi tạo LSTM (input=34, hidden=64, layers=1, classes=4) và nạp trọng số đã train
lstm_model = EmotionLSTM(input_size=34, hidden_size=64, num_layers=1, num_classes=4)
lstm_model.load_state_dict(torch.load('lstm_emotion.pth'))
lstm_model.eval() # Chuyển sang chế độ suy luận (không train)

# 2. Cấu hình Nhãn dán
# Lưu ý: OpenCV không hỗ trợ hiển thị tiếng Việt có dấu tốt, nên ta dùng tiếng Việt không dấu
EMOTION_DICT = {
    0: "Binh thuong (Neutral)",
    1: "Cang thang (Stress)",
    2: "Met moi (Fatigue)",
    3: "Vui ve (Happy)"
}

# 3. Chuẩn bị Camera và Hàng đợi (Buffer)
cap = cv2.VideoCapture(0)
sequence_buffer = deque(maxlen=15) # Lưu trữ đúng 15 frames liên tiếp
current_emotion = "Dang phan tich..."

while cap.isOpened():
    start_time = time.time()
    success, frame = cap.read()
    if not success: break

    # Chạy YOLO để lấy khung xương khuôn mặt
    results = yolo_model(frame, verbose=False)
    annotated_frame = results[0].plot()

    if results[0].keypoints is not None and len(results[0].keypoints) > 0:
        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        
        # Đảm bảo YOLO nhận diện đủ 17 điểm mốc
        if keypoints.shape[0] == 17:
            # --- CHUẨN HÓA DỮ LIỆU (Cực kỳ quan trọng) ---
            nose_x, nose_y = keypoints[0][0], keypoints[0][1]
            pts_shifted = keypoints - [nose_x, nose_y]
            face_width = np.max(keypoints[:, 0]) - np.min(keypoints[:, 0]) + 1e-6
            pts_normalized = pts_shifted / face_width
            
            # Thêm mảng 34 tọa độ đã chuẩn hóa vào hàng đợi
            sequence_buffer.append(pts_normalized.flatten())

            # --- DỰ ĐOÁN BẰNG LSTM ---
            # Chỉ dự đoán khi đã gom đủ 15 khung hình
            if len(sequence_buffer) == 15:
                # Chuyển đổi thành Tensor của PyTorch: shape (1 batch, 15 frames, 34 features)
                input_tensor = torch.tensor([list(sequence_buffer)], dtype=torch.float32)
                
                with torch.no_grad(): # Tắt tính gradient để tiết kiệm CPU
                    output = lstm_model(input_tensor)
                    _, predicted_class = torch.max(output.data, 1)
                    current_emotion = EMOTION_DICT[predicted_class.item()]

    # Tính toán thông số phần cứng
    fps = 1.0 / (time.time() - start_time)
    cpu_usage = psutil.cpu_percent()

    # Hiển thị Kết quả Cảm xúc và Thông số lên màn hình
    cv2.putText(annotated_frame, f"Trang thai: {current_emotion}", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.putText(annotated_frame, f"FPS: {fps:.1f} | CPU: {cpu_usage}%", (10, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("SIC Emotion Real-time Test", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()