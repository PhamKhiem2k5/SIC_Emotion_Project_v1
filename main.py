import cv2
from ultralytics import YOLO
import time
import psutil
import os
import csv

# 1. Cấu hình file lưu dữ liệu
csv_filename = "emotion_dataset.csv"
# Tạo header (tiêu đề cột) nếu file chưa tồn tại. 
# YOLO11-pose thường xuất ra 17 điểm mốc, mỗi điểm có tọa độ (x, y)
if not os.path.exists(csv_filename):
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['label'] + [f'pt_{i}_{axis}' for i in range(17) for axis in ('x', 'y')]
        writer.writerow(header)

# 2. Khởi tạo mô hình và Camera
model = YOLO('yolo11n-pose.pt')
cap = cv2.VideoCapture(0)

# 3. Biến trạng thái để kiểm soát việc ghi dữ liệu
recording = False
current_label = -1
frames_recorded = 0
MAX_FRAMES = 300 # Lưu 300 khung hình cho mỗi lần bấm phím

while cap.isOpened():
    start_time = time.time()
    success, frame = cap.read()
    if not success: break

    # Chạy suy luận YOLO
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    # Xử lý trích xuất tọa độ
    if results[0].keypoints is not None and len(results[0].keypoints) > 0:
        # Lấy tọa độ (x, y) của người đầu tiên
        keypoints = results[0].keypoints.xy[0].cpu().numpy() 
        
        # Nếu đang trong chế độ "Ghi", tiến hành lưu vào file CSV
        if recording:
            # Duỗi phẳng mảng tọa độ (từ 2D thành 1D) để lưu thành 1 hàng
            flattened_kpts = keypoints.flatten().tolist()
            
            with open(csv_filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([current_label] + flattened_kpts)
            
            frames_recorded += 1
            
            # Hiển thị cảnh báo màu đỏ trên màn hình để bạn biết hệ thống đang ghi
            cv2.putText(annotated_frame, f"REC: Label {current_label} ({frames_recorded}/{MAX_FRAMES})", 
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Tự động dừng ghi khi đủ 300 frames
            if frames_recorded >= MAX_FRAMES:
                recording = False
                print(f"Đã lưu xong {MAX_FRAMES} frames cho nhãn {current_label}!")

    # Đo lường tài nguyên
    fps = 1.0 / (time.time() - start_time)
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

    # Hiển thị thông số
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"CPU: {cpu_usage}%", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"RAM: {ram_usage:.1f} MB", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Data Collection Mode", annotated_frame)

    # Bắt sự kiện bàn phím
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    # Nhấn các phím 0, 1, 2, 3 để bắt đầu ghi nhãn tương ứng
    elif key in [ord('0'), ord('1'), ord('2'), ord('3')]:
        if not recording: # Chỉ cho phép ghi nếu lần bấm trước đã hoàn thành
            current_label = int(chr(key))
            recording = True
            frames_recorded = 0
            print(f"Bắt đầu ghi dữ liệu cho nhãn: {current_label}")

cap.release()
cv2.destroyAllWindows()