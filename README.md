Bản demo chạy chưa ổn


1/Thu thập Dữ liệu (Data Collection)

Tại Terminal, gõ lệnh python main.py
Bắt đầu diễn xuất nét mặt và nhấn các phím số tương ứng trên bàn phím để hệ thống ghi hình:

  Phím 0 - Bình thường: Trạng thái làm việc tiêu chuẩn, cơ mặt thả lỏng.
  
  Phím 1 - Căng thẳng: Nhíu mày, căng cơ mặt.
  
  Phím 2 - Mệt mỏi: Mắt chớp chậm, ngáp, hoặc nhắm mắt.
  
  Phím 3 - Vui vẻ: Cười, giãn cơ mặt.

2/Huấn luyện Mô hình (Training)

Tại Terminal, gõ lệnh python train.py và nhấn Enter.
File này sẽ tự động đọc dữ liệu tọa độ, tiến hành chuẩn hóa (Normalization) để loại bỏ hiệu ứng xa gần của camera, và bắt đầu huấn luyện mạng LSTM.

Khi quá trình chạy hoàn tất (chỉ số Loss giảm, Accuracy tăng), hệ thống sẽ xuất ra một tệp trọng số (ví dụ: lstm_emotion.pth hoặc best_model.pth). Đây chính là "bộ não" đã được học thuộc biểu cảm của bạn.

3/Chạy Nhận diện Thực tế

Mở Terminal và gõ lệnh chạy file thực tế: python tes.py
