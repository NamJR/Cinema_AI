Hướng dẫn chạy dự án nhận diện khuôn mặt (tiếng Việt)

Yêu cầu:
- Python 3.8+
- Mạng Internet để tải mô hình ONNX (hoặc bạn tải thủ công vào thư mục `models/`)

Cài đặt:
1. Tạo virtualenv và kích hoạt (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Cài đặt phụ thuộc:

```powershell
pip install -r requirements.txt
```

Tải mô hình:
- Đặt các tệp mô hình ONNX vào thư mục `models/` với tên `scrfd.onnx` và `arcface.onnx`.
- Tham khảo hướng dẫn trong `models/README.md` để biết nơi tải xuống.

Chạy server (Windows):

```powershell
run.bat
```

Hoặc chạy trực tiếp với uvicorn:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

API:
- `POST /register/{user_id}`: đăng ký khuôn mặt (multipart form file)
- `POST /verify` : xác thực khuôn mặt

Ghi chú:
- Kiểm tra `database.py` để biết cấu hình SQLite/SQLAlchemy.
- Nếu không phát hiện khuôn mặt, kiểm tra ảnh đầu vào và mô hình `scrfd.onnx`.
