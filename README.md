# AIC 2025 - Video Retrieval System

Hệ thống Tìm kiếm Video Thông minh (Multi-modal Video Retrieval System) được xây dựng cho các cuộc thi AI (như AIC). Hệ thống cho phép tìm kiếm phân đoạn video chính xác thông qua văn bản tiếng Việt dựa trên sự kết hợp của nhiều AI models (SigLIP2, Jina-CLIP, PaddleOCR, YOLOv11) và truy xuất đa nhánh (Multi-branch Retrieval).

## 📂 Cấu trúc thư mục (Project Structure)

Toàn bộ giải pháp được chia thành 3 phần chính:

- **`video_retrieval/`** (Python / FastAPI)
  - Hệ thống Backend xử lý cốt lõi. Chứa toàn bộ Pipeline (Offline extraction & Online retrieval).
  - Tích hợp FAISS (Vector DB siêu tốc), SQLite, BM25.
  - Phân tích câu truy vấn tiếng Việt bằng LLM (Ollama / Gemini).
- **`web/`** (React / TypeScript / Vite)
  - Giao diện người dùng (Frontend).
  - Cho phép người dùng nhập query, hiển thị kết quả, trích xuất metadata và xem trực tiếp video tại khung hình (frame) được chỉ định.
- **`model_training/`**
  - Chứa mã nguồn để fine-tune các mô hình AI trên tập dữ liệu đặc thù của cuộc thi.

## 🚀 Hướng dẫn khởi động nhanh (Quick Start)

### 1. Khởi động Backend (API)
Mở terminal và trỏ vào thư mục `video_retrieval/`:
```bash
cd video_retrieval
pip install -r requirements.txt
pip install faiss-cpu  # hoặc faiss-gpu nếu dùng card NVIDIA

# Khởi chạy server FastAPI (Port 8000)
python api/main.py
```

### 2. Khởi động Frontend (UI)
Mở một terminal khác và trỏ vào thư mục `web/`:
```bash
cd web
npm install
npm run dev
```
Truy cập giao diện tại `http://localhost:5173`.

## ☁️ Quy trình xử lý Video (Kaggle Pipeline)

Việc trích xuất hàng ngàn Video (chạy AI models) rất tốn tài nguyên. Bạn có thể chạy quá trình nặng này trên **Kaggle (Free GPU)** thay vì chạy ở máy tính cá nhân (Local):
1. Đưa mã nguồn `video_retrieval` và các video `.mp4` lên Kaggle Notebook.
2. Chạy quá trình trích xuất (`scripts/index_videos.py`).
3. Tải các file `.faiss` và `.sqlite` (Kaggle Output) về máy tính.
4. Bỏ các file đó vào `video_retrieval/data/indexes/`. Bỏ video vào `data/videos/`.
5. Bật Web UI và tìm kiếm ngay lập tức trên máy Local của bạn với độ trễ tính bằng mili-giây!

👉 *Xem hướng dẫn chi tiết từng bước tại `video_retrieval/README.md`*
