# Video Retrieval System (Backend)

Hệ thống xử lý cốt lõi cho ứng dụng tìm kiếm video, sử dụng kiến trúc AI đa phương thức (SigLIP2, Jina-CLIP, FAISS, Gemini/Ollama).

## 📂 Kết cấu thư mục (Folder Structure)

```text
video_retrieval/
├── agent/            # Xử lý logic ReAct Agent (chưa kích hoạt mặc định)
├── api/              # FastAPI endpoints (routes, models) cho giao diện Web
├── config/           # Cấu hình hệ thống (Database, Embedding, Settings, v.v.)
├── data/             # THƯ MỤC QUAN TRỌNG: Chứa dữ liệu chạy của hệ thống
│   ├── indexes/      # -> Chứa file FAISS (.faiss) và file Metadata SQLite (.sqlite)
│   ├── videos/       # -> Chứa các file video gốc (.mp4) để trích xuất frame và xem trên Web
│   └── frames/       # -> Chứa các ảnh frame tạm thời (có thể xóa sau khi nhúng xong)
├── database/         # Logic kết nối Database: FAISS client và SQLite metadata DB
├── interfaces/       # Các class Data (Pydantic models, Abstract base classes)
├── models/           # Các wrapper bọc AI Models (SigLIP2, Jina, YOLO, PaddleOCR)
├── pipelines/        # Luồng Orchestrator: Offline (Trích xuất) & Online (Tìm kiếm)
├── query_parser/     # Xử lý ngôn ngữ tự nhiên, dịch tiếng Việt -> Anh bằng LLM (Ollama/Gemini)
├── retrieval/        # Các nhánh tìm kiếm (Branches), Temporal Refinement (Xử lý 2 sự kiện)
├── scripts/          # Các script chạy độc lập: index_videos.py, check_system.py,...
├── services/         # Tầng Business Logic: Frame extraction, Embedding service, OCR service
└── utils/            # Các hàm hỗ trợ (Logging, file utils...)
```

## ☁️ Hướng dẫn chạy trích xuất trên Kaggle

Việc trích xuất hàng ngàn Video rất tốn tài nguyên và thời gian. Chạy trên Kaggle (miễn phí GPU T4x2 hoặc P100) là lựa chọn tối ưu.

### Bước 1: Trích xuất trên Kaggle
1. Đăng nhập Kaggle và chọn **Create > New Notebook**. Sau đó chọn **File > Import Notebook** và tải lên file `kaggle_pipeline.ipynb` (nếu có) từ thư mục gốc của project.
2. Để lấy code mới nhất vào Kaggle, tạo một ô code (cell) và dùng lệnh `git clone` với GitHub Token của bạn (lấy tại *Developer Settings > Personal access tokens* trên GitHub):
   ```bash
   !git clone https://<YOUR_GITHUB_TOKEN>@github.com/Thien-Dan/agent_retrieval.git
   ```
3. Upload bộ video `.mp4` lên làm Kaggle Dataset và add vào Notebook.
4. Đảm bảo Notebook đã bật mạng (Internet on) và chọn GPU (Accelerator). Chạy các ô lệnh trong file notebook để cài đặt thư viện và tiến hành trích xuất.
5. Kiểm tra code đảm bảo luồng lưu file Database (FAISS, SQLite) trỏ vào thư mục `/kaggle/working/` để có thể tải về sau đó.

### Bước 2: Lưu và thiết lập Data sau khi chạy xong
1. Sau khi chạy xong, Kaggle sẽ sinh ra các file `.faiss` và `.sqlite` nằm trong `/kaggle/working/data/indexes/`.
2. Bấm Save Version (Save & Run All) để Kaggle lưu Output lại.
3. **Tải toàn bộ Output (FAISS + SQLite) về máy tính cá nhân**.
4. Chép các file vừa tải về vào đúng thư mục `video_retrieval/data/indexes/` trên máy tính của bạn.
5. Chép các file video gốc (`.mp4`) vào thư mục `video_retrieval/data/videos/`.

Lúc này, hệ thống trên máy cá nhân của bạn đã có đủ Dữ Liệu Vectơ (FAISS) và Dữ liệu Gốc (Videos). Bạn có thể khởi động UI và tìm kiếm ngay lập tức với tốc độ mili-giây!

## 💻 Hướng dẫn chạy hoàn toàn trên máy cá nhân (Local)

Nếu máy bạn có GPU mạnh (hoặc không ngại chạy CPU hơi chậm), bạn hoàn toàn có thể bỏ qua Kaggle và chạy mọi thứ trên máy cá nhân.

### Bước 1: Setup Dữ liệu & Môi trường
1. Bỏ toàn bộ video `.mp4` vào thư mục `video_retrieval/data/videos/`.
2. Cài đặt thư viện: `pip install -r requirements.txt`.
3. Tạo file `.env` từ `.env.example` và điền key Gemini (nếu dùng):
   `cp .env.example .env`

### Bước 2: Kiểm tra sức khỏe hệ thống
Chạy script kiểm tra xem mọi thứ (FAISS, SigLIP2, LLM...) đã sẵn sàng chưa:
```bash
python scripts/check_system.py
```

### Bước 3: Trích xuất Video (Offline Pipeline)
Chạy script sau để tự động cắt frame, đẩy qua AI Models và tạo DB:
```bash
python scripts/index_videos.py --video_dir data/videos/
```
*Lưu ý: Quá trình này sẽ tự động sinh ra các file `.faiss` và `.sqlite` lưu vào `data/indexes/`.*

### Bước 4: Khởi động API và Web (Online Pipeline)
Sau khi trích xuất xong (hoặc đã tải DB từ Kaggle về), hãy bật server:
```bash
python api/main.py
```
API sẽ chạy ở `http://localhost:8000`. Bây giờ hãy qua thư mục `web/` để bật Frontend là bạn có thể bắt đầu tìm kiếm!
