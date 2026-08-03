# Web Frontend (UI) - AIC Video Retrieval

Giao diện người dùng cho hệ thống tìm kiếm Video được xây dựng bằng React, TypeScript và Vite.

## Tính năng (Features)

- **Thanh tìm kiếm:** Hỗ trợ nhập query tự nhiên bằng tiếng Việt.
- **Tùy chỉnh chế độ OCR:** Cho phép bật/tắt chế độ OCR (Agent Parse) hoặc (Thủ công) ngay trên UI.
- **Hiển thị kết quả:** Lưới video (grid) hiển thị frame hình, điểm số (score), metadata, và các đối tượng (objects/ocr) tìm thấy trong frame.
- **Video Player:** Click vào kết quả sẽ mở popup (modal) để xem video tại chính xác thời điểm (timestamp) của frame đó.

## Hướng dẫn cài đặt và chạy (Setup)

Yêu cầu: Đã cài đặt [Node.js](https://nodejs.org/).

1. **Cài đặt thư viện (Install Dependencies):**
```bash
npm install
```

2. **Chạy Môi trường Phát triển (Development Server):**
```bash
npm run dev
```

3. **Build bản Production:**
```bash
npm run build
```

## Lưu ý kết nối với Backend

Mặc định, ứng dụng frontend này sẽ gọi API tới Backend qua địa chỉ: `http://localhost:8000`.
Đảm bảo bạn đã khởi chạy `video_retrieval` API server trước khi tìm kiếm trên UI.
