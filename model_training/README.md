# Model Training Workspace

Fine-tuning code cho từng model trong hệ thống Video Retrieval.

## Cấu trúc

```
model_training/
├── siglip2/         ← Fine-tune SigLIP2 (CLIP branch)
├── bge_m3/          ← Fine-tune BGE-M3 (OCR embedding branch)
├── yolov11/         ← Fine-tune YOLOv11 (Object detection)
└── paddle_ocr/      ← Fine-tune PaddleOCR (OCR recognition)
```

## Workflow

```
1. Chuẩn bị dữ liệu (mỗi model có format riêng)
2. Sửa config.yaml trong folder model tương ứng
3. Chạy train.py
4. Chạy export_config.py → nhận đoạn config
5. Paste config vào video_retrieval/.env
```

## Ví dụ nhanh

```bash
# Fine-tune SigLIP2
cd siglip2
pip install -r requirements.txt
python train.py --config config.yaml

# Sau khi train xong
python export_config.py --checkpoint ./outputs/best
# → In ra đoạn config để paste vào video_retrieval/.env
```
