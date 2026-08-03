import json

file_path = "scripts/kaggle_pipeline.ipynb"
with open(file_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

for cell in notebook.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        if any("kaggle_indexes" in line for line in source) or any("data_bundle" in line for line in source):
            new_source = [
                "# 6. Đóng gói thư mục data/indexes và data/frames thành 2 file ZIP riêng biệt để download\n",
                "import shutil\n",
                "import os\n",
                "print(\"Đang nén thư mục data/indexes (chứa SQLite, Milvus Lite, BM25)...\")\n",
                "shutil.make_archive(\"/kaggle/working/kaggle_indexes\", \"zip\", \"data/indexes\")\n",
                "print(\"✅ Hoàn tất nén data/indexes!\")\n",
                "if os.path.exists(\"data/frames\"):\n",
                "    print(\"\\nĐang nén thư mục data/frames (chứa ảnh, file có thể rất nặng)...\")\n",
                "    shutil.make_archive(\"/kaggle/working/kaggle_frames\", \"zip\", \"data/frames\")\n",
                "    print(\"✅ Hoàn tất nén data/frames!\")\n",
                "print(\"\\n✅ Đã xong! Hãy tải kaggle_indexes.zip và kaggle_frames.zip ở cột bên phải về máy tính.\")\n"
            ]
            cell["source"] = new_source

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
