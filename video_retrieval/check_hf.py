import os
from pathlib import Path

p = Path.home() / '.cache' / 'huggingface' / 'hub' / 'models--google--siglip2-so400m-patch14-384'
print(f"Checking Path: {p}")
if p.exists():
    total_size = sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
    print(f"Downloaded Size: {total_size / (1024*1024):.2f} MB")
else:
    print("Folder not found. It might be downloading to a temp folder or has not started writing.")
