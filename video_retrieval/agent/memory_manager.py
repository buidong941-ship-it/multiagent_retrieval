"""
Phase 11: Memory Manager
Quản lý 3 tầng bộ nhớ:
- Working Memory: Trạng thái tạm thời của 1 truy vấn. Xóa sau khi xong.
- Session Memory: Ngữ cảnh hội thoại. Lưu qua nhiều truy vấn.
- Knowledge Memory: Dữ liệu dùng chung, cấu hình, caches. Không bị xóa.
"""
from typing import Any, Dict

class MemoryManager:
    def __init__(self):
        self.working_memory: Dict[str, Any] = {}
        self.session_memory: Dict[str, Any] = {}
        self.knowledge_memory: Dict[str, Any] = {}

    def set_working(self, key: str, value: Any):
        self.working_memory[key] = value

    def get_working(self, key: str, default: Any = None) -> Any:
        return self.working_memory.get(key, default)

    def clear_working_memory(self):
        self.working_memory.clear()

    def set_session(self, key: str, value: Any):
        self.session_memory[key] = value

    def get_session(self, key: str, default: Any = None) -> Any:
        return self.session_memory.get(key, default)
    
    def clear_session_memory(self):
        self.session_memory.clear()

    def set_knowledge(self, key: str, value: Any):
        self.knowledge_memory[key] = value

    def get_knowledge(self, key: str, default: Any = None) -> Any:
        return self.knowledge_memory.get(key, default)
