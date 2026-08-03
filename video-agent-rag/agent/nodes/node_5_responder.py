from langchain_core.messages import AIMessage
from agent.state import VideoRetrievalState
from core.logger import get_logger

log = get_logger(__name__)


def _format_video_id(frame_id: str) -> str:
    """Extract a human-friendly video name from the frame_id string."""
    # frame_id format is typically: "video_name/frame_XXXXX"
    parts = frame_id.split("/")
    if len(parts) >= 2:
        return parts[0]
    return frame_id


async def responder_node(state: VideoRetrievalState) -> dict:
    log.info("--- [NODE] 5_responder ---")

    # --- Case 1: Pure conversational reply ---
    if state.get("intent") == "chat":
        response_text = state.get("chat_response") or "Xin chào! Mình có thể giúp gì cho bạn?"
        log.info("  Mode: chat")
        return {"messages": [AIMessage(content=response_text)]}

    # --- Case 2: Search result response ---
    results = state.get("retrieved_videos", [])
    confidence = state.get("confidence_score", 0.0)

    if not results:
        content = (
            "😔 Xin lỗi bạn, mình không tìm được video nào khớp với mô tả đó.\n\n"
            "Bạn có thể thử lại với:\n"
            "• Mô tả màu sắc cụ thể hơn (ví dụ: xe đỏ, áo xanh)\n"
            "• Chữ viết xuất hiện trên màn hình (biển số, bảng hiệu)\n"
            "• Hành động hoặc sự kiện cụ thể đang xảy ra"
        )
    else:
        top_3 = results[:3]

        # Group by video for a cleaner display
        video_groups: dict = {}
        for r in top_3:
            vid = _format_video_id(r.get("frame_id", "Unknown"))
            if vid not in video_groups:
                video_groups[vid] = []
            video_groups[vid].append(r)

        content = f"✅ Mình đã tìm thấy **{len(results)} kết quả** phù hợp! Dưới đây là top {len(top_3)} frame nổi bật:\n\n"

        for i, res in enumerate(top_3, 1):
            frame_id = res.get("frame_id", "N/A")
            video_name = _format_video_id(frame_id)
            score = res.get("score", 0.0)
            content += f"**{i}.** 🎬 Video: `{video_name}`\n"
            content += f"   📌 Frame: `{frame_id}` | Điểm: `{score:.4f}`\n\n"

        if len(results) > 3:
            content += f"_(Nhấn 'Mở cửa sổ 200 kết quả' để xem tất cả {len(results)} kết quả)_"

    log.info(f"  Search Response provided. Results shown: {len(results)}, confidence: {confidence:.4f}")
    return {"messages": [AIMessage(content=content)]}
