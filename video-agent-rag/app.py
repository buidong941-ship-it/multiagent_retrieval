import streamlit as st
from langchain_core.messages import HumanMessage
import sys
import os

# Add both the agent root and video_retrieval to Python path
agent_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
video_retrieval_root = os.path.join(agent_root, 'video_retrieval')

if agent_root not in sys.path:
    sys.path.insert(0, agent_root)
if video_retrieval_root not in sys.path:
    sys.path.insert(0, video_retrieval_root)

# Import the LangGraph agent we just built (updated path)
from agent.graph import build_graph

# Configure Streamlit page - Light theme and wide layout
st.set_page_config(page_title="Agentic Video Retrieval", page_icon="🌤️", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for a lively, light theme
st.markdown("""
<style>
    /* Header styling */
    h1 {
        color: #ff6b6b !important;
        font-family: 'Comic Sans MS', 'Arial', sans-serif;
        text-shadow: 1px 1px 2px #cccccc;
    }
    
    /* Input area */
    .stChatInputContainer {
        border-radius: 20px !important;
        border: 2px solid #1890ff !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌤️ Agentic Video Retrieval Chat")
st.markdown("💬 Trò chuyện với Agent (Qwen 2.5) để tìm kiếm video thông minh.")

# Custom HTML renderers for chat bubbles
def render_user_msg(content):
    st.markdown(f"""
    <div style='display: flex; justify-content: flex-end; width: 100%; margin-bottom: 15px;'>
        <div style='background-color: #0084FF; color: #FFFFFF; padding: 12px 18px; border-radius: 20px 20px 5px 20px; max-width: 75%; font-family: system-ui, sans-serif; font-size: 16px; box-shadow: 0px 3px 6px rgba(0,0,0,0.1); line-height: 1.5;'>
            {content}
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_agent_msg(content):
    st.markdown(f"""
    <div style='display: flex; justify-content: flex-start; width: 100%; margin-bottom: 15px;'>
        <div style='background-color: #FFFFFF; color: #1E1E1E; padding: 12px 18px; border-radius: 20px 20px 20px 5px; max-width: 75%; font-family: system-ui, sans-serif; font-size: 16px; box-shadow: 0px 3px 6px rgba(0,0,0,0.08); border: 1px solid #E5E5EA; line-height: 1.5;'>
            {content}
        </div>
    </div>
    """, unsafe_allow_html=True)

@st.dialog("Tất cả 200 kết quả", width="large")
def show_all_results(results):
    cols = st.columns(4)
    for idx, res in enumerate(results):
        with cols[idx % 4]:
            frame_path = res.get("frame_path")
            if frame_path and os.path.exists(frame_path):
                st.image(frame_path, caption=f"Top {idx+1} | Score: {res.get('score', 0):.2f}", use_column_width=True)
            else:
                st.info(f"Frame ID: {res.get('frame_id')}")

import asyncio

# 1. Initialize LangGraph app + a PERSISTENT event loop in session state.
#    Reusing the same loop prevents aiosqlite/aiofiles Queue conflicts across turns.
if "graph_app" not in st.session_state:
    st.session_state.graph_app = build_graph()
    st.session_state.config = {"configurable": {"thread_id": "streamlit_chat_session"}}
    # Create ONE event loop for this session and never replace it
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    st.session_state.event_loop = _loop

# 2. Initialize UI chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Render previous messages
for msg_idx, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        render_user_msg(message["content"])
    else:
        render_agent_msg(message["content"])
        
        # If this assistant message has retrieved videos, display them
        if "videos" in message and message["videos"] and message.get("is_clear", False):
            st.markdown("### 🎥 Top 3 Kết quả:")
            cols = st.columns(3)
            for idx, res in enumerate(message["videos"][:3]):
                with cols[idx % 3]:
                    frame_path = res.get("frame_path")
                    if frame_path and os.path.exists(frame_path):
                        st.image(frame_path, caption=f"{res.get('frame_id')} (Điểm: {res.get('score', 0):.2f})", use_column_width=True)
                    else:
                        st.info(f"Frame ID: {res.get('frame_id')} (Chưa có ảnh)")
                        
            # Button to open the full 200 results dialog
            if st.button("Mở cửa sổ 200 kết quả chi tiết", key=f"btn_200_{msg_idx}"):
                show_all_results(message["videos"])

# 4. Handle new user input
if prompt := st.chat_input("Nhập yêu cầu tìm kiếm của bạn... (VD: Tìm xe ô tô màu đỏ)"):
    # Display user message
    render_user_msg(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Call the Agent
    with st.spinner("Agent đang phân tích và tìm kiếm... (Xin chờ, có thể mất vài giây nếu Model đang load)"):
        try:
            user_msg = HumanMessage(content=prompt)

            async def run_graph():
                async for event in st.session_state.graph_app.astream(
                    {"messages": [user_msg]}, st.session_state.config
                ):
                    pass
                return await st.session_state.graph_app.aget_state(st.session_state.config)

            # Reuse the SAME event loop created at session startup.
            # This prevents aiosqlite/Queue 'bound to different event loop' errors.
            loop: asyncio.AbstractEventLoop = st.session_state.event_loop
            asyncio.set_event_loop(loop)
            final_state = loop.run_until_complete(run_graph())
            agent_reply = final_state.values["messages"][-1].content
            
            # Display and save text
            render_agent_msg(agent_reply)
            
            retrieved_videos = final_state.values.get("retrieved_videos", [])
            is_clear = final_state.values.get("is_clear", False)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": agent_reply,
                "videos": retrieved_videos,
                "is_clear": is_clear
            })
            
            # Immediately render the images for this turn (so it shows before next rerun)
            if retrieved_videos and is_clear:
                st.markdown("### 🎥 Top 3 Kết quả:")
                cols = st.columns(3)
                for idx, res in enumerate(retrieved_videos[:3]):
                    with cols[idx % 3]:
                        frame_path = res.get("frame_path")
                        if frame_path and os.path.exists(frame_path):
                            st.image(frame_path, caption=f"{res.get('frame_id')} (Điểm: {res.get('score', 0):.2f})", use_column_width=True)
                        else:
                            st.info(f"Frame ID: {res.get('frame_id')} (Chưa có ảnh)")
                            
                # For the immediate render, we can't safely use st.button because it will disappear on next input
                # However, Streamlit will trigger a rerun immediately after this block finishes if needed.
                # Actually, calling st.rerun() here is safest to sync the UI with the history loop.
                st.rerun()
                
        except Exception as e:
            st.error(f"Có lỗi xảy ra: {e}")
