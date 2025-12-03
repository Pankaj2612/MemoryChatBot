import streamlit as st
from module.ai_api import (
    extract_user_memory,
    generate_raw_reply,
    generate_memory_aware_reply,
)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Memory-Aware Chatbot",
    layout="wide",
)

# --- HEADER ---
st.markdown(
    """
<div style="text-align:center; margin-bottom: 20px;">
    <h1>🧠 Memory-Aware Chatbot</h1>
    <p style="color: #666; font-size: 18px;">
        Build user memory from chat history & compare raw vs personality-aware responses side-by-side.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# Session state setup
if "user_memory" not in st.session_state:
    st.session_state.user_memory = None

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# --- SIDEBAR ---
st.sidebar.markdown("## Provide Chat History")
chat_history_input = st.sidebar.text_area(
    "Paste previous chat messages (one per line):",
    height=250,
    placeholder="Hi...\nI like sarcastic chatbots...\nI enjoy dark humour...",
)

build_memory_btn = st.sidebar.button("Build Memory")

if build_memory_btn and chat_history_input.strip():
    messages = [line.strip() for line in chat_history_input.split("\n") if line.strip()]
    st.session_state.chat_messages = messages
    st.session_state.user_memory = extract_user_memory(messages)
    st.sidebar.success("🧩 Memory extracted successfully!")

# --- MAIN SECTION ---
st.markdown("## Chat")
user_question = st.text_input(
    "Enter your question/message:", placeholder="Ask something..."
)

if st.button("Generate Responses"):
    if not st.session_state.user_memory:
        st.warning(" Please provide chat history and build memory first!")
    else:
        with st.spinner("🤖 Thinking..."):
            raw_reply = generate_raw_reply(user_question)
            memory_reply = generate_memory_aware_reply(
                st.session_state.user_memory, user_question
            )

        st.markdown("---")
        st.markdown("### Responses")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Raw Response")
            st.info(raw_reply)

        with col2:
            st.markdown("####  Memory-Aware Response")
            st.success(memory_reply)

# --- SHOW MEMORY ---
if st.session_state.user_memory:
    st.markdown("---")
    st.markdown("###  Extracted Memory")
    with st.expander("Click to view extracted memory"):
        st.json(st.session_state.user_memory)
