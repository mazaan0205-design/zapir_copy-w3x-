import streamlit as st
import requests
from PyPDF2 import PdfReader

st.set_page_config(page_title="W3S Agent Builder", layout="wide")

# --- SIDEBAR: Knowledge Management ---
with st.sidebar:
    st.title("⚙️ Agent Settings")
    
    st.subheader("1. Instructions")
    instructions = st.text_area("System Prompt:", "You are a professional assistant.", height=150)
    
    st.divider()
    
    st.subheader("2. Knowledge Base")
    uploaded_file = st.file_uploader("Upload PDF/TXT", type=["pdf", "txt"])
    
    if st.button("🚀 Sync to Database") and uploaded_file:
        with st.spinner("Processing..."):
            # Extract text from file
            raw_text = ""
            if uploaded_file.type == "application/pdf":
                pdf_reader = PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    raw_text += page.extract_text()
            else:
                raw_text = uploaded_file.read().decode("utf-8")
            
            # Send to FastAPI local server
            try:
                payload = {"text": raw_text}
                res = requests.post("http://127.0.0.1:8000/add_knowledge", data=payload)
                st.success(res.json()["message"])
            except:
                st.error("Connection Error: Is api.py running on port 8000?")

# --- MAIN: Chat Interface ---
st.title("🤖 W3S Live Chat")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# User Input
if user_input := st.chat_input("Ask me anything about your data..."):
    # Store and show user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get response from FastAPI
    with st.chat_message("assistant"):
        try:
            api_payload = {"user_query": user_input, "instructions": instructions}
            response = requests.post("http://127.0.0.1:8000/chat", data=api_payload)
            full_res = response.json()["response"]
            st.markdown(full_res)
            st.session_state.chat_history.append({"role": "assistant", "content": full_res})
        except:
            st.error("Backend API is not responding.")
