
import streamlit as st
import os
from dotenv import load_dotenv

# Expert Tools: Shredding & Searching
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from PyPDF2 import PdfReader

# 1. Load environment variables
load_dotenv()

st.set_page_config(page_title="W3S Builder - Expert RAG", layout="wide")

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- THE "EXPERT" PROCESSING FUNCTION ---
def process_to_vector_store(uploaded_files):
    text = ""
    for file in uploaded_files:
        if file.type == "application/pdf":
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        else:
            text += file.read().decode("utf-8")
    
    # Shredding: Breaking text into 1000-character pieces (Expert move)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_text(text)
    
    # Embedding: Turning text into searchable math numbers
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Store: Keeping the pieces in memory
    vector_store = Chroma.from_texts(
        texts=splits, 
        embedding=embeddings, 
        collection_name="w3s_knowledge"
    )
    return vector_store

# --- SIDEBAR (The Builder Controls) ---
with st.sidebar:
    st.title("🛠️ W3S Builder")
    
    st.subheader("1. Instructions")
    instructions = st.text_area(
        "Define Agent Rules:", 
        value="You are a helpful assistant. Use the provided context to answer questions.", 
        height=200
    )
    
    st.divider()
    
    st.subheader("2. Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload reference files", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    
    if st.button("Sync Knowledge Base"):
        if uploaded_files:
            with st.spinner("Processing documents into chunks..."):
                st.session_state.vector_store = process_to_vector_store(uploaded_files)
                st.success("Knowledge Base is ready!")
        else:
            st.warning("Please upload files first.")

# --- MAIN SCREEN (The Chat) ---
st.title("🤖 W3S Personal Workspace")

# Display History
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Handle Chat
if user_input := st.chat_input("Ask about your documents..."):
    st.chat_message("user").markdown(user_input)
    
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("⚠️ No API Key found!")
        st.stop()

    with st.chat_message("assistant"):
        try:
            llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key)
            
            # THE RAG STEP: Pull only the 3 best chunks from memory[cite: 8]
            context = ""
            if st.session_state.vector_store:
                docs = st.session_state.vector_store.similarity_search(user_input, k=3)
                context = "\n---\n".join([d.page_content for d in docs])
            
            # Combine everything for the AI
            full_prompt = f"{instructions}\n\nRELEVANT CONTEXT:\n{context}"
            
            payload = [SystemMessage(content=full_prompt)] + st.session_state.messages + [HumanMessage(content=user_input)]
            
            response = llm.invoke(payload)
            st.markdown(response.content)
            
            # Update History
            st.session_state.messages.append(HumanMessage(content=user_input))
            st.session_state.messages.append(AIMessage(content=response.content))
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
