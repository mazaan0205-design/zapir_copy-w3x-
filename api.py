import os
from fastapi import FastAPI, Form
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# Initialize Embeddings and Vector DB
# This creates a folder called 'chroma_db' in your directory
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

@app.post("/add_knowledge")
async def add_knowledge(text: str = Form(...)):
    # Break text into 1000-character chunks with a little overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_text(text)
    
    # Clean old data and add new
    vector_db.add_texts(chunks)
    return {"status": "Success", "message": f"Knowledge synchronized: {len(chunks)} chunks created."}

@app.post("/chat")
async def chat(user_query: str = Form(...), instructions: str = Form(...)):
    # 1. RETRIEVAL: Find the 3 most relevant chunks from ChromaDB
    docs = vector_db.similarity_search(user_query, k=3)
    context = "\n---\n".join([d.page_content for d in docs])

    # 2. AUGMENTATION: Build the context-aware prompt
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
    
    system_prompt = f"""
    INSTRUCTIONS: {instructions}
    
    USE THIS DATA TO ANSWER:
    {context}
    
    If the answer isn't in the data, say you don't know based on the knowledge base.
    """
    
    # 3. GENERATION: Get response from Groq
    response = llm.invoke([
        SystemMessage(content=system_prompt), 
        HumanMessage(content=user_query)
    ])
    
    return {"response": response.content}
