"""
rag_app.py
PDF Q&A Chatbot with Embedding Fallback (Google Gemini → HuggingFace)
"""

import asyncio
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile
import os
import re

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(page_title="📚 PDF Q&A Chatbot", page_icon="🤖")
st.title("📚 Ask Questions from Your PDF")

# --- LLM and Embeddings ---
@st.cache_resource
def load_llm_and_embeddings():
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.3, streaming=True)

    try:
        st.info("🔹 Using Google Gemini embeddings...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    except Exception as e:
        st.warning(f"⚠️ Google embeddings unavailable ({e}). Switching to HuggingFace.")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    return llm, embeddings

llm, embeddings = load_llm_and_embeddings()

# --- Detect chapter title ---
def detect_chapter(text):
    match = re.search(r'(Chapter|CHAPTER|Section|SECTION)\s+\d+[:.\s-]*(.+)', text)
    if match:
        return match.group(0).strip()
    return None

# --- Build vectorstore ---
@st.cache_resource
def build_vectorstore_from_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    documents = []
    last_chapter = "Unknown"

    for page in pages:
        chapter_title = detect_chapter(page.page_content)
        if chapter_title:
            last_chapter = chapter_title
        page.metadata["chapter"] = last_chapter
        documents.append(page)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    try:
        vectorstore = FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            st.warning("⚠️ Google quota exceeded. Switching to HuggingFace embeddings...")
            alt_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(chunks, alt_embeddings)
        else:
            raise e

    return vectorstore

# --- File upload ---
uploaded_file = st.file_uploader("📄 Upload a PDF", type=["pdf"])
if uploaded_file and "vectorstore" not in st.session_state:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    st.session_state.vectorstore = build_vectorstore_from_pdf(tmp_path)
    st.success("✅ PDF processed and ready for Q&A!")

# --- Chat memory ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("🔄 Reset Chat"):
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat loop ---
if "vectorstore" in st.session_state:
    if prompt := st.chat_input("💬 Ask a question about your PDF"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        retriever = st.session_state.vectorstore.as_retriever()
        relevant_docs = retriever.get_relevant_documents(prompt)

        context_parts = []
        for doc in relevant_docs:
            page = doc.metadata.get("page", "N/A")
            chapter = doc.metadata.get("chapter", "Unknown")
            content = f"[Page: {page}, Chapter: {chapter}]\n{doc.page_content}"
            context_parts.append(content)
        context = "\n\n".join(context_parts)

        response = llm.invoke(f"Context:\n{context}\n\nQuestion: {prompt}")

        st.chat_message("assistant").markdown(response.content)
        st.session_state.messages.append({"role": "assistant", "content": response.content})
