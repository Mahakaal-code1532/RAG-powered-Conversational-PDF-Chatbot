import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import tempfile
import os
import re
import json

# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(page_title="📚 PDF Q&A Chatbot", page_icon="🤖")
st.title("📚 Ask Questions from Your PDF")

# Load LLM and embeddings once
@st.cache_resource
def load_llm_and_embeddings():



    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.3, streaming=True)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return llm, embeddings

llm, embeddings = load_llm_and_embeddings()

# Function to detect chapter title from text
def detect_chapter(text):
    match = re.search(r'(Chapter|CHAPTER|Section|SECTION)\s+\d+[:.\s-]*(.+)', text)
    if match:
        return match.group(0).strip()
    return None

# Build vectorstore and add metadata (page number + chapter heading)
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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# Upload PDF and process
uploaded_file = st.file_uploader("📄 Upload a PDF", type=["pdf"])
if uploaded_file and "vectorstore" not in st.session_state:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    st.session_state.vectorstore = build_vectorstore_from_pdf(tmp_path)
    st.success("✅ PDF processed and ready for Q&A!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Reset chat
if st.button("🔄 Reset Chat"):
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input and response
if "vectorstore" in st.session_state:
    if prompt := st.chat_input("💬 Ask a question about your PDF"):
        # Show user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Retrieve relevant documents
        retriever = st.session_state.vectorstore.as_retriever()
        relevant_docs = retriever.get_relevant_documents(prompt)

        # Format context with metadata
        context_parts = []
        for doc in relevant_docs:
            page = doc.metadata.get("page", "N/A")
            chapter = doc.metadata.get("chapter", "Unknown")
            content = f"[Page: {page}, Chapter: {chapter}]\n{doc.page_content}"
            context_parts.append(content)
        context = "\n\n".join(context_parts)

        # Generate prompt depending on context availability
        if context.strip():
            full_prompt = f"""
The following context was extracted from a PDF book:

{context}

Question: {prompt}

If the answer is clearly present in the context above, answer using it and mention it is from the book.
If the context does not contain the answer, respond from your general knowledge and clearly say the answer is not from the PDF.
""".strip()
        else:
            full_prompt = f"""
The PDF does not seem to contain relevant information.

Please answer the following question using your own general knowledge. Let the user know the answer is not from the PDF.

Question: {prompt}
""".strip()

        # Get answer
        response = llm.invoke(full_prompt)

        # Display AI message
        st.chat_message("assistant").markdown(response.content)
        st.session_state.messages.append({"role": "assistant", "content": response.content})

# Save chat history at the end
if st.session_state.get("messages"):
    history_data = []
    for msg in st.session_state.messages:
        history_data.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    with open("chat_history.json", "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2, ensure_ascii=False)
