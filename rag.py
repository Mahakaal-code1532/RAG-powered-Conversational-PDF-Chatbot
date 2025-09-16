""""
rag + play in terminal
"""

import asyncio
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# 1. Load environment variables
load_dotenv()

# 2. Initialize the LLM and Embeddings
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.3) 
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 3. Load PDF and Extract Text
pdf_path = "document/The Story of My Life, by Helen Keller.pdf"  # 📄 <--- Replace this with your actual file name
loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"✅ Loaded {len(documents)} pages")

# 4. Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
docs = text_splitter.split_documents(documents)

# 5. Embed and create vector store
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_index")  # saves locally

# 6. Ready to ask questions
print("\n✅ Setup completed. You can now ask questions related to the PDF!")


# Later, load again (skip embedding)
# vectorstore = FAISS.load_local("faiss_index", embeddings)


while True:
    query = input("\nYou: ")
    if query.lower() in ["exit", "quit"]:
        print("Bot: Goodbye!")
        break

    # 7. Search for relevant documents
    retriever = vectorstore.as_retriever()
    relevant_docs = retriever.get_relevant_documents(query)

    if not relevant_docs:
        print("Bot: Sorry, I couldn't find anything relevant.")
        continue

    # 8. Create context to pass to LLM
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""
    You are an AI assistant. Use the following extracted parts of a document to answer the question.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    # 9. Get Answer from LLM
    response = llm.invoke(prompt)

    # 10. Show answer
    print(f"Bot: {response.content}")
