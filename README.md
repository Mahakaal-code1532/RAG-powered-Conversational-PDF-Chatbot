# 📚 PDF Q&A Chatbot + Memory Chatbot

This project is a **Retrieval-Augmented Generation (RAG) chatbot** built
with [LangChain](https://www.langchain.com/),
[Streamlit](https://streamlit.io/), and Google's [Gemini
API](https://ai.google.dev/).\
It allows you to **upload a PDF** and ask natural language questions
(RAG bot), or simply **chat without uploading a file** (memory chatbot).

✨ Features: - Upload any PDF and ask questions (RAG bot). - Chat
without a PDF, with memory of past turns (Memory bot). - Uses **Google
Gemini** (chat + embeddings) by default. - Automatic **fallback to
HuggingFace embeddings** when Google API quota is exceeded. - Caches
embeddings with FAISS for faster re-use. - Preserves **page number** and
**chapter metadata** in PDF answers.

------------------------------------------------------------------------

## 🛠️ Installation

Clone the repo and install requirements:

``` bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
pip install -r requirements.txt
```

Additionally, install HuggingFace embeddings (needed for fallback):

``` bash
pip install sentence-transformers
```

------------------------------------------------------------------------

## ⚙️ Configuration

1.  Create a `.env` file in the project root:

``` env
GOOGLE_API_KEY="your_google_api_key_here"
```

-   You can create a Gemini API key from [Google AI
    Studio](https://aistudio.google.com/app/apikey).\
-   Free tier has strict quotas → HuggingFace fallback ensures chatbot
    keeps working.

2.  Check `requirements.txt` (important packages):

```{=html}
<!-- -->
```
    streamlit
    python-dotenv
    langchain
    langchain-community
    langchain-text-splitters
    langchain-google-genai
    faiss-cpu
    PyPDF2
    sentence-transformers

------------------------------------------------------------------------

## ▶️ Running the Apps

### 📚 RAG Chatbot (PDF Q&A)

From your terminal (PowerShell in VS Code):

``` powershell
streamlit run rag_app.py
```

This will open a local web app at <http://localhost:8501>.

-   Upload a PDF → Processed into chunks → Ask questions with
    page/chapter references.

### 🤖 Memory Chatbot (No PDF)

If you just want a chatbot that **remembers your past conversation**:

``` powershell
streamlit run app_memory_bot.py
```

This chatbot uses Gemini directly, without needing to upload a file.

------------------------------------------------------------------------

## 💻 Usage

1.  For **RAG chatbot**:
    -   Upload a PDF.
    -   Ask your questions in the chat box.
    -   Get answers with relevant context (page + chapter).
    -   Reset chat anytime.
2.  For **Memory chatbot**:
    -   Just start chatting → the bot remembers what you said earlier.

------------------------------------------------------------------------

## 🧠 How It Works

-   **RAG bot**:
    1.  Load PDF (`PyPDFLoader`)\
    2.  Split into chunks (`RecursiveCharacterTextSplitter`)\
    3.  Embed with Gemini or HuggingFace → store in FAISS\
    4.  Retrieve relevant chunks → pass to Gemini LLM\
    5.  Generate context-aware answers
-   **Memory bot**:
    -   Uses `st.session_state` to store conversation history\
    -   Gemini LLM responds considering the previous context

------------------------------------------------------------------------

## 📦 Project Structure

    ├── rag_app.py                  # Streamlit app (PDF Q&A chatbot)
    ├── rag.py                      # Terminal version (RAG bot)
    ├── app_memory_bot.py           # Streamlit chatbot with memory (no PDF)
    ├── conversational_memory_bot.py# Terminal chatbot with memory
    ├── requirements.txt            # Dependencies
    ├── .env                        # API keys
    └── faiss_index/                # Saved FAISS vectorstore (auto-created)

------------------------------------------------------------------------

## ⚠️ Notes

-   Keep your `.env` secret → never upload API keys to GitHub.\
-   HuggingFace fallback requires \~80 MB model download on first run.\
-   Free Gemini quotas:
    [Docs](https://ai.google.dev/gemini-api/docs/rate-limits).

------------------------------------------------------------------------

#
