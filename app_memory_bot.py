import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import json

# Load environment variables
load_dotenv()

# Initialize the model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.5)

# Setup Streamlit page
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 AI Chatbot with Memory")

# Initialize chat history in session_state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    system_message = SystemMessage(content="You are a helpful AI assistant who remembers previous conversation.")
    st.session_state.chat_history.append(system_message)

# User input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Save user message
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Get AI response
    response = llm.invoke(st.session_state.chat_history)

    # Save AI response
    st.session_state.chat_history.append(AIMessage(content=response.content))

# Display conversation
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

# Save chat history when user presses a button
if st.button("💾 Save Chat History"):
    history_data = []
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            role = "system"
        history_data.append({"role": role, "content": message.content})

    with open("chat_history.json", "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2, ensure_ascii=False)

    st.success("Chat history saved to 'chat_history.json' 📄✅")
