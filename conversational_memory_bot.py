from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import json

# Load environment variables
load_dotenv()

# Initialize the model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.5)

# Initialize chat history list
chat_history = []

# Set an initial system message
system_message = SystemMessage(content="You are a helpful AI assistant who remembers previous conversation.")
chat_history.append(system_message)

# Chat loop
print("Bot: Hello! How can I assist you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Bot: Goodbye!")
        break

    # Save user input to chat history
    chat_history.append(HumanMessage(content=user_input))

    # Get AI response using the full chat history
    response = llm.invoke(chat_history)

    # Save AI response to chat history
    chat_history.append(AIMessage(content=response.content))

    # Print AI response
    print(f"Bot: {response.content}")


# # Print the full conversation at the end (optional)
# print("\n---- Full Chat History ----")
# for message in chat_history:
#     role = "User" if isinstance(message, HumanMessage) else "Bot" if isinstance(message, AIMessage) else "System"
#     print(f"{role}: {message.content}")


# Save as structured .json
history_data = []
for message in chat_history:
    if isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        role = "assistant"
    else:
        role = "system"
    history_data.append({"role": role, "content": message.content})

with open("chat_history.json", "w", encoding="utf-8") as f:
    json.dump(history_data, f, indent=2, ensure_ascii=False)

print("\nChat history saved to 'chat_history.json' 📄✅")