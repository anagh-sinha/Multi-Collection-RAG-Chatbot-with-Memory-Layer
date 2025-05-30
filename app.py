import os
import json
from memory import MemoryManager
from retrieve import Retriever
from llm_agent import LLMChatbot

def load_json(file_path):
    """Utility to load a JSON file (returns None if file not found)."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None

if __name__ == "__main__":
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    index_file = os.path.join(data_dir, "index.json")
    # If index is not built yet, run the ingestion process
    if not os.path.exists(index_file):
        try:
            import ingest
            ingest.main()
        except Exception as e:
            print(f"Error during ingestion: {e}")
            exit(1)

    # Load profile and past chat history data
    profile_data = load_json(os.path.join(data_dir, "user_profile.json"))
    chat_history = load_json(os.path.join(data_dir, "chat_history.json"))

    # Initialize memory and preload any past chat history into it
    memory = MemoryManager()
    if chat_history:
        for msg in chat_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            memory.add_message(role, content)
    # Initialize the retriever with the indexed data
    retriever = Retriever(index_file)

    # Create the LLM-based chatbot agent
    agent = LLMChatbot(memory, retriever, user_profile=profile_data)

    # Greet the user
    if profile_data and profile_data.get("name"):
        name = profile_data["name"]
        print(f"Assistant: Hello {name}, I'm your sleep assistant. How can I help you today?")
    else:
        print("Assistant: Hello, I'm your AI sleep assistant. How can I help you today?")
    
    # Chat loop
    try:
        while True:
            user_input = input("You: ")
            if not user_input.strip():
                continue
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Assistant: Goodbye! Take care.")
                break

            # Get assistant response and print it
            assistant_reply = agent.get_response(user_input)
            print(f"Assistant: {assistant_reply}")
    except KeyboardInterrupt:
        print("\nAssistant: (Session ended)")
