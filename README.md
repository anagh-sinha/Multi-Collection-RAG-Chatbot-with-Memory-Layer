# Naptick-AI-Challenge
## Task 1: Multi-Collection RAG Chatbot with Memory Layer

This directory contains an implementation of a Retrieval-Augmented Generation (RAG) chatbot with multiple data sources and a memory layer. The chatbot retrieves information from various collections (wearable data, user profile, location info, custom knowledge base) and maintains conversational context through a summarization-based memory.

## Project Structure
Multi-Collection-RAG-Chatbot-with-Memory-Layer/
├── ingest.py # Ingest data and build the knowledge index
├── retrieve.py # Retrieve relevant info from indexed data
├── memory.py # Manage chat history with memory (summaries)
├── llm_agent.py # LLM-based agent combining retrieval and memory
├── app.py # Main application script for the chatbot
├── data/ # Sample data files for knowledge and context
│ ├── wearable_data.json
│ ├── chat_history.json
│ ├── user_profile.json
│ ├── location_data.json
│ └── custom_collection.json
└── requirements.txt # Python dependencies


## Setup and Installation
1. **Install Dependencies**: Navigate to this `Multi-Collection-RAG-Chatbot-with-Memory-Layer` directory in VS Code and run:  
   ```bash
   pip install -r requirements.txt

   ```
   This installs the OpenAI API client, SentenceTransformers, NumPy, etc.
   
2. **Configure OpenAI API Key**: Set your OpenAI API key as an environment variable, e.g.:
On Linux/Mac: export OPENAI_API_KEY="YOUR_API_KEY"
On Windows (PowerShell): $Env:OPENAI_API_KEY="YOUR_API_KEY"

This key is needed for the chatbot to call the OpenAI API for answers and memory summarization.

3. **Ingest Data**: Run the ingestion script to index the data:
 ```bash
python ingest.py
```

This reads all JSON files in the data/ folder and creates data/index.json used for retrieval. (Re-run this if you update the data files.

## Running the Chatbot
Start the chatbot by running
```bash
python app.py
```
This will initialize the bot (loading the index and any chat history) and then prompt you for input in the console. 

# Example:
Assistant: "Hello Alice, I'm your sleep assistant. How can I help you today?"
You can then type a question, e.g. "How did I sleep on 2025-05-19?"
Assistant: "On 2025-05-19, your wearable recorded about 6 hours 50 minutes of sleep with a score of 76. It wasn't your best night. Let's work on improving that..."
(The assistant pulled data from the wearable_data collection.)
The assistant's answers will combine information from your data and general sleep coaching knowledge. It also remembers earlier conversation. For instance, if you say "Give me some tips to sleep better," it might reply with advice that includes your profile or custom tips:
Assistant: "Sure. Maintaining good sleep hygiene is important. Avoid caffeine at least 6 hours before bedtime, and try to limit screen time before bed. Also, a consistent bedtime routine can help you fall asleep easier."

## To end the chat, type "exit", "quit", or "bye".


