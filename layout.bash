naptick-ai-challenge/
├── Task1_RAG_Chatbot/
│   ├── data/
│   │   ├── wearable_data.json         # Dummy wearable metrics
│   │   ├── chat_history.json          # Sample past conversations
│   │   ├── user_profile.json          # User demographic & preferences
│   │   ├── location_data.json         # Simulated geotags & places
│   │   └── custom_collection.json     # Domain-specific docs
│   ├── src/
│   │   ├── ingest.py                  # Data ingestion into vector store
│   │   ├── retrieve.py                # Retrieval logic across collections
│   │   ├── memory.py                  # Conversation memory module
│   │   ├── llm_agent.py               # LLM prompt & generation
│   │   └── app.py                     # Streamlit chat interface
│   ├── requirements.txt               # Python dependencies
│   └── README.md                      # Setup & usage instructions
│
├── Task2_Voice_Coach/
│   ├── data/
│   │   ├── sleep_diary.csv            # Dummy sleep logs
│   │   ├── wearable_sleep.csv         # Simulated wearable sleep metrics
│   │   └── coaching_dialogues.json    # Fine-tuning Q/A pairs
│   ├── src/
│   │   ├── stt.py                     # Whisper speech-to-text wrapper
│   │   ├── coach.py                   # LLM-based coaching logic
│   │   ├── tts.py                     # Text-to-speech with Coqui TTS
│   │   └── voice_app.py               # CLI interface for voice I/O
│   ├── requirements.txt               # Python dependencies
│   └── README.md                      # Setup & usage instructions
│
└── PDF_Writeup.pdf                    # Short write-up summary
