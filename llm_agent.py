import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY") # Set the OpenAI API key from the environment variable by using the os.getenv() function.

class LLMChatbot:
    """
    An LLM-driven chatbot that uses a Retriever for knowledge and a MemoryManager for context.
    """
    def __init__(self, memory_manager, retriever, user_profile=None): #self is used to refer to the instance of the class. User profile is optional. init is a constructor.
        self.memory = memory_manager
        self.retriever = retriever
        # Build a base system prompt from user profile data for personalization
        if user_profile:
            name = user_profile.get("name")
            issues = user_profile.get("sleep_issues")
            profile_info = ""
            if name:
                profile_info += f"The user's name is {name}. "
            if issues:
                profile_info += f"The user has sleep issues: {issues}. "
            self.base_prompt = (
                "You are a helpful AI sleep coaching assistant. " +
                profile_info +
                "Be empathetic and provide clear, practical advice."
                "Always address by name and use the user's name in your responses."
            )
        else:
            self.base_prompt = (
                "You are a helpful AI assistant and sleep coach. "
                "Be empathetic and provide clear, practical advice to the user."
                "Always address by name."
            )
        # Ensure OpenAI API key is configured (read from env if not passed explicitly)
        if not openai.api_key:
            print("Warning: OpenAI API key not set. Please set OPENAI_API_KEY environment variable.")

    def get_response(self, user_input): 
        """
        Process the user's input: update memory, retrieve relevant info, and generate a response via OpenAI.
        """
        # Record the user's message in memory
        self.memory.add_message("user", user_input)
        # Retrieve relevant knowledge snippets from our data
        knowledge_results = self.retriever.get_relevant(user_input, top_k=3)            
        # Format retrieved info into a context string
        context_str = ""
        if knowledge_results: # If there are any knowledge results
            context_str = "Relevant information:\n"
            for res in knowledge_results: # Iterate through the knowledge results   
                src = res.get("source", "info") # Get the source from the knowledge result
                text = res.get("text", "") # Get the text from the knowledge result
                context_str += f"- ({src}) {text}\n" # Add the source and text to the context string
        # Get recent conversation context (including any summary) from memory
        memory_context = self.memory.get_context()
        # Construct the messages for the OpenAI ChatCompletion
        messages = []
        # Base system instruction (assistant persona and user profile info)
        messages.append({"role": "system", "content": self.base_prompt})                        
        # Add retrieved knowledge as an additional system message if available
        if context_str:
            messages.append({"role": "system", "content": context_str.strip()})
        # Add conversation history from memory (excluding any summary already included above)
        for msg in memory_context:
            # Skip summary message here to avoid duplication, since context_str covers knowledge
            if msg["role"] == "system" and msg["content"].startswith("Summary"):
                continue
            messages.append(msg)
        # Add the current user question at the end
        messages.append({"role": "user", "content": user_input})
        # Query the LLM (OpenAI ChatCompletion)
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            assistant_reply = response['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"Error getting response from LLM: {e}")
            assistant_reply = "I'm sorry, I'm unable to answer that right now."
        # Record the assistant's reply in memory
        self.memory.add_message("assistant", assistant_reply)
        return assistant_reply
