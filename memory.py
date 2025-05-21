import openai

class MemoryManager:
    """
    Manages conversation memory with capability to summarize older messages to prevent context overflow.
    """
    def __init__(self, max_messages=10):
        # Recent message history (as a list of {"role": ..., "content": ...})
        self.messages = []
        # Stored summary of older interactions (string, inserted as needed)
        self.summary = None
        self.max_messages = max_messages  # threshold to trigger summarization

    def add_message(self, role, content):
        """
        Add a message (user or assistant) to memory.
        If memory exceeds max_messages, older messages are summarized.
        """
        self.messages.append({"role": role, "content": content})
        # Trigger summarization if too many messages
        if len(self.messages) > self.max_messages:
            self._summarize_older_messages()

    def _summarize_older_messages(self):
        """
        Summarize older messages using the OpenAI API (or fallback) and store the summary.
        Only the latest 2 messages are kept in detail, the rest are replaced by the summary.
        """
        if len(self.messages) <= 2:
            return  # Not enough messages to summarize
        # Separate messages to summarize vs. messages to keep
        to_summarize = self.messages[:-2]
        remaining = self.messages[-2:]
        # Prepare a text transcript of the conversation to summarize
        convo_text = ""
        for msg in to_summarize:
            if msg["role"] == "user":
                convo_text += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                convo_text += f"Assistant: {msg['content']}\n"
        summary_prompt = (
            "Summarize the following conversation between a user and an assistant, focusing on key points:\n" 
            + convo_text
        )
        try:
            # Ensure API key is set externally
            if not openai.api_key:
                raise Exception("OpenAI API key is not set.")
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=150,
                temperature=0.2
            )
            summary_text = response['choices'][0]['message']['content'].strip()
        except Exception as e:
            # Fallback: simple manual summary if API not available
            summary_text = ""
            for msg in to_summarize:
                if msg["role"] == "user":
                    summary_text += f"User said: {msg['content'][:50]}... "
                elif msg["role"] == "assistant":
                    summary_text += f"Assistant replied: {msg['content'][:50]}... "
            summary_text = summary_text[:200] + "..."
        # Update or create the summary
        if self.summary:
            self.summary += " " + summary_text
        else:
            self.summary = summary_text
        # Reset messages to only keep the last two (recent context)
        self.messages = remaining
        # Insert the summary into the message list as a system-level context
        if self.summary:
            self.messages.insert(0, {"role": "system", "content": f"Summary of previous conversation: {self.summary}"})

    def get_context(self):
        """
        Get the current context messages (including summary if present) for building the LLM prompt.
        Returns a list of message dicts.
        """
        # Return a copy of messages (to avoid external modification)
        return list(self.messages)
