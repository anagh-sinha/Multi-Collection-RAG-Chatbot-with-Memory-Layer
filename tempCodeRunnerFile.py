 try:
        while True:
            user_input = input("You: ")
            if not user_input.strip():
                continue
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Assistant: Goodbye! Take care.")
                break