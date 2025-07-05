class Chatbot:
    def __init__(self, model_handler, vector_store):
        self.model = model_handler
        self.vector_store = vector_store
        self.chat_history = []
    
    def get_response(self, question):
        # Get relevant context
        relevant_chunks = self.vector_store.search(question)
        context = "\n".join(relevant_chunks)
        
        # Generate response using the model
        response = self.model.generate_response(question, context)
        
        # Store in chat history
        self.chat_history.append({"question": question, "response": response})
        
        return response
