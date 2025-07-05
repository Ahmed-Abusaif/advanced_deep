import gradio as gr
from model_handler import DeepSeekHandler
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from chatbot import Chatbot

def main():
    model_handler = DeepSeekHandler()
    doc_processor = DocumentProcessor()
    vector_store = VectorStoreManager()
    chatbot = Chatbot(model_handler, vector_store)

    def upload_and_process(file):
        if file is not None:
            doc_chunks = doc_processor.process_document(file.name)
            vector_store.store_embeddings(doc_chunks)
            return "Document processed and stored!"
        return "Please upload a PDF file."

    def chat_interface(user_input):
        if user_input:
            response = chatbot.get_response(user_input)
            return response
        return "Please enter a question."

    with gr.Blocks() as demo:
        gr.Markdown("# Course Material Chatbot")
        with gr.Row():
            file_input = gr.File(label="Upload course material (PDF)", file_types=[".pdf"])
            upload_btn = gr.Button("Process Document")
            upload_output = gr.Textbox(label="Upload Status")
        with gr.Row():
            user_input = gr.Textbox(label="Ask a question about the course material:")
            chat_output = gr.Textbox(label="Chatbot Response")
            ask_btn = gr.Button("Ask")
        
        upload_btn.click(upload_and_process, inputs=file_input, outputs=upload_output)
        ask_btn.click(chat_interface, inputs=user_input, outputs=chat_output)

    demo.launch()

if __name__ == "__main__":
    main()
