# Importing Required Modules
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_community.document_loaders import PyPDFDirectoryLoader
import gradio as gr
from dotenv import load_dotenv
load_dotenv()
import os

# Setting the Mistral API Key and Initializing the Language Model
mistral_llm = ChatMistralAI(model="mistral-small", temperature=0, api_key=os.getenv("MISTRAL_API_KEY"))

# Loading Documents 
loader = PyPDFDirectoryLoader("/documents")
documents = loader.load()

# Splitting Documents into Chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# Creating Embeddings and Initializing Vector Store
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(docs, embedding=embedding)
retriever = vectordb.as_retriever()

# Initializing Conversation Memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)


# Creating the Conversational Retrieval Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=mistral_llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    output_key="answer"
)

# Building the Gradio Chatbot Interface
def respond_to_user(message, history):
    try:
        message_lower = message.lower().strip()
        greetings = ["hi", "hello", "hey", "good morning", "good evening", "what's up", "how are you"]
        exit_phrases = ["bye", "goodbye", "see you later", "exit"]

        if message_lower in greetings:
            return "🧚‍♀️ Hello! Ask me anything about fairy tales and I’ll do my best to help!"
        if message_lower in exit_phrases:
            return "👋 Bye! Have a magical day! 🌟"

        response = qa_chain.invoke({"question": message})
        answer = response.get("answer", "") or response.get("result", "")

        if "don't know" in answer.lower() or "not sure" in answer.lower():
          answer += " 😊 I'm sorry, I don't know the answer to this question. But I'm always learning!"

        return answer + " \n🧙‍♀️Thanks for asking!\n Do you want to ask anything else?"

        #return response["answer"]
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return f"Error:\n{str(e)}"


# Creating the Gradio UI for the Fairy Tale Chatbot
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🧚‍♀️ Welcome to Your Magical Fairy Tale Chatbot!")
    gr.Markdown("Talk to classic fairy tales like never before ✨ Ask about plots, characters, morals, and more.")
    gr.Image("/images/bg.gif", height=278, width = 500)
    gr.ChatInterface(
        fn=respond_to_user,
        title="🧚 Fairy Tale RAG Chatbot",
        description="Ask anything about your favourite fairy tales!",
        examples=["Does the little mermaid sing?", "Who helped Rapunzel escape?"],
        type="messages"
        )
demo.launch(share=True, inline=False)


