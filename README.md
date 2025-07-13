# 📄 PDF Chatbot with LangChain + Mistral

This is an NLP project that builds a chatbot capable of answering questions from PDF documents using **LangChain**, **Mistral**, and **Chroma** vector store.

## 🚀 Features

- Upload and read PDFs
- Extract text using `PyPDF`
- Convert text to embeddings (Sentence Transformers)
- Create vector store with Chroma
- Ask questions through a simple UI
- Use Mistral / HuggingFace models via LangChain

## 🛠️ Tech Stack

- Python
- LangChain + Mistral
- Sentence Transformers
- Chroma
- Gradio
- PyPDF

## 🗂️ Folder Structure

FairyTaleChatbot/
├── index.ipynb # All code
├── requirements.txt # Python dependencies
├── README.md # Project guide
├── documents/ # Sample PDFs
├── images/ # Background images
└── .gitignore

## 🧪 Run the Project

```bash
# 1. Clone the repo
git clone https://github.com/hetvis-pro/pdf-chatbot.git
cd pdf-chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook app.ipynb

```
