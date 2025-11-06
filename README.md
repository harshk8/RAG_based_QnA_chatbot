# RAG_based_QnA_chatbot
AI Document Q&amp;A Chatbot using LangChain and Hugging Face

🧠 AI Document Q&A Chatbot (RAG Pipeline)
💬 Chat with your own PDFs using Generative AI + LangChain + Hugging Face



🧠 Architecture (RAG Pipeline)

    PDF/Text Document → Text Chunking → Embeddings → Vector Store
                                ↓
                           User Question
                                ↓
                     Retrieve Relevant Chunks
                                ↓
                      LLM Generates Answer


🚀 Overview

Have you ever wished you could ask questions directly from your PDFs or text files — just like chatting with ChatGPT?
This project makes that possible using a Retrieval-Augmented Generation (RAG) pipeline!

It combines LangChain, Hugging Face Transformers, and Chroma Vector Database to build an intelligent chatbot that can read and answer questions from any uploaded document. 📄🤖

🧩 Key Features

✅ Upload & Process PDFs or Text Files – Read any document easily
✅ Retrieval-Augmented Generation (RAG) – Get context-aware answers from your own data
✅ LLM-powered Answers – Uses google/flan-t5-large from Hugging Face
✅ Semantic Search with Embeddings – Finds the most relevant parts of your document
✅ Built-in Streamlit UI – Simple, elegant interface for interactive chatting
✅ Fully Local – Works without sending your private data to external APIs

⚙️ Tech Stack
Component	Description
LangChain	Framework for LLM orchestration & chaining
Hugging Face Transformers	Open-source LLM for text generation
ChromaDB	Vector database for efficient document retrieval
SentenceTransformers	For creating embeddings from text
Streamlit	Web interface for the chatbot
Python	Core language powering the backend
🧠 How RAG Works

RAG = Retrieval + Generation

Load Document → Read PDF or text

Chunk Text → Split into small sections

Embed Text → Convert each section into a vector (numerical meaning)

Store in Vector DB → Save in Chroma for semantic search

Retrieve Context → Find chunks relevant to user’s query

Generate Answer → LLM uses both retrieved context + query to answer

🧩 Result: An LLM that understands your custom data — not just what it was trained on!

💻 How to Run the Project
1️⃣ Clone this Repository
git clone https://github.com/your-username/rag-qa-chatbot.git
cd rag-qa-chatbot

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit App
streamlit run app.py

4️⃣ Upload and Ask!

Upload any PDF or text file, then ask:

“What is this document about?”
“Who is the author?”
“Summarize the key points.”

📘 Example

Try it with the sample document:
📄 sample_ai_intro.pdf — Introduction to Artificial Intelligence

You can ask:

“What are the types of AI?”
“What are the limitations of AI?”

🌟 Future Enhancements

🧱 Add chat history + memory

🔍 Support multiple documents

☁️ Deploy on Hugging Face Spaces / Streamlit Cloud

🧠 Add OpenAI or Gemini API for better generation quality

💾 Persistent storage for uploaded document embeddings

🧑‍💻 Author

👋 [Your Name]
Generative AI & LLM Enthusiast | Learning from Krish Naik’s Complete GenAI Course

“Building smart apps that truly understand your data.”
