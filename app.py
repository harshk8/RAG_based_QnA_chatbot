import streamlit as st
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline

# Streamlit UI
st.set_page_config(page_title="📄 Document Q&A Chatbot", layout="wide")
st.title("🧠 AI Document Q&A Chatbot (RAG Pipeline)")
st.write("Upload your document and ask any question about it.")

uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

if uploaded_file:
    # Load document
    with open("temp_file.pdf" if uploaded_file.type == "application/pdf" else "temp_file.txt", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Read document
    if uploaded_file.type == "application/pdf":
        loader = PyPDFLoader("temp_file.pdf")
    else:
        loader = TextLoader("temp_file.txt")
    documents = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    # Create embeddings
    st.info("🔍 Creating embeddings & indexing document...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create local vector store
    vectorstore = Chroma.from_documents(docs, embeddings)

    # Load Hugging Face LLM
    st.info("🧠 Loading model...")
    llm_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        max_length=512,
        temperature=0
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    # Create RetrievalQA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    # Chat interface
    st.success("✅ Document processed! Ask your questions below:")
    query = st.text_input("💬 Ask a question about the document:")

    if query:
        with st.spinner("Generating answer..."):
            result = qa_chain({"query": query})
            st.write("### 🧾 Answer:")
            st.write(result["result"])

            # Show source snippets
            with st.expander("📚 Source Snippets"):
                for doc in result["source_documents"]:
                    st.write(doc.page_content[:300] + "...")
