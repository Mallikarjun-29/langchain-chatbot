import streamlit as st
from dotenv import load_dotenv
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


# Steps 4-5 run every time user asks a question
def build_vector_store(uploaded_file):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    # Load
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    
    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    # Embed and store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return vector_store

st.title("📄 RAG Document Assistant")
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = build_vector_store(uploaded_file)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if question := st.chat_input("Ask a question about your document..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Build RAG chain and get answer
    retriever = st.session_state.vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_template("""
    Use the following context to answer the question.
    If you don't know the answer, say "I don't know".
    
    Context: {context}
    Question: {question}
    """)
    llm = ChatGroq(model_name="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    answer = rag_chain.invoke(question)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()