import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
import tempfile
import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

prompt_template = """You are a helpful assistant.
Answer the user's question using the context and chat history provided.

Context: {context}
Chat History: {chat_history}
Question: {question}

Rules:
1. If your answer is based on the context above, add [DOC] at the end.
2. If you used your own knowledge (context was irrelevant), add [LLM] at the end.

Helpful Answer:"""

prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=prompt_template
)

def build_vector_store(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        tmp_path = f.name
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.unlink(tmp_path)
    return vectorstore

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            return_messages=True
        )
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

init_session_state()

with st.sidebar:
    st.title("💬 Helpful Chatbot")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload a PDF (optional)",
        type="pdf"
    )

    if uploaded_file:
        if st.session_state.vector_store is None:
            with st.spinner("Building knowledge base..."):
                st.session_state.vector_store = build_vector_store(uploaded_file)
            st.success("PDF loaded!")

        if st.session_state.chain is None:
            st.session_state.chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=st.session_state.vector_store.as_retriever(),
                memory=st.session_state.memory,
                combine_docs_chain_kwargs={"prompt": prompt}
            )

    st.markdown("---")
    if st.button("🗑️ New Chat"):
        st.session_state.messages = []
        st.session_state.memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            return_messages=True
        )
        st.session_state.chain = None
        st.session_state.vector_store = None
        st.rerun()

st.title("💬 Helpful Chatbot")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if question := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            if st.session_state.chain:
                response = st.session_state.chain(
                    {"question": question}
                )
                answer = response["answer"]
            else:
                answer = llm.invoke(question).content + " [LLM]"

        st.write(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })