import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
)

# ── Prompts ───────────────────────────────────────────────────
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", """Given the chat history and the latest user question,
reformulate the question as a standalone question.
If no reformulation is needed, return it as is."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant.
Answer the user's question using the context below.

Rules:
1. If your answer is based on the context, add [DOC] at the end.
2. If you used your own knowledge, add [LLM] at the end.

Context: {context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# ── Vector store builder ──────────────────────────────────────
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

# ── Chain builder ─────────────────────────────────────────────
def build_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )
    question_answer_chain = create_stuff_documents_chain(
        llm, answer_prompt
    )
    return create_retrieval_chain(
        history_aware_retriever, question_answer_chain
    )

# ── Session state init ────────────────────────────────────────
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

init_session_state()

# ── Sidebar ───────────────────────────────────────────────────
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
                st.session_state.chain = build_chain(
                    st.session_state.vector_store
                )
            st.success("PDF loaded!")

    st.markdown("---")
    if st.button("🗑️ New Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.chain = None
        st.session_state.vector_store = None
        st.rerun()

# ── Main UI ───────────────────────────────────────────────────
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
                response = st.session_state.chain.invoke({
                    "input": question,
                    "chat_history": st.session_state.chat_history
                })
                answer = response["answer"]
            else:
                answer = llm.invoke(question).content + " [LLM]"

            # Update chat history (keep last 5 turns = 10 messages)
            st.session_state.chat_history.extend([
                HumanMessage(content=question),
                AIMessage(content=answer)
            ])
            if len(st.session_state.chat_history) > 10:
                st.session_state.chat_history = st.session_state.chat_history[-10:]

        st.write(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })