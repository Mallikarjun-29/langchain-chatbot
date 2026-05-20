from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

loader = PyPDFLoader("documents/Mallikharjun_Analyst.pdf")
documents = loader.load()
chunk_size = 2900
chunk_overlap = 200
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
print(len(documents))
print(documents)
chunks = text_splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever()
result = retriever.invoke("What is Mallikharjun's experience with Databricks?")
print("The result is**********************************:", result)
print("Number of pages:", len(documents))
print("Number of chunks:", len(chunks))

prompt = ChatPromptTemplate.from_template("""
Use the following context to answer the question.
If you don't know the answer, just say "I don't know".

Context: {context}

Question: {question}
""")
load_dotenv()
llm = ChatGroq(model_name="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
answer = rag_chain.invoke("What is Mallikharjun's experience with Databricks?")
print(answer)