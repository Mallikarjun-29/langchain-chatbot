from dotenv import load_dotenv
import os
load_dotenv()
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory

# Page title
st.title("🤖 IT Support Chatbot")

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=5, return_messages=True)

if "chart_history" not in st.session_state:
    st.session_state.chat_history = []

# 1. LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)
memory = ConversationBufferWindowMemory(k=5, return_messages=True)

# 2. Prompt Template
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {role} assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# 3. Output Parser
parser = StrOutputParser()

# 4. Chain = prompt | llm | parser
chain = template | llm | parser

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.write(chat["content"])

# User input
if question := st.chat_input("Ask me anything..."):
    
    # Show user message
    with st.chat_message("user"):
        st.write(question)
    
    # Get response
    history = st.session_state.memory.chat_memory.messages
    response = chain.invoke({
    "question": question,
    "history": history,
    "role": "IT"
})
    
    # Save to memory
    st.session_state.memory.chat_memory.add_user_message(question)
    st.session_state.memory.chat_memory.add_ai_message(response)
    
    # Save to chat history
    st.session_state.chat_history.append({"role": "user", "content": question})
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # Show bot response
    with st.chat_message("assistant"):
        st.write(response)