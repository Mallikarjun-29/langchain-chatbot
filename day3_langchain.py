from dotenv import load_dotenv
import os

load_dotenv()

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory


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

# 5. Run it
role = "IT"
print(f"🤖 {role} Bot ready! Type 'quit' to exit\n")

while True:
    question = input("You: ")
    if question.lower() == "quit":
        break
    
    history = memory.chat_memory.messages
    response = chain.invoke({
        "role": role,
        "question": question,
        "history": history
    })
    
    memory.chat_memory.add_user_message(question)
    memory.chat_memory.add_ai_message(response)
    
    print(f"\n🤖 Bot: {response}\n")