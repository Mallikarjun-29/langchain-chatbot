import streamlit as st
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain import hub
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
import re

# ===== CONSTANTS =====
EXIT_WORDS = ["exit", "back", "quit", "main", "go back", "stop", "done"]
MIN_ARTICLE_LENGTH = 200

# ===== HELPERS =====
def fresh_article_memory():
    return ConversationBufferWindowMemory(
        k=10, memory_key="article_history", return_messages=True
    )

def fresh_agent_memory():
    return ConversationBufferWindowMemory(
        k=5, memory_key="chat_history", return_messages=True
    )

def reset_article_state():
    st.session_state.current_article_url = None
    st.session_state.current_article_title = None
    st.session_state.article_memory = fresh_article_memory()

def reset_all():
    st.session_state.messages = []
    st.session_state.memory = fresh_agent_memory()
    st.session_state.article_store = []
    reset_article_state()

def fetch_article(url: str):
    try:
        docs = WebBaseLoader(url).load()
        content = docs[0].page_content
        if len(content.strip()) < MIN_ARTICLE_LENGTH:
            return None, "🔒 This article is paywalled or blocked. Try another one!"
        return content, None
    except Exception as e:
        return None, f"❌ Could not fetch article: {str(e)}"

def chat_with_article(article_content: str, question: str) -> str:
    history = st.session_state.article_memory.load_memory_variables({})
    chat_history = history.get("article_history", "")
    response = st.session_state.llm.invoke(
        f"You are a financial analyst. Use ONLY the article below to answer.\n\n"
        f"Article: {article_content}\n\n"
        f"Conversation so far:\n{chat_history}\n\n"
        f"User question: {question}"
    )
    output = response.content
    st.session_state.article_memory.save_context(
        {"input": question}, {"output": output}
    )
    return output

# ===== SESSION STATE INIT =====
def init_session_state():
    defaults = {
        "messages": [],
        "memory": fresh_agent_memory(),
        "article_memory": fresh_article_memory(),
        "article_store": [],
        "current_article_url": None,
        "current_article_title": None,
        "llm": None,
        "agent_executor": None,
        "react_prompt": None,
        "last_groq_key": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# ===== SIDEBAR =====
st.title("📈 Financial Research Agent")

with st.sidebar:
    st.header("⚙️ Settings")

    groq_api_key = st.text_input(
        "Groq API Key:",
        type="password",
        help="Free at https://console.groq.com/keys"
    )

    st.divider()
    st.markdown("### 💡 How to use")
    st.markdown("""
    - Ask about any stock: `TSLA news` or `AAPL price`
    - Chat with a saved article: type `3` or `article 3`
    - Chat with any URL: paste `https://...`
    - Type `exit` to leave article chat
    """)

    st.divider()

    if st.button("✨ New Chat", use_container_width=True, type="primary"):
        reset_all()
        st.session_state.agent_executor = None
        st.toast("✨ New chat started!", icon="🚀")
        st.rerun()

    if st.button("🗑️ Clear Article Memory", use_container_width=True):
        st.session_state.article_store = []
        reset_article_state()
        st.toast("🧹 Article memory cleared!", icon="✅")
        st.rerun()

# ===== GROQ KEY CHECK =====
if not groq_api_key:
    st.warning("👈 Please enter your Groq API key to start! Free at https://console.groq.com/keys")
    st.stop()

# ===== LLM + AGENT SETUP (cached, only rebuilds when key changes) =====
if st.session_state.last_groq_key != groq_api_key:
    st.session_state.llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key
    )
    if st.session_state.react_prompt is None:
        st.session_state.react_prompt = hub.pull("hwchase17/react")

    @tool
    def reverse_text(text: str) -> str:
        """Use this tool when the user asks to reverse any text or string.
        Input should be the text to reverse. Returns the reversed text."""
        return text[::-1]

    @tool
    def calculator(expression: str) -> str:
        """Use this tool to perform mathematical calculations like addition,
        subtraction, multiplication, division.
        Input should be a math expression like '2 + 2'."""
        if not re.match(r'^[\d\s\+\-\*\/\.\(\)]+$', expression):
            return "Invalid expression!"
        return str(eval(expression))

    @tool
    def get_stock_price(query: str) -> str:
        """ALWAYS use this tool for ANY stock price or news request.
        Input should be ONLY the stock symbol like: TSLA or AAPL
        If user asks for specific number of articles, include it like: TSLA:10
        Returns real-time price and news."""
        import yfinance as yf

        parts = query.strip().split(":")
        symbol = parts[0].strip().upper()
        count = int(parts[1].strip()) if len(parts) > 1 and parts[1].strip().isdigit() else 5

        stock = yf.Ticker(symbol)
        info = stock.info
        news = stock.news[:count]

        price = (info.get('currentPrice') or info.get('regularMarketPrice')
                 or info.get('previousClose') or "Unavailable")
        price_label = ("Live Price" if info.get('currentPrice')
                       else "Market Price" if info.get('regularMarketPrice')
                       else "Previous Close" if info.get('previousClose')
                       else "Price")

        result = f"{price_label} of {symbol}: ${price}\n\nLatest {count} News:\n"
        articles = []

        for i, article in enumerate(news):
            content = article.get('content', {})
            title = content.get('title', 'No title')
            summary = content.get('summary', 'No description')
            link = content.get('canonicalUrl', {}).get('url', 'No link')
            result += f"\n{i+1}. {title}\n{summary}\n🔗 {link}\n---\n"
            articles.append({"index": i + 1, "title": title, "url": link})

        st.session_state.article_store = articles
        return result

    tools = [calculator, reverse_text, get_stock_price]
    agent = create_react_agent(
        llm=st.session_state.llm,
        tools=tools,
        prompt=st.session_state.react_prompt
    )
    st.session_state.agent_executor = AgentExecutor(
        agent=agent, tools=tools,
        memory=st.session_state.memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=15
    )
    st.session_state.last_groq_key = groq_api_key

# ===== CHAT UI =====
if st.session_state.current_article_title:
    st.info(f"📰 Chatting with: **{st.session_state.current_article_title}** | Type `exit` to go back")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ===== INTENT DETECTION + ROUTING =====
if prompt_input := st.chat_input("Ask about stocks, paste a URL, or type article number..."):

    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    output = ""
    lower_input = prompt_input.lower().strip()

    with st.chat_message("assistant"):

        # ── EXIT article chat ──────────────────────────────────────
        if any(w in lower_input for w in EXIT_WORDS) and st.session_state.current_article_url:
            reset_article_state()
            output = "✅ Back to main agent! Ask me about any stock."
            st.markdown(output)

        # ── NEW URL → load & summarize ─────────────────────────────
        elif prompt_input.startswith(("https://", "http://")):
            with st.spinner("📰 Fetching article..."):
                content, error = fetch_article(prompt_input)
                if error:
                    output = error
                    st.warning(output)
                else:
                    reset_article_state()
                    st.session_state.current_article_url = prompt_input
                    st.session_state.current_article_title = prompt_input[:60] + "..."
                    response = st.session_state.llm.invoke(
                        f"You are a financial analyst. Summarize this article clearly, "
                        f"highlighting key financial insights:\n\n{content}"
                    )
                    output = response.content
                    st.markdown(output)
                    st.success("💬 Article loaded! Ask me anything about it.")

        # ── NUMBER → load saved article ────────────────────────────
        elif re.search(r'\d+', prompt_input) and st.session_state.article_store:
            with st.spinner("📰 Loading article..."):
                match = re.search(r'\d+', prompt_input)
                index = int(match.group()) - 1

                if index < 0 or index >= len(st.session_state.article_store):
                    output = f"❌ Article {index+1} not found. I have {len(st.session_state.article_store)} articles."
                    st.markdown(output)
                else:
                    article = st.session_state.article_store[index]
                    url, title = article["url"], article["title"]

                    if st.session_state.current_article_url != url:
                        reset_article_state()
                        st.session_state.current_article_url = url
                        st.session_state.current_article_title = title

                    content, error = fetch_article(url)
                    if error:
                        output = error
                        st.warning(output)
                        reset_article_state()
                    else:
                        output = chat_with_article(content, prompt_input)
                        st.info(f"📰 **{title}**")
                        st.markdown(output)

        # ── FOLLOW-UP on loaded article ────────────────────────────
        elif st.session_state.current_article_url:
            with st.spinner("💭 Reading article..."):
                content, error = fetch_article(st.session_state.current_article_url)
                if error:
                    output = error + " Type `exit` to go back."
                    st.warning(output)
                else:
                    output = chat_with_article(content, prompt_input)
                    st.info(f"📰 **{st.session_state.current_article_title}**")
                    st.markdown(output)

        # ── AGENT question ─────────────────────────────────────────
        else:
            with st.spinner("🤔 Thinking..."):
                try:
                    response = st.session_state.agent_executor.invoke({"input": prompt_input})
                    output = response["output"]
                    st.markdown(output)
                except Exception as e:
                    output = f"❌ Error: {str(e)}"
                    st.markdown(output)

    st.session_state.messages.append({"role": "assistant", "content": output})
