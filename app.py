import streamlit as st
import os
import dotenv
import uuid

if os.name == 'posix':
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage

from rag_methods import (
    load_doc_to_db, 
    load_url_to_db,
    stream_llm_response,
    stream_llm_rag_response,
)

dotenv.load_dotenv()

MODELS = [
        "openai/gpt-4-turbo",
        "openai/gpt-4o-mini",
        "openai/gpt-3.5-turbo-0125",
        "anthropic/claude-3-5-sonnet-20240620",
        "anthropic/claude-3-opus-20240229",
        "anthropic/claude-3-haiku-20240307",
        "groq/llama3-8b-8192",
        "groq/mixtral-8x7b-32768",
        "groq/gemma2-9b-it"
    ]

st.set_page_config(
    page_title="CHATKU AI", 
    page_icon="📚", 
    layout="centered", 
    initial_sidebar_state="expanded"
)

# --- Header ---
st.html("""<h2 style="text-align: center;"><i>Chatku AI </i> 📚🔍</h2>""")

# --- Initial Setup ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hai, apa yang bisa saya bantu?"}
    ]

# --- Side Bar LLM API Tokens ---
with st.sidebar:
        default_openai_api_key = os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") is not None else ""
        with st.popover("🔐 OpenAI"):
            openai_api_key = st.text_input(
                "OpenAI API Key (https://platform.openai.com/)", 
                value=default_openai_api_key, 
                type="password",
                key="openai_api_key",
            )

        default_anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") if os.getenv("ANTHROPIC_API_KEY") is not None else ""
        with st.popover("🔐 Anthropic"):
            anthropic_api_key = st.text_input(
                "Anthropic API Key (https://console.anthropic.com/)", 
                value=default_anthropic_api_key, 
                type="password",
                key="anthropic_api_key",
            )

        default_groq_api_key = os.getenv("GROQ_API_KEY") if os.getenv("GROQ_API_KEY") is not None else ""
        with st.popover("🔐 Groq"):
            groq_api_key = st.text_input(
                "Groq API Key (https://console.groq.com)", 
                value=default_groq_api_key, 
                type="password",
                key="groq_api_key",
            )

# --- Main Content ---
missing_openai = openai_api_key == "" or openai_api_key is None or "sk-" not in openai_api_key
missing_anthropic = anthropic_api_key == "" or anthropic_api_key is None
missing_groq = groq_api_key == "" or groq_api_key is None
if missing_openai and missing_anthropic and missing_groq:
    st.write("#")
    st.warning("⬅️ Masukkan API Key Yang Benar...")
else:
    # Sidebar
    with st.sidebar:
        st.divider()
        models = []
        for model in MODELS:
            if "openai" in model and not missing_openai:
                models.append(model)
            elif "anthropic" in model and not missing_anthropic:
                models.append(model)
            elif "groq" in model and not missing_groq:
                models.append(model)
            elif "azure-openai" in model:
                models.append(model)

        st.selectbox(
            "Pilih Model", 
            options=models,
            key="model",
        )

        cols0 = st.columns(2)
        with cols0[0]:
            is_vector_db_loaded = ("vector_db" in st.session_state and st.session_state.vector_db is not None)
            st.toggle(
                "RAG", 
                value=is_vector_db_loaded, 
                key="use_rag", 
                disabled=not is_vector_db_loaded,
            )

        with cols0[1]:
            st.button("Hapus Chat", on_click=lambda: st.session_state.messages.clear(), type="primary")

        st.header("Retrieval Augmented Generation:")
            
        # File upload input for RAG with documents
        st.file_uploader(
            "📄 Upload a document", 
            type=["pdf", "txt", "docx", "md"],
            accept_multiple_files=True,
            on_change=load_doc_to_db,
            key="rag_docs",
        )

        # URL input for RAG with websites
        st.text_input(
            "🌐 Masukkan URL", 
            placeholder="https://example.com",
            on_change=load_url_to_db,
            key="rag_url",
        )

        with st.expander(f"📚 Documents in DB ({0 if not is_vector_db_loaded else len(st.session_state.rag_sources)})"):
            st.write([] if not is_vector_db_loaded else [source for source in st.session_state.rag_sources])

    # Main chat app
    model_provider = st.session_state.model.split("/")[0]
    if model_provider == "openai":
        llm_stream = ChatOpenAI(
            api_key=openai_api_key,
            model_name=st.session_state.model.split("/")[-1],
            temperature=0.3,
            streaming=True,
        )
    elif model_provider == "anthropic":
        llm_stream = ChatAnthropic(
            api_key=anthropic_api_key,
            model=st.session_state.model.split("/")[-1],
            temperature=0.3,
            streaming=True,
        )
    elif model_provider == "groq":
        llm_stream = ChatGroq(
            api_key=groq_api_key,
            model_name=st.session_state.model.split("/")[-1],
            temperature=0.3,
            streaming=True,
        )


    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Your message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages]

            if not st.session_state.use_rag:
                st.write_stream(stream_llm_response(llm_stream, messages))
            else:
                st.write_stream(stream_llm_rag_response(llm_stream, messages))

with st.sidebar:
    st.divider()
    st.write("Created with &#10084; by Rizky Dharma A.R")
    st.write("📋[GitHub](https://github.com/rizkydharma02)")