import streamlit as st
import os
import time
import requests
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re

st.set_page_config(page_title="My Book AI", page_icon="✨", layout="centered")

fairytale_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Playfair+Display:ital,wght@0,400;0,700;1,400&display=swap');

.stApp {
    background-image: url("https://images.unsplash.com/photo-1518709268805-4e9042af9f23?ixlib=rb-4.0.3&auto=format&fit=crop&w=2560&q=80");
    background-color: #121914;
    background-size: cover !important;
    background-position: center center !important;
    background-repeat: no-repeat !important;
    background-attachment: fixed !important;
    color: #e3ebd8;
    font-family: 'Playfair Display', serif;
}

.stApp p, .stApp div, .stApp span, .stApp label {
    color: #e3ebd8 !important;
}

h1, h2, h3 {
    font-family: 'Cinzel', serif !important;
    color: #d4c4a1 !important;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.9);
    text-align: center;
}

[data-testid="stSidebar"] {
    background-color: rgba(18, 25, 20, 0.85) !important;
    border-right: 2px solid #5a6b5a;
}

[data-testid="stChatMessage"] {
    background-color: rgba(15, 20, 15, 0.85) !important;
    border: 1px solid #5a6b5a;
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.6);
    font-size: 1.1rem;
}

[data-testid="stChatInput"] {
    border-radius: 10px !important;
    border: 2px solid #5a6b5a !important;
    background-color: rgba(20, 25, 20, 0.9) !important;
}

.stButton > button {
    background-color: rgba(40, 50, 40, 0.9);
    color: #d4c4a1 !important;
    border: 1px solid #d4c4a1;
    border-radius: 5px;
    font-family: 'Cinzel', serif;
    font-weight: bold;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background-color: rgba(60, 75, 60, 0.9);
    box-shadow: 0 0 15px rgba(212, 196, 161, 0.4);
    border-color: #f3e5ab;
}
</style>
"""
st.markdown(fairytale_css, unsafe_allow_html=True)

st.title("✨ The Enchanted Library ✨")
st.write("Ask any question about the characters or plot of your chosen tale.")

if not os.environ.get("GROQ_API_KEY") or not os.environ.get("PINECONE_API_KEY") or not os.environ.get("HF_TOKEN"):
    st.error("Missing API Keys! Please ensure GROQ_API_KEY, PINECONE_API_KEY, and HF_TOKEN are set in the Render dashboard.")
    
with st.sidebar:
    st.header("📚 Welcome to the Library")
    st.markdown("This AI assistant is pre-configured and ready to use! Just type your question in the chat.")
    
    st.markdown("---")
    st.header("📜 Available Tomes")
    st.markdown("The following texts are currently bound within our Pinecone archives:")
    
    books = [
        "Pride and Prejudice",
        "Frankenstein",
        "Little Women",
        "Crime and Punishment",
        "The Mahabharata",
        "Bhagavad Gita",
        "Sense and Sensibility",
        "The Yoga-Vasishtha Maharamayana"
    ]
    
    for book in books:
        st.markdown(f"- ✨ *{book}*")
        
    st.markdown("---")
    st.info("💡 **Developer Note:** The data ingestion pipeline is handled securely off-server to ensure lightning-fast inference.")

class HFRouterEmbeddings(Embeddings):
    """Lightweight embeddings wrapper that calls HF's current Inference Providers
    router endpoint directly via plain HTTP, instead of loading the model
    locally. Keeps memory footprint small enough for free-tier hosting.
    Includes retry logic since HF's serverless backend can be slow to
    'wake up' a model on the first request (cold start)."""

    def __init__(self, model_name: str, api_key: str):
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{model_name}/pipeline/feature-extraction"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "X-Wait-For-Model": "true"
        }

    def _embed(self, texts, max_retries=4):
        last_error = None
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json={"inputs": texts},
                    timeout=60
                )
                if response.status_code in (503, 504):
                    # Model is cold-starting or the gateway timed out waiting; retry.
                    last_error = f"{response.status_code}: {response.text[:200]}"
                    time.sleep(5 * (attempt + 1))
                    continue
                response.raise_for_status()
                return response.json()
            except requests.exceptions.Timeout:
                last_error = "Request timed out"
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"HF embedding API did not respond after {max_retries} attempts. Last error: {last_error}")

    def embed_documents(self, texts):
        return self._embed(texts)

    def embed_query(self, text):
        return self._embed([text])[0]


@st.cache_resource
def load_database():
    if not os.environ.get("HF_TOKEN"):
        st.error("Missing HF_TOKEN in Render environment variables!")

    embedding_model = HFRouterEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        api_key=os.environ.get("HF_TOKEN")
    )

    vector_db = PineconeVectorStore(
        index_name="enchanted-library", 
        embedding=embedding_model
    )
    return vector_db.as_retriever(search_kwargs={"k": 8})

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_text := st.chat_input("Ask a question about your books..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing the text..."):
            try:
                retriever = load_database()
                llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
                
                history_list = st.session_state.messages[:-1]
                if len(history_list) > 0:
                    history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history_list[-4:]])
                    rewrite_prompt = ChatPromptTemplate.from_template(
                        "Recent conversation:\n{history}\n\n"
                        "User's new question: '{question}'\n\n"
                        "Rewrite this follow-up question into a clear, standalone question that includes specific character names or book titles. "
                        "Return ONLY the rewritten question without any extra words."
                    )
                    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
                    standalone_question = rewrite_chain.invoke({"history": history_text, "question": prompt_text})
                else:
                    standalone_question = prompt_text
                    history_text = "No previous history."
                
                retrieved_docs = retriever.invoke(standalone_question)
                context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
                
                system_prompt = """You are a highly intelligent literary expert and AI assistant. 
You have vast general knowledge about books, literature, and characters.

STRICT OUTPUT RULES:
1. Answer the user's question directly, confidently, and factually.
2. If the provided Context is helpful, use it to enrich your answer with specific details. 
3. If the Context does not contain the answer, SEAMLESSLY use your own general knowledge to answer the question fully. Do not mention that the context was missing.
4. NEVER use robotic phrases like "Based on the provided snippets", "In the text", or "It is not explicitly stated". Just answer the question.

Conversation History:
{history}

Context:
{context}
"""
                
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}")
                ])
                
                chain = prompt_template | llm | StrOutputParser()
                
                response = chain.invoke({
                    "context": context_text, 
                    "history": history_text,
                    "input": standalone_question
                })
                
                st.markdown(response)
                
                with st.expander("👀 See how the AI thought & what it read"):
                    st.write(f"**1. The AI interpreted your question as:** `{standalone_question}`")
                    st.write("**2. Here are the exact chunks of text retrieved:**")
                    if len(retrieved_docs) > 0:
                        for i, doc in enumerate(retrieved_docs):
                            st.info(f"**Snippet {i+1}:**\n{doc.page_content}")
                    else:
                        st.write("No matching snippets found. The AI answered from its own memory.")
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"Something went wrong: {e}")
