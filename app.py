import streamlit as st
import os
import requests
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
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

if not os.environ.get("GROQ_API_KEY") or not os.environ.get("PINECONE_API_KEY"):
    st.error("Missing API Keys! Please ensure they are set in the Render dashboard.")
if not os.environ.get("GROQ_API_KEY") or not os.environ.get("PINECONE_API_KEY"):
    st.error("Missing API Keys! Please ensure GROQ_API_KEY and PINECONE_API_KEY are set.")

with st.sidebar:
    st.header("📚 About this App")
    st.markdown("This AI assistant is pre-configured and ready to use! Just type your question below.")
    
    st.markdown("---")
    st.header("➕ Add a New Book")
    st.markdown("Want to chat with a different book? Search for it below (e.g., 'Frankenstein', 'Dracula').")
    
    new_book_title = st.text_input("Book Title:")
    
    if st.button("Download & Add to Library"):
        if new_book_title:
            with st.spinner(f"Searching Gutenberg for '{new_book_title}'..."):
                try:
                    search_url = f"https://gutendex.com/books/?search={new_book_title.replace(' ', '%20')}"
                    response = requests.get(search_url).json()
                    
                    if response['count'] == 0:
                        st.error("Book not found in the Gutenberg library.")
                    else:
                        book_data = response['results'][0]
                        actual_title = book_data['title']
                        formats = book_data['formats']
                        
                        text_url = None
                        for key, url in formats.items():
                            if 'text/plain' in key:
                                text_url = url
                                break
                                
                        if not text_url:
                            st.error(f"Sorry, a plain text version of '{actual_title}' is not available.")
                        else:
                            st.info(f"Found '{actual_title}'. Downloading...")
                            text_response = requests.get(text_url)
                            safe_title = re.sub(r'[^A-Za-z0-9]', '_', actual_title)
                            file_path = f"temp_{safe_title}.txt"
                            
                            with open(file_path, "w", encoding="utf-8") as file:
                                file.write(text_response.text)
                                
                            st.info("Reading and chunking chapters...")
                            loader = TextLoader(file_path, encoding="utf-8")
                            documents = loader.load()
                            
                            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                            chunks = text_splitter.split_documents(documents)
                            
                            for chunk in chunks:
                                chunk.metadata = {"book_title": actual_title}
                                
                            st.info("Saving to Pinecone Cloud...")
                            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                            
                            PineconeVectorStore.from_documents(
                                documents=chunks, 
                                embedding=embedding_model, 
                                index_name="enchanted-library"
                            )
                            
                            os.remove(file_path)
                            st.cache_resource.clear() 
                            st.success(f"Successfully added '{actual_title}'! You can now chat with it.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please type a book title first!")

@st.cache_resource
def load_database():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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
