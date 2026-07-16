✨ **The Enchanted Library**

A Retrieval-Augmented Generation (RAG) chatbot that lets you ask questions about the characters and plot of a curated set of classic books, novels, and scriptures. Built with Streamlit, LangChain, Pinecone, and Groq.

**Render Link** : https://book-ai-chatbot.onrender.com/

**Live Books**

1) Pride and Prejudice
2) Frankenstein
3) Little Women
4) Crime and Punishment
5) The Mahabharata
6) Bhagavad Gita
7) Sense and Sensibility
8) The Yoga-Vasishtha Maharamayana

**How It Works**

The project has two independent parts:

**1. Ingestion (ingestion.ipynb) — run once,locally**

**For each book**:

Searches Gutendex (a Project Gutenberg API) for the title
Downloads the plain-text version
Splits it into ~1000-character chunks with 200-character overlap
Embeds each chunk locally using sentence-transformers/all-MiniLM-L6-v2
Upserts the embeddings + chunk text into a Pinecone index named enchanted-library

**2. App (app.py) — the deployed chatbot**

User asks a question in the Streamlit chat UI
If there's prior conversation history, an LLM call rewrites the question into a standalone query (resolving pronouns like "he"/"she" into actual names)
The standalone question is embedded and used to retrieve the top 8 most relevant chunks from Pinecone
The retrieved chunks + conversation history + question are passed to a Groq-hosted LLM (llama-3.1-8b-instant) to generate the final answer
An expander below the answer shows what the AI retrieved and how it interpreted the question, for transparency

**Tech Stack**

**Purpose **                      **Tool **                               
UI                                Streamlit
Orchestration                     LangChain
Vector database                   Pinecone
LLM inference                     Groq (llama-3.1-8b-instant)
Embeddings                        fastembed (local, ONNX runtime) running sentence-transformers/all-MiniLM-L6-v2
Hosting                           Render
