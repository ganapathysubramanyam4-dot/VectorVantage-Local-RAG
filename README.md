# ⚡ VectorVantage AI Assistant

An Enterprise Hybrid Retrieval-Augmented Generation (RAG) Neural Workspace built with Streamlit, LangChain, and Ollama. This application allows users to upload PDF documents and engage in a history-aware smart chat using a hybrid search mechanism (Dense Vector + Sparse Lexical).

## 🚀 Features
* **Hybrid Search Architecture:** Combines semantic search via **FAISS** (using `all-MiniLM-L6-v2` embeddings) with keyword matching via **BM25 Retriever** for ultra-precise document retrieval.
* **History-Aware Query Synthesis:** Automatically restructures conversational follow-up questions into standalone keyword queries based on previous chat context.
* **Factual Boundary Enforced Guardrails:** Strict prompt engineering ensures the AI answers *only* using the provided document context, eliminating hallucinations.
* **Premium SaaS UI/UX:** A minimalist, clean, and modern custom-styled interface featuring floating thread chat bubbles and real-world 1-indexed document source badges.
* **Automated Document Insights:** Dedicated workspace to instantly generate highly structured executive summaries and data nodes from uploaded documents.

## 🛠️ Tech Stack
* **Frontend/UI:** Streamlit, Custom HTML/CSS injection
* **Orchestration:** LangChain (Community, HuggingFace, Text Splitters)
* **Vector Store:** FAISS (Facebook AI Similarity Search)
* **Keyword Search:** BM25 Retriever
* **Local LLM Engine:** Ollama (Running Mistral-7B)
* **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2`

## ⚙️ Prerequisites
Before running the application, make sure you have **Ollama** installed and the **Mistral** model downloaded locally.

1. **Install Ollama:** Download from [ollama.com](https://ollama.com)
2. **Pull the Mistral Model:** Open your command prompt/terminal and run:
   ```bash
   ollama pull mistral
