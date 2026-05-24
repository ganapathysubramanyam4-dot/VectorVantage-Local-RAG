import streamlit as st
import os
import tempfile
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

# --- Environment Setup ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==============================================================================
# 1. PREMIUM HIGH-END INTERFACE DESIGN SYSTEM (SaaS Minimalist CSS Injection)
# ==============================================================================
st.set_page_config(
    page_title="VectorVantage AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght=300;400;500;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif !important;
        background-color: #fcfcfd !important;
    }
    
    /* Premium Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #f8fafc !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown p {
        color: #0f172a !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
    }
    
    /* Minimalist Main Header Styling */
    .saas-header-box {
        padding: 15px 0px;
        margin-bottom: 25px;
        border-bottom: 1px solid #e4e4e7;
    }
    .saas-title {
        font-size: 28px;
        font-weight: 700;
        color: #09090b;
        letter-spacing: -0.75px;
        margin: 0;
    }
    .saas-subtitle {
        font-size: 14px;
        color: #71717a;
        margin-top: 4px;
        font-weight: 400;
    }

    /* Floating SaaS Thread Chat Bubbles */
    .conversation-thread {
        width: 100%;
        margin-bottom: 20px;
        display: flex;
        flex-direction: column;
    }
    .avatar-row {
        display: flex;
        align-items: center;
        margin-bottom: 6px;
        gap: 8px;
    }
    .avatar-icon-user {
        width: 24px;
        height: 24px;
        background-color: #09090b;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 10px;
        font-weight: 600;
    }
    .avatar-icon-ai {
        width: 24px;
        height: 24px;
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 10px;
        font-weight: 600;
    }
    .profile-name {
        font-size: 13px;
        font-weight: 600;
        color: #27272a;
    }
    .saas-bubble {
        padding: 14px 18px;
        border-radius: 12px;
        max-width: 82%;
        line-height: 1.6;
        font-size: 15px;
        color: #09090b;
    }
    .user-align { align-self: flex-end; }
    .user-saas-bubble {
        background-color: #f4f4f5;
        border: 1px solid #e4e4e7;
        border-top-right-radius: 2px;
    }
    .ai-align { align-self: flex-start; }
    .ai-saas-bubble {
        background-color: #ffffff;
        border: 1px solid #e4e4e7;
        border-top-left-radius: 2px;
        box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.02);
    }

    /* Citation Badges Frame */
    .source-container {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 10px;
        padding-top: 8px;
        border-top: 1px dashed #e4e4e7;
    }
    .source-title-text {
        font-size: 11px;
        font-weight: 700;
        color: #71717a;
        width: 100%;
    }
    .source-badge {
        background-color: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
        padding: 3px 6px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Main Structural Typography Header Block
st.markdown("""
<div class="saas-header-box">
    <h1 class="saas-title">📑 VectorVantage AI Assistant</h1>
    <p class="saas-subtitle">Enterprise Hybrid Retrieval-Augmented Generation Neural Workspace</p>
</div>
""", unsafe_allow_html=True)

# --- Optimized Embeddings Loading ---
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

embeddings = load_embeddings()
DB_FAISS_PATH = 'faiss_index'

# ==============================================================================
# 2. SIDEBAR: DOCUMENT MANAGEMENT & ML STATUS
# ==============================================================================
with st.sidebar:
    st.title("🚀 VectorVantage")
    st.markdown("---")
    
    st.header("⚙️ ML System Status")
    st.info("""
    **Search:** Hybrid (BM25 + FAISS)  
    **Memory:** History-Aware Context Re-writing  
    **Guardrail:** Context Boundary Enforced  
    **Model:** Mistral-7B via Ollama  
    """)
    
    st.divider()
    st.header("📂 Document Management")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("Process Documents", use_container_width=True):
        if uploaded_files:
            with st.spinner("Building Intelligent Index..."):
                all_docs = []
                if not os.path.exists("temp_pdf"):
                    os.makedirs("temp_pdf")
                
                for uploaded_file in uploaded_files:
                    file_path = os.path.join("temp_pdf", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    loader = PyPDFLoader(file_path)
                    data = loader.load()
                    
                    # 🛠️ CRITICAL ACCURACY TWEAK 1: Explicitly map exact structural file metrics to child chunks
                    for idx, doc in enumerate(data):
                        doc.metadata["source"] = uploaded_file.name
                        doc.metadata["page"] = int(doc.metadata.get("page", idx))
                        
                    all_docs.extend(data)
                
                # Dynamic text chunks optimization split map
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=550, chunk_overlap=120)
                splits = text_splitter.split_documents(all_docs)
                
                db = FAISS.from_documents(splits, embeddings)
                db.save_local(DB_FAISS_PATH)
                st.session_state.splits = splits
                st.success("Indexing Successful!")
        else:
            st.error("Please upload a PDF.")

    st.divider()
    if st.button("Reset Conversation", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = ""
        st.rerun()

# ==============================================================================
# 3. MAIN UI WORKSPACE & MULTI-TURN CHAT PIPELINE
# ==============================================================================
tab1, tab2 = st.tabs(["💬 Smart Chat", "📊 Document Insights"])

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ""

with tab1:
    # Render historical conversation strings inside modern custom UI containers
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="conversation-thread user-align">
                <div class="avatar-row" style="justify-content: flex-end;">
                    <span class="profile-name">You</span>
                    <div class="avatar-icon-user">U</div>
                </div>
                <div class="saas-bubble user-saas-bubble user-align">{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            badges_html = ""
            if "sources" in message and message["sources"]:
                badges_html = '<div class="source-container"><div class="source-title-text">🎯 VERIFIED REFERENCE PAGES:</div>'
                for src in message["sources"]:
                    badges_html += f'<span class="source-badge">📄 {src}</span>'
                badges_html += '</div>'
                
            st.markdown(f"""
            <div class="conversation-thread ai-align">
                <div class="avatar-row">
                    <div class="avatar-icon-ai">VV</div>
                    <span class="profile-name">VectorVantage AI</span>
                </div>
                <div class="saas-bubble ai-saas-bubble ai-align">
                    {message["content"]}
                    {badges_html}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Core user query processing system
    if user_query := st.chat_input("Ask VectorVantage..."):
        st.markdown(f"""
        <div class="conversation-thread user-align">
            <div class="avatar-row" style="justify-content: flex-end;">
                <span class="profile-name">You</span>
                <div class="avatar-icon-user">U</div>
            </div>
            <div class="saas-bubble user-saas-bubble user-align">{user_query}</div>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "user", "content": user_query})

        if os.path.exists(DB_FAISS_PATH) and "splits" in st.session_state:
            llm = Ollama(model="mistral", base_url="http://127.0.0.1:11434", temperature=0.2)
            
            # 🛠️ CRITICAL ENHANCEMENT 2: History-Aware Query Synthesis Framework
            search_query = user_query
            if st.session_state.chat_history:
                try:
                    rewrite_prompt = (
                        f"Review the following chat log history, analyze the context, and reshape the new user follow-up "
                        f"question into a standalone keyword search query for document vector pulling.\n\n"
                        f"Chat Logs:\n{st.session_state.chat_history}\n"
                        f"New Follow-up Question: {user_query}\n\n"
                        f"Standalone Search Query (Output only the query string without prefaces):"
                    )
                    search_query = llm.invoke(rewrite_prompt).strip()
                except:
                    search_query = user_query

            # Dynamic Streaming Response Frame Initialization
            st.markdown("""
            <div class="avatar-row">
                <div class="avatar-icon-ai">VV</div>
                <span class="profile-name">VectorVantage AI</span>
            </div>
            """, unsafe_allow_html=True)
            
            response_placeholder = st.empty()
            
            # Executing Hybrid Retrieval Protocols (FAISS Vector + BM25 Lexical)
            db = FAISS.load_local(DB_FAISS_PATH, load_embeddings(), allow_dangerous_deserialization=True)
            vector_docs = db.similarity_search(search_query, k=3)
            
            bm25 = BM25Retriever.from_documents(st.session_state.splits)
            bm25.k = 3
            keyword_docs = bm25.invoke(search_query)
            
            # 🛠️ CRITICAL ENHANCEMENT 3: De-duplication and Exact Source-Page Calibration Matrix
            combined_docs = vector_docs + keyword_docs
            seen_contents = set()
            unique_docs = []
            source_info_list = []
            
            for d in combined_docs:
                content_hash = d.page_content.strip()
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    unique_docs.append(d)
                    
                    # Compute exact real-world 1-indexed conversion values
                    f_name = os.path.basename(d.metadata.get('source', 'Unknown Document'))
                    p_num = int(d.metadata.get('page', 0)) + 1
                    source_info_list.append(f"{f_name} (Page: {p_num})")
            
            context = "\n\n".join([f"--- CHUNK FACT --- \n {d.page_content}" for d in unique_docs])
            
            # 🛠️ CRITICAL ENHANCEMENT 4: Factual Boundary Control Guardrail Prompt Engineering
            final_prompt = (
                f"You are a Strict Factual Knowledge Analyst. Your mission is to provide an analytical response to the question "
                f"using ONLY the provided factual context blocks. Do not use external information.\n\n"
                f"STRICT SYSTEM COMPLIANCE LAWS:\n"
                f"1. Answer the question comprehensively based strictly on the data chunks inside the <context> tag.\n"
                f"2. Use bullet points or numbers to clean up your explanation structure.\n"
                f"3. CRITICAL RULE: If the context fragments do not contain enough facts to generate an accurate answer, or if the data is missing, reply EXACTLY with: 'I cannot find the answer within the uploaded documents.' Do not improvise, hypothesize, or invent anything outside the text frames.\n"
                f"4. Do not state page numbers or document names inside your sentences. The system processes citations externally.\n\n"
                f"<context>\n{context}\n</context>\n\n"
                f"Question: {user_query}\n\n"
                f"Strict Direct Answer:"
            )
            
            full_response = ""
            try:
                # Execution via Local Streaming Neural Pipeline
                for chunk in llm.stream(final_prompt):
                    full_response += chunk
                    response_placeholder.markdown(f'<div class="saas-bubble ai-saas-bubble ai-align">{full_response}▌</div>', unsafe_allow_html=True)
                
                # Render verified source badges cleanly below bubble
                badges_html = ""
                unique_citations = list(set(source_info_list))
                
                # Check if guardrail triggered, if so, prune citations frame
                if "I cannot find the answer within the uploaded documents" in full_response:
                    unique_citations = []
                    
                if unique_citations:
                    badges_html = '<div class="source-container"><div class="source-title-text">🎯 VERIFIED REFERENCE PAGES:</div>'
                    for src in unique_citations:
                        badges_html += f'<span class="source-badge">📄 {src}</span>'
                    badges_html += '</div>'
                
                response_placeholder.markdown(f'<div class="saas-bubble ai-saas-bubble ai-align">{full_response}{badges_html}</div>', unsafe_allow_html=True)
                
                # Multi-turn memory buffer updates
                st.session_state.chat_history += f"\nUser: {user_query}\nAI: {full_response}\n"
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "sources": unique_citations
                })
                
            except Exception as e:
                st.error(f"Ollama Connection Error. Ensure your backend neural instances are live in CMD via 'ollama run mistral'")
        else:
            st.warning("Please process documents first!")

# ==============================================================================
# 4. TAB 2: AUTOMATED DOCUMENT INSIGHTS WORKSPACE
# ==============================================================================
with tab2:
    st.header("🔍 Automated Document Analysis")
    if "splits" in st.session_state:
        st.write(f"**Total Text Chunks Analyzed:** {len(st.session_state.splits)}")
        if st.button("Generate Key Insights Framework", use_container_width=True):
            with st.spinner("Analyzing..."):
                try:
                    insight_llm = Ollama(model="mistral", base_url="http://127.0.0.1:11434", temperature=0.4)
                    
                    # Smart global slice tracking matrices sampling
                    total_chunks = len(st.session_state.splits)
                    sample_size = min(3, total_chunks)
                    
                    combined_sample = ""
                    for idx in range(sample_size):
                        chunk_data = st.session_state.splits[idx * (total_chunks // sample_size)]
                        f_name = os.path.basename(chunk_data.metadata.get('source', 'Document'))
                        combined_sample += f"\n[Document Fragment from {f_name}]:\n{chunk_data.page_content[:600]}\n"
                    
                    insight_prompt = (
                        f"Extract and generate an executive operational summary based on the document frames. "
                        f"Provide exactly 5 highly structured business-ready bullet points highlighting milestones, data metrics, or topics.\n\n"
                        f"Data Nodes:\n{combined_sample}"
                    )
                    insight = insight_llm.invoke(insight_prompt)
                    st.info(insight)
                except Exception as e:
                    st.error("Failed to generate insights. Check if Ollama instance is operating correctly.")
    else:
        st.info("Upload and process a document to see insights here.")