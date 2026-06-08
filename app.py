import streamlit as st
import os
import tempfile
from rag_pipeline import (
    load_pdf,
    split_documents,
    create_vectorstore,
    answer_question,
    generate_summary,
    generate_quiz,
    important_topics
)

st.set_page_config(
    page_title="Smart Study Assistant",
    page_icon="🎓",
    layout="wide"
)

# ---- Custom CSS ----
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }
    
    /* Remove white space at top */
    #root > div:first-child {
        background: #0f1117 !important;
    }

    header[data-testid="stHeader"] {
        background: #0f1117 !important;
        border-bottom: 1px solid #1e2d47 !important;
    }

    .block-container {
        padding-top: 1rem !important;
    }

    div[data-testid="stToolbar"] {
        background: #0f1117 !important;
    }

    .main { background-color: #0f1117; }

    .stApp {
        background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 100%);
        color: #e8edf5;
    }

    /* Header */
    .header-box {
        background: linear-gradient(135deg, #1e3a5f, #2d1b69);
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        margin-bottom: 30px;
        border: 1px solid rgba(79,142,255,0.3);
        box-shadow: 0 8px 32px rgba(79,142,255,0.15);
    }

    .header-box h1 {
        font-size: 2.8rem;
        font-weight: 700;
        color: white;
        margin: 0;
    }

    .header-box p {
        color: #a0b4d0;
        font-size: 1.1rem;
        margin-top: 10px;
    }

    /* Answer box */
    .answer-box {
        background: linear-gradient(135deg, #1a2744, #1e1b3a);
        border: 1px solid rgba(79,142,255,0.3);
        border-radius: 16px;
        padding: 28px;
        margin-top: 20px;
        line-height: 1.9;
        font-size: 1rem;
        color: #d4e0f0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }

    .answer-label {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #4f8eff;
        margin-bottom: 14px;
    }

    /* Feature cards */
    .feature-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        height: 100%;
    }

    .feature-card h3 { color: #e8edf5; font-size: 1.1rem; }
    .feature-card p  { color: #6b7fa3; font-size: 0.9rem; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #111827 !important;
        border-right: 1px solid #1e2d47;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4f8eff, #7c3aed) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        width: 100% !important;
        transition: transform 0.2s !important;
        box-shadow: 0 4px 15px rgba(79,142,255,0.3) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(79,142,255,0.5) !important;
    }

    /* Input */
    .stTextInput > div > div > input {
        background: #1a2235 !important;
        border: 1px solid #2a3a55 !important;
        border-radius: 12px !important;
        color: #e8edf5 !important;
        padding: 14px !important;
        font-size: 1rem !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #111827;
        border-radius: 12px;
        padding: 6px;
        gap: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #6b7fa3;
        font-weight: 500;
        padding: 10px 20px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(79,142,255,0.2), rgba(124,58,237,0.2)) !important;
        color: #4f8eff !important;
        font-weight: 600 !important;
    }

    /* Success/Info boxes */
    .stSuccess, .stInfo {
        border-radius: 12px !important;
    }

    div[data-testid="stMarkdownContainer"] p {
        color: #c8d4e8;
        line-height: 1.8;
    }
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown("""
<div class="header-box">
    <h1>🎓 Smart Study Assistant</h1>
    <p>Upload your PDF and instantly get answers, summaries, quizzes and more!</p>
</div>
""", unsafe_allow_html=True)

# ---- Sidebar ----
with st.sidebar:
    st.markdown("## 📄 Upload Your PDF")
    st.markdown("Supported: Syllabus, Notes, Textbook chapters")
    st.divider()

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        with st.spinner("⏳ Processing PDF... Please wait"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            docs   = load_pdf(tmp_path)
            chunks = split_documents(docs)
            vectorstore = create_vectorstore(chunks)
            st.session_state.vectorstore = vectorstore
            os.unlink(tmp_path)

        st.success("✅ PDF processed!")
        st.info(f"📄 Pages: {len(docs)}  |  🧩 Chunks: {len(chunks)}")

    st.divider()
    st.markdown("**🔧 Powered by:**")
    st.markdown("🔗 LangChain  \n🗄️ ChromaDB  \n🤖 Groq Llama 3  \n🤗 HuggingFace")

# ---- Main Content ----
if "vectorstore" in st.session_state:

    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Ask Questions",
        "📝 Summary",
        "🎯 Quiz Generator",
        "🔍 Important Topics"
    ])

    with tab1:
        st.markdown("### 💬 Ask Anything From Your PDF")
        question = st.text_input("", placeholder="e.g. What is Software Engineering? Explain in detail.")

        if st.button("🚀 Get Answer"):
            if question.strip():
                with st.spinner("🔍 Thinking..."):
                    answer = answer_question(question, st.session_state.vectorstore)
                st.markdown(f"""
                <div class="answer-box">
                    <div class="answer-label">🤖 AI Answer</div>
                    {answer.replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Please type a question!")

    with tab2:
        st.markdown("### 📝 Document Summary")
        st.markdown("Get a complete detailed summary of your entire PDF")
        if st.button("📝 Generate Summary"):
            with st.spinner("⏳ Generating summary..."):
                summary = generate_summary(st.session_state.vectorstore)
            st.markdown(f"""
            <div class="answer-box">
                <div class="answer-label">📝 Summary</div>
                {summary.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.markdown("### 🎯 Auto Quiz Generator")
        st.markdown("Auto-generate MCQ questions from your PDF")
        if st.button("🎯 Generate Quiz"):
            with st.spinner("⏳ Creating quiz..."):
                quiz = generate_quiz(st.session_state.vectorstore)
            st.markdown(f"""
            <div class="answer-box">
                <div class="answer-label">🎯 Quiz Questions</div>
                {quiz.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)

    with tab4:
        st.markdown("### 🔍 Important Topics")
        st.markdown("Find key topics to focus on for your exam")
        if st.button("🔍 Find Topics"):
            with st.spinner("⏳ Analyzing..."):
                topics = important_topics(st.session_state.vectorstore)
            st.markdown(f"""
            <div class="answer-box">
                <div class="answer-label">🔍 Key Topics</div>
                {topics.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)

else:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="feature-card">
            <h3>💬 Ask Questions</h3>
            <p>Ask anything from your uploaded PDF and get detailed answers instantly</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="feature-card">
            <h3>📝 Get Summary</h3>
            <p>Get a comprehensive bullet point summary of your entire document</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="feature-card">
            <h3>🎯 Generate Quiz</h3>
            <p>Auto-generate MCQ questions to test your knowledge from the PDF</p>
        </div>""", unsafe_allow_html=True)