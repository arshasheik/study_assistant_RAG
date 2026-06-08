# study_assistant_RAG
**# study_assistant-RAG
# 🎓 Smart Student Study Assistant

An AI-powered RAG (Retrieval-Augmented Generation) application that allows students to upload any PDF (syllabus, notes, textbook) and interact with it intelligently.

## ✨ Features

- 💬 **Ask Questions** — Get detailed answers from your PDF
- 📝 **Auto Summary** — Generate comprehensive bullet-point summaries
- 🎯 **Quiz Generator** — Auto-generate MCQ questions for exam prep
- 🔍 **Important Topics** — Find key topics to focus on

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| LangChain | RAG Framework |
| ChromaDB | Vector Database |
| Groq (Llama 3) | Free LLM |
| HuggingFace | Embeddings |
| Streamlit | Web UI |
| PyMuPDF | PDF Reader |

## 🚀 How to Run

1. Clone the repository
2. Create virtual environment
```bash
   python -m venv venv
   venv\Scripts\activate
```
3. Install dependencies
```bash
   pip install -r requirements.txt
```
4. Add your Groq API key in `.env`
5. 5. Run the app
```bash
   streamlit run app.py
```

## 📁 Project Structure

smart_study_assistant/
├── app.py              # Streamlit UI
├── rag_pipeline.py     # RAG logic
├── requirements.txt    # Dependencies
├── .env                # API Key
└── README.md           # This file

## 👨‍💻 Built With

- Python 3.10+
- LangChain 0.2.x
- Streamlit 1.x

## 📌 Note

This app uses only the content from the uploaded PDF to answer questions — ensuring accurate, hallucination-free responses.
**
