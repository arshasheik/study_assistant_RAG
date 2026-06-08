import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

def load_pdf(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    return chunks

def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    return vectorstore

def answer_question(question, vectorstore):
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        max_tokens=1024
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    prompt = ChatPromptTemplate.from_template("""
You are an expert study assistant helping students understand their study material deeply.
Answer the question in DETAIL using the context provided below.
- Use bullet points where needed
- Explain concepts clearly and thoroughly
- Give examples if mentioned in the context
- Minimum 5-8 lines of explanation
- If the answer is not in the context, say "I couldn't find this in the document."

Context:
{context}

Question:
{question}

Detailed Answer:
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(question)

def generate_summary(vectorstore):
    return answer_question(
        "Give a very detailed and comprehensive summary of the entire document. Cover all major topics, concepts, and key points in bullet points with explanations.",
        vectorstore
    )

def generate_quiz(vectorstore):
    return answer_question(
        "Generate 5 detailed multiple choice questions with 4 options each. Mark the correct answer clearly. Questions should cover the most important topics in the document.",
        vectorstore
    )

def important_topics(vectorstore):
    return answer_question(
        "List all the important topics in this document that a student must study. For each topic give a brief explanation of why it is important and what to focus on.",
        vectorstore
    )