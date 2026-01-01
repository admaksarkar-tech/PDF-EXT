import streamlit as st
import pdfplumber
import pandas as pd
import pytesseract
from pdf2image import convert_from_path
import re
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# --- CONFIGURATION ---
st.set_page_config(page_title="Research Paper AI Analyst", layout="wide")

# Sidebar for API Key
st.sidebar.header("🔑 AI Setup")
api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

st.title("📄 Research Paper Intelligence Hub")
st.markdown("Extract precise data (DOI, Tables) and Chat with scanned/digital papers.")

# --- FUNCTIONS ---

def extract_doi(text):
    """Finds DOI using Regex"""
    doi_pattern = r'\b(10[.][0-9]{4,}(?:[.][0-9]+)*/(?:(?!["&\'<>])\S)+)\b'
    match = re.search(doi_pattern, text)
    return match.group(1) if match else "Not found"

def ocr_pdf(uploaded_file):
    """Converts scanned PDF pages to text using OCR"""
    # Save temp file for pdf2image
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    images = convert_from_path("temp.pdf")
    full_text = ""
    for img in images:
        full_text += pytesseract.image_to_string(img)
    return full_text

def process_pdf(uploaded_file, use_ocr_fallback=True):
    """Main processing logic"""
    text_content = ""
    tables = []
    
    with pdfplumber.open(uploaded_file) as pdf:
        # 1. Try Digital Extraction
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_content += page_text + "\n"
            
            # Extract Tables
            page_tables = page.extract_table()
            if page_tables:
                tables.extend(page_tables)
                
    # 2. Fallback to OCR if text is empty (Scanned PDF)
    if not text_content.strip() and use_ocr_fallback:
        st.info("⚠️ Scanned PDF detected. Running OCR (this takes a moment)...")
        text_content = ocr_pdf(uploaded_file)
        
    return text_content, tables

# --- MAIN APP LOGIC ---
uploaded_file = st.file_uploader("Upload Research Paper (PDF)", type="pdf")

if uploaded_file:
    # Create Tabs
    tab1, tab2 = st.tabs(["📊 Precise Data Extractor", "🤖 Chat Bot"])
    
    # Process the file once
    if 'text_content' not in st.session_state:
        with st.spinner("Analyzing document structure..."):
            text, tables = process_pdf(uploaded_file)
            st.session_state.text_content = text
            st.session_state.tables = tables
            st.session_state.doi = extract_doi(text)

    # --- TAB 1: DATA EXTRACTOR ---
    with tab1:
        st.subheader("1. Metadata")
        col1, col2 = st.columns(2)
        col1.metric("DOI Found", st.session_state.doi)
        col2.metric("Char Count", len(st.session_state.text_content))
        
        st.subheader("2. Extracted Tables")
        if st.session_state.tables:
            # Convert list of lists to DataFrame
            df = pd.DataFrame(st.session_state.tables[1:], columns=st.session_state.tables[0])
            st.dataframe(df)
            
            # Export Options
            csv = df.to_csv(index=False).encode('utf-8')
            json_data = df.to_json(orient="records")
            
            c1, c2 = st.columns(2)
            c1.download_button("Download CSV", csv, "data.csv", "text/csv")
            c2.download_button("Download JSON", json_data, "data.json", "application/json")
        else:
            st.warning("No structured tables detected.")

    # --- TAB 2: CHAT BOT ---
    with tab2:
        if not api_key:
            st.error("Please enter an OpenAI API Key in the sidebar to use the Chatbot.")
        else:
            st.subheader("Ask questions about the paper")
            
            # Setup Chat Engine (Only runs once)
            if 'qa_chain' not in st.session_state:
                with st.spinner("Training AI on this paper..."):
                    # Split text into chunks
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    texts = text_splitter.split_text(st.session_state.text_content)
                    docs = [Document(page_content=t) for t in texts]
                    
                    # Create Vector Store
                    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                    vectorstore = FAISS.from_documents(docs, embeddings)
                    
                    # Create Chain
                    st.session_state.qa_chain = RetrievalQA.from_chain_type(
                        llm=ChatOpenAI(openai_api_key=api_key, model_name="gpt-3.5-turbo"),
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever()
                    )
            
            # Chat Interface
            user_query = st.text_input("Ask a question (e.g., 'What is the methodology?')")
            if user_query:
                with st.spinner("Thinking..."):
                    response = st.session_state.qa_chain.run(user_query)
                    st.write(response)