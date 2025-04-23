import streamlit as st
from sentence_transformers import SentenceTransformer
from pdf2image import convert_from_bytes
from PIL import Image
import pandas as pd
import faiss
import numpy as np
import torch
import pdfplumber
import nltk
import easyocr
from nltk.tokenize import sent_tokenize

nltk.download("punkt")

# ---- Device & Model Loading ----
device = "cpu"  # Force to use CPU
st.set_page_config(page_title="📚 Multimodal RAG with OCR", layout="wide")
st.title("📖 Advanced Multimodal RAG Chatbot 🚀")
st.markdown("Upload **text, PDFs, or images** for Retrieval-Augmented Generation.")

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# Initialize EasyOCR reader
reader = easyocr.Reader(["en"])

# ---- Utility: Flatten Multirow Header Tables ----
def flatten_multiheader_table(df: pd.DataFrame) -> list:
    if isinstance(df.columns, pd.MultiIndex):
        flat_columns = []
        for col in df.columns:
            clean_col = " - ".join([str(c).strip() for c in col if "Unnamed" not in str(c)])
            flat_columns.append(clean_col)
        df.columns = flat_columns
    else:
        df.columns = [str(col).strip() for col in df.columns]

    chunks = []
    for idx, row in df.iterrows():
        parts = []
        for col, val in row.items():
            if pd.notnull(val) and str(val).strip() != "":
                parts.append(f"{col}: {val}")
        text_chunk = ", ".join(parts)
        chunks.append(text_chunk)
    return chunks

# ---- Utility: Extract Text from Images using EasyOCR ----
def extract_text_from_image(image: Image.Image):
    img = np.array(image)
    result = reader.readtext(img)
    text = " ".join([res[1] for res in result])  # Extracting the text part
    return text

# ---- Utility: Extract Text/Structured Data from Files ----
def extract_text(file):
    ext = file.name.split(".")[-1].lower()
    if ext == "txt":
        return file.read().decode("utf-8")
    elif ext in ["jpg", "jpeg", "png"]:
        img = Image.open(file).convert("RGB")
        return extract_text_from_image(img)
    elif ext == "pdf":
        text_chunks = []
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                # Extract raw text
                if page.extract_text():
                    text_chunks.append(page.extract_text())

                # Extract structured tables
                tables = page.extract_tables()
                for table in tables:
                    if not table or not table[0]:
                        continue
                    df = pd.DataFrame(table[1:], columns=table[0])
                    table_chunks = flatten_multiheader_table(df)
                    text_chunks.extend(table_chunks)
        return "\n".join(text_chunks)
    else:
        return ""

# ---- Streamlit UI ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "docs" not in st.session_state:
    st.session_state.docs = []
if "index" not in st.session_state:
    st.session_state.index = None

uploaded_files = st.file_uploader("📎 Upload files", type=["txt", "pdf", "jpg", "jpeg", "png"], accept_multiple_files=True)

# ---- Preview & Process ----
if uploaded_files:
    st.subheader("📄 Preview & Edit Extracted Text")
    extracted_docs = []
    for file in uploaded_files:
        st.markdown(f"**{file.name}**")
        extracted_text = extract_text(file)
        edited = st.text_area(f"📝 {file.name}", value=str(extracted_text), height=250)
        extracted_docs.append(edited)

    if st.button("📚 Process Files"):
        with st.spinner("Embedding..."):
            st.session_state.docs = sum([sent_tokenize(doc) for doc in extracted_docs], [])

            st.markdown("Chunks being embedded:")
            st.write(st.session_state.docs)

            embeddings = embedder.encode(st.session_state.docs, convert_to_numpy=True)
            dim = embeddings.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(embeddings)
            st.session_state.index = index
        st.success("✅ Documents processed and indexed!")

# ---- Chat UI ----
st.markdown("---")
st.subheader("💬 Chat with Knowledge Base")

for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["question"])
    with st.chat_message("assistant"):
        st.markdown(chat["answer"])

user_query = st.chat_input("Ask something...")

if user_query:
    if not st.session_state.index:
        st.warning("Please upload and process documents first.")
    else:
        query_embed = embedder.encode([user_query], convert_to_numpy=True)
        distances, indices = st.session_state.index.search(query_embed, k=3)
        retrieved_context = "\n".join([st.session_state.docs[i] for i in indices[0]])

        answer = f"**Retrieved Context:**\n{retrieved_context}"

        st.session_state.chat_history.append({
            "question": user_query,
            "answer": answer
        })

        with st.chat_message("user"):
            st.markdown(user_query)
        with st.chat_message("assistant"):
            st.markdown(answer)

if st.button("🧼 Clear Chat"):
    st.session_state.chat_history = []
    st.success("Chat history cleared.")
