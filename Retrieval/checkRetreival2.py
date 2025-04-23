import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import DonutProcessor, VisionEncoderDecoderModel, AutoTokenizer
from pdf2image import convert_from_bytes
from PIL import Image
import pandas as pd
import faiss
import numpy as np
import torch
import pdfplumber
import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

# ---- Device & Model Loading ----
device = "cpu"  # Force to use CPU
st.set_page_config(page_title="📚 Multimodal RAG with Donut", layout="wide")
st.title("📖 Advanced Multimodal RAG Chatbot 🚀")
st.markdown("Upload **text, PDFs, or images** with complex tables/forms for Retrieval-Augmented Generation.")

@st.cache_resource
def load_donut():
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
    model.to(device)
    return processor, model

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

processor, donut_model = load_donut()
embedder = load_embedder()

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

# ---- Utility: Donut Extraction ----
def extract_structured_from_image(image: Image.Image, question="Read and transcribe the entire document."):
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    
    # Updated Prompt
    task_prompt = question

    tokenizer = AutoTokenizer.from_pretrained("naver-clova-ix/donut-base", use_fast=True)
    decoder_input_ids = tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

    outputs = donut_model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=1536,  # Increased for long documents
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
    )
    
    sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    try:
        sequence = processor.token2json(sequence)  # parse into structured JSON if possible
    except:
        pass  # fallback to plain text if JSON parsing fails
    return sequence

# ---- Utility: Extract Text/Structured Data ----
def extract_text(file):
    ext = file.name.split(".")[-1].lower()
    if ext == "txt":
        return file.read().decode("utf-8")
    elif ext in ["jpg", "jpeg", "png"]:
        img = Image.open(file).convert("RGB")
        return extract_structured_from_image(img)
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
