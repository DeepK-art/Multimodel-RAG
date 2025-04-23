import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import nltk
import PyPDF2
import pytesseract
from PIL import Image
import pandas as pd
import pdfplumber
import os
import matplotlib.pyplot as plt

# nltk.download("punkt")
from nltk.tokenize import sent_tokenize

st.set_page_config(page_title="📚 Retrieval Assistant", layout="wide")

st.title(" Retrieval from data 🔍 🤖 ")
st.markdown("Upload text, PDF, image, CSV, or XLSX files to provide background context")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Session states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "docs" not in st.session_state:
    st.session_state.docs = []
if "index" not in st.session_state:
    st.session_state.index = None

# ----------- File Upload & Preview -----------
uploaded_files = st.file_uploader(
    "📎 Upload multiple files (txt, pdf, jpg/jpeg, csv, xlsx)", 
    type=["txt", "pdf", "jpg", "jpeg", "csv", "xlsx"], 
    accept_multiple_files=True
)

def extract_text(file):
    extension = file.name.split(".")[-1].lower()

    if extension == "txt":
        return file.read().decode("utf-8"), "text"
    
    elif extension == "pdf":
        text_chunks = []
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                # Extract text
                if page.extract_text():
                    text_chunks.append(page.extract_text())

                # Extract tables
                st.markdown("Extracting a table Sir ")
                tables = page.extract_tables()
                for table in tables:
                    # Skip empty tables
                    if not table or not table[0]:
                        continue
                    headers = table[0]
                    for row in table[1:]:
                        key_value_pairs = [
                        f"{headers[i]}: {row[i]}" for i in range(min(len(headers), len(row)))
                    ]
                    text_chunks.append(", ".join(key_value_pairs))
    
        return "\n".join(text_chunks), "mixed"


    elif extension in ["jpg", "jpeg"]:
        image = Image.open(file)
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

        n_boxes = len(ocr_data['text'])
        lines = {}
        for i in range(n_boxes):
            if int(ocr_data['conf'][i]) > 30 and ocr_data['text'][i].strip():
                y = ocr_data['top'][i]
                # Round y to group rows together
                y_group = y // 10 * 10
                lines.setdefault(y_group, []).append(ocr_data['text'][i])

        # Sort lines vertically
        sorted_rows = [lines[key] for key in sorted(lines.keys())]
        formatted_rows = [" | ".join(row) for row in sorted_rows]

        return "\n".join(formatted_rows), "table"

    elif extension == "csv":
        df = pd.read_csv(file)
        rows = df.astype(str).apply(
        lambda row: ", ".join([f"{col}: {row[col]}" for col in df.columns]),
        axis=1
        ).tolist()
        return "\n".join(rows), "table"

    elif extension == "xlsx":
        df = pd.read_excel(file)
        rows = df.astype(str).apply(
        lambda row: ", ".join([f"{col}: {row[col]}" for col in df.columns]),
        axis=1
        ).tolist()
        return "\n".join(rows), "table"

    return "", "unknown"

def smart_chunk(text, mode="text"):
    if mode == "text" or mode == "image":
        return sent_tokenize(text)
    elif mode == "table" or mode == "mixed":
        return text.split("\n")
    else:
        return [text]

if uploaded_files:
    st.subheader("📄 File Preview")
    extracted_docs = []

    for file in uploaded_files:
        st.markdown(f"**{file.name}**")
        raw_text, file_type = extract_text(file)
        edited_text = st.text_area(f"✏️ Edit extracted text from {file.name}:", value=raw_text, height=300)
        extracted_docs.append((edited_text, file_type))

    if st.button("📚 Process Files"):
        with st.spinner("Embedding documents..."):
            all_chunks = []
            for doc, mode in extracted_docs:
                chunks = smart_chunk(doc, mode)
                all_chunks.extend([chunk.strip() for chunk in chunks if chunk.strip() != ""])
            st.session_state.docs = all_chunks

            st.markdown("Chunks being embedded:")
            st.write(st.session_state.docs)

            doc_embeddings = embedder.encode(st.session_state.docs, convert_to_numpy=True)
            dim = doc_embeddings.shape[1]
            idx = faiss.IndexFlatL2(dim)
            idx.add(doc_embeddings)
            st.session_state.index = idx

        st.success("Documents indexed and ready for retrieval!")

# ----------- Retrieval Display -----------
st.markdown("---")
st.subheader("💬 Chat :")

for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["question"])
    with st.chat_message("Retrieved"):
        st.markdown(chat["answer"])

user_query = st.chat_input("Ask something...")

if user_query:
    if not st.session_state.index or not st.session_state.docs:
        st.warning("⚠️ Please upload and process documents first.")
    else:
        query_embedding = embedder.encode([user_query], convert_to_numpy=True)
        distances, indices = st.session_state.index.search(query_embedding, k=3)
        retrieved = "\n".join([st.session_state.docs[i] for i in indices[0]])
        
        ## To display the vectors
        # 🔢 Display Vectors & Plot
        # -------------------------------
        st.markdown("## 🔍 Vector Insight")

        # 📈 Query Vector Plot
        st.subheader("📈 Query Embedding Plot")
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(query_embedding[0], marker="o", markersize=2)
        ax.set_title("Query Embedding Vector")
        ax.set_xlabel("Dimension Index")
        ax.set_ylabel("Embedding Value")
        st.pyplot(fig)

        # 🧬 Raw Query Vector
        with st.expander("🔢 Raw Query Embedding Vector"):
            st.write(query_embedding[0].tolist())

        # 📄 Retrieved Vectors & Chunks
        st.subheader("📎 Retrieved Document Embeddings")
        for rank, i in enumerate(indices[0]):
            chunk = st.session_state.docs[i]
            retrieved_vector = embedder.encode([chunk], convert_to_numpy=True)

            with st.expander(f"📄 Chunk #{rank+1} (Index {i})"):
                st.markdown("**Document Chunk:**")
                st.code(chunk, language="markdown")

                st.markdown("**🔢 Embedding Vector:**")
                st.write(retrieved_vector[0].tolist())

                 # Plot the embedding
                fig, ax = plt.subplots(figsize=(12, 3))
                ax.plot(retrieved_vector[0], marker="x", markersize=2, label=f"Retrieved Vector #{rank + 1}")
                ax.set_title(f"Retrieved Embedding Vector #{rank + 1}")
                ax.set_xlabel("Dimension Index")
                ax.set_ylabel("Embedding Value")
                ax.legend()
                st.pyplot(fig)

                # Display the distance to the query vector
                st.markdown(f"**Distance to Query Vector (L2):** {distances[0][rank]:.4f}")

        ## ---------------

        st.session_state.chat_history.append({
            "question": user_query,
            "answer": retrieved
        })

        with st.chat_message("user"):
            st.markdown(user_query)
        with st.chat_message("Retrieved"):
            st.markdown(retrieved)

# ----------- Reset Button -----------
if st.button("🧼 Clear Chat"):
    st.session_state.chat_history = []
    st.success("Chat history cleared.")
