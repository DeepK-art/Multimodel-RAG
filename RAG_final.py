import streamlit as st
import pdfplumber
import fitz
import google.generativeai as genai
import numpy as np
from PIL import Image
import io
import os
from transformers import CLIPProcessor, CLIPModel
import torch
import faiss
import tiktoken
import json
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# ===== CONFIGURATION =====
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
token_encoder = tiktoken.encoding_for_model("gpt-4")

# ===== STREAMLIT UI =====
st.set_page_config(page_title="📄 Multimodal RAG ", layout="wide")
st.title("📚 Multimodal PDF RAG with Text, Tables (JSON), and Image Summaries")

st.sidebar.title("Settings ⚙️")
api_key = st.sidebar.text_input("Enter your Gemini API Key:", type="password")
top_k = st.sidebar.slider("Number of Chunks to Retrieve", 1, 10, 3)
max_tokens_limit = st.sidebar.number_input("Max Tokens for Retrieved Context", 100, 6000, 3000)

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
query = st.text_input("Enter your question:")

# ===== UTILITY FUNCTIONS =====

def caption_image_with_gemini(image, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash-001")
    response = model.generate_content([
        "Describe this image in 1-2 sentences, focusing on its content, without losing any numeric values.",
        image
    ])
    caption = response.text.strip()
    return caption

def table_to_json_with_gemini(image, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash-001")
    response = model.generate_content([
        "Understand the table structure logically and convert the table from this image to a JSON object. Ensure the JSON represents the table structure with headers and rows accurately.",
        image
    ])
    try:
        json_str = response.text.strip()
        if json_str.startswith("```json") and json_str.endswith("```"):
            json_str = json_str[7:-3].strip()
        return json.loads(json_str)
    except Exception as e:
        st.error(f"Error parsing table JSON: {str(e)}")
        return []

def extract_pdf_content(pdf_file, api_key):
    texts = []
    image_captions = []
    table_jsons = []

    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.read())

    # Extract text using pdfplumber
    with pdfplumber.open("temp.pdf") as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                # Use nltk sentence tokenizer instead of splitting by \n\n
                sentences = sent_tokenize(text)
                texts.extend([sentence.strip() for sentence in sentences if sentence.strip()])

    # Extract images and tables using fitz
    pdf_doc = fitz.open("temp.pdf")
    for page_num in range(len(pdf_doc)):
        page = pdf_doc[page_num]
        
        # Extract images
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            caption = caption_image_with_gemini(image, api_key)
            image_captions.append(caption)

        # Extract table images
        page_obj = pdf.pages[page_num]
        tables = page_obj.extract_tables()
        for table_idx, table in enumerate(tables):
            if table:
                table_bbox = page_obj.find_tables()[table_idx].bbox
                rect = fitz.Rect(table_bbox)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                table_json = table_to_json_with_gemini(image, api_key)
                if table_json:
                    table_jsons.append(table_json)

    os.remove("temp.pdf")   
    return texts, image_captions, table_jsons

def get_text_embeddings(texts):
    embeddings = []
    if texts:
        text_inputs = clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_embeddings = clip_model.get_text_features(**text_inputs).cpu().numpy()
        for i, emb in enumerate(text_embeddings):
            embeddings.append((emb / np.linalg.norm(emb), texts[i]))
    return embeddings

def build_faiss_index(embeddings):
    dim = embeddings[0][0].shape[0]
    index = faiss.IndexFlatIP(dim)
    vectors = np.array([emb[0] for emb in embeddings]).astype("float32")
    index.add(vectors)

    # 🔢 Display Vectors & Plot
    st.markdown("## 🔍 Vector Insight")

    for i, (vector, chunk_text) in enumerate(embeddings):
        st.markdown(f"### 🧩 Chunk {i+1}")

        with st.expander(f"Show Vector for Chunk {i+1}"):
            st.write(chunk_text)

            st.subheader("📈 Embedding Plot")
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.plot(vector, marker="o", markersize=2)
            ax.set_title(f"Chunk {i+1} Embedding Vector")
            ax.set_xlabel("Dimension Index")
            ax.set_ylabel("Embedding Value")
            st.pyplot(fig)

    return index, embeddings


def similarity_search(query, index, embeddings, k=3):
    query_inputs = clip_processor(text=[query], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        query_emb = clip_model.get_text_features(**query_inputs).cpu().numpy()[0]
    query_emb = query_emb / np.linalg.norm(query_emb)
    D, I = index.search(np.array([query_emb]).astype("float32"), k)
    results = [embeddings[idx][1] for idx in I[0] if idx < len(embeddings)]
    return results

def count_tokens(text):
    tokens = token_encoder.encode(text)
    return len(tokens)

def build_limited_context(retrieved_texts, max_tokens):
    context = ""
    total_tokens = 0
    for chunk in retrieved_texts:
        chunk_tokens = count_tokens(chunk)
        if total_tokens + chunk_tokens > max_tokens:
            break
        context += chunk + "\n"
        total_tokens += chunk_tokens
    return context, total_tokens

def synthesize_answer(query, context, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-pro")
    final_prompt = f"Question: {query}\n\nContext:\n{context}\n\nProvide a detailed but concise answer."
    response = model.generate_content(final_prompt)
    st.markdown(context)
    return response.text, count_tokens(final_prompt), count_tokens(response.text)


# ===== MAIN FLOW =====
if uploaded_file:
    if not api_key:
        st.warning("⚠️ Please enter your Gemini API key in the sidebar first.")
    else:
        with st.spinner("Processing your PDF..."):
            texts, image_captions, table_jsons = extract_pdf_content(uploaded_file, api_key)
            all_chunks = []

            for text in texts:
                all_chunks.append("[TEXT]\n" + text)

            for caption in image_captions:
                all_chunks.append("[IMAGE SUMMARY]\n" + caption)

            for table_json in table_jsons:
                json_text = json.dumps(table_json, indent=2)
                all_chunks.append("[TABLE JSON]\n" + json_text)

            # Display extracted content
            st.subheader("Extracted Content from PDF")

            if texts:
                st.markdown("### 📝 Extracted Text")
                for idx, text in enumerate(texts, 1):
                    st.write(f"**Text Chunk {idx}:**")
                    st.write(text)
                    st.markdown("---")

            if image_captions:
                st.markdown("### 🖼️ Image Summaries")
                for idx, caption in enumerate(image_captions, 1):
                    st.write(f"**Image {idx} Summary:**")
                    st.write(caption)
                    st.markdown("---")

            if table_jsons:
                st.markdown("### 📊 Table JSONs")
                for idx, table_json in enumerate(table_jsons, 1):
                    st.write(f"**Table {idx} JSON:**")
                    st.json(table_json)
                    st.markdown("---")

            if not all_chunks:
                st.error("No extractable content found in the PDF!")
            else:
                embeddings = get_text_embeddings(all_chunks)
                index, embeddings = build_faiss_index(embeddings)

                if query:
                    with st.spinner("Generating answer..."):
                        retrieved_texts = similarity_search(query, index, embeddings, k=top_k)
                        context, tokens_in = build_limited_context(retrieved_texts, max_tokens_limit)

                        st.subheader("Retrieved Context:")
                        for chunk in retrieved_texts:
                            if chunk.startswith("[TEXT]"):
                                st.markdown("**📝 Text Chunk:**")
                                st.write(chunk.replace("[TEXT]", "").strip())
                            elif chunk.startswith("[IMAGE SUMMARY]"):
                                st.markdown("**🖼️ Image Summary:**")
                                st.write(chunk.replace("[IMAGE SUMMARY]", "").strip())
                            elif chunk.startswith("[TABLE JSON]"):
                                st.markdown("**📊 Table JSON:**")
                                st.json(json.loads(chunk.replace("[TABLE JSON]", "").strip()))

                        SAFE_TOKEN_LIMIT = 900000
                        if tokens_in > SAFE_TOKEN_LIMIT:
                            st.error(f"⚠️ Context too large: {tokens_in} tokens (limit ~1M). Reduce Top-K or chunk size.")
                            st.stop()

                        try:
                            answer, token_in, token_out = synthesize_answer(query, context, api_key)
                            st.subheader("Generated Answer:")
                            st.write(answer)

                            st.subheader("🔢 Token Usage")
                            st.write(f"Prompt Tokens: {token_in}")
                            st.write(f"Answer Tokens: {token_out}")

                        except Exception as e:
                            st.error(f"Error while generating answer: {str(e)}")
else:
    st.info("👆 Upload a PDF to start!")

