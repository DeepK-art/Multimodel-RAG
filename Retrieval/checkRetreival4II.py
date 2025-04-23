import streamlit as st
import os
from unstructured.partition.pdf import partition_pdf
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import base64
import faiss
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration, DonutProcessor, DonutForConditionalGeneration
from PIL import Image as PILImage
import io
import torch

# Setup API keys
os.environ["GROQ_API_KEY"] = "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"  # <-- Replace here
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Embedding function
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize FAISS index
index = None
docstore = {}

# Load BLIP2 model (small one)
device = torch.device("cpu" if torch.cuda.is_available() is False else "cuda")

# Ensure correct model loading with device handling
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model_blip = model_blip.to(device)  # Move model to the correct device

# Load Donut model for extracting structured data from PDF forms or tables
donut_processor = DonutProcessor.from_pretrained("naver/donut-base")
donut_model = DonutForConditionalGeneration.from_pretrained("naver/donut-base").to(device)

def partition_pdf_file(file_path):
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )
    return chunks

def extract_text_from_chunks(chunks):
    texts = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            texts.append(chunk.text)  # Ensure it's a string here
    st.code(texts)
    return texts

def extract_tables_from_chunks(chunks):
    tables = []
    for chunk in chunks:
        if "Table" in str(type(chunk)):
            # Using Donut to extract tables and structured data
            table_data = chunk.metadata.text_as_html
            tables.append(table_data)
    st.code(tables)
    return tables

def extract_images_from_chunks(chunks):
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64

# Donut Model Processing for Table Extraction
def extract_table_with_donut(image_bytes):
    inputs = donut_processor(image_bytes, return_tensors="pt").to(device)
    output = donut_model.generate(**inputs)
    decoded_output = donut_processor.decode(output[0], skip_special_tokens=True)
    return decoded_output

def get_summary_from_table(table_html):
    prompt = ChatPromptTemplate.from_template("""
    Summarize the information in this HTML table, including the column headers:
    {element}
    """)
    model = ChatGroq(temperature=0.5, model="mixtral-8x7b-32768")
    chain = prompt | model | StrOutputParser()
    summary = chain.invoke({"element": table_html})
    st.markdown("Summary of the table:\n")
    st.code(summary)
    return summary

def get_summary_from_image(img_b64):
    image_bytes = base64.b64decode(img_b64)
    image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")

    # Process image with BLIP
    inputs = processor(image, return_tensors="pt").to(device)  # Ensure tensors are moved to the correct device
    output = model_blip.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

def add_to_faiss(text, doc_id):
    global index
    if isinstance(text, str):  # Ensure it's a valid string
        embedding = embedding_function.embed_query(text)
        if index is None:
            index = faiss.IndexFlatL2(len(embedding))
        index.add(np.array([embedding], dtype=np.float32))
        docstore[len(docstore)] = {"doc_id": doc_id, "content": text}
    else:
        print(f"Error: Expected a string but got {type(text)}")

def search_faiss(query, top_k=3):
    global index
    if index is None:
        return []

    query_embedding = embedding_function.embed_query(query)
    D, I = index.search(np.array([query_embedding], dtype=np.float32), top_k)
    results = []
    for idx in I[0]:
        if idx != -1:
            results.append(docstore[idx]["content"])
    return results

# Streamlit app
st.title('🧠 Multimodal RAG Chatbot')

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file is not None:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    chunks = partition_pdf_file("uploaded.pdf")

    texts = extract_text_from_chunks(chunks)
    tables = extract_tables_from_chunks(chunks)
    images = extract_images_from_chunks(chunks)

    st.success("PDF Processed!")

    # Summarize and store in FAISS
    for text in texts:
        if isinstance(text, str):  # Ensure it's a valid string
            add_to_faiss(text, doc_id="text")
        else:
            print(f"Skipped invalid text: {type(text)}")

    # Use AI model to extract and summarize tables
    for table in tables:
        summary = get_summary_from_table(table)  # Passing the full table HTML
        add_to_faiss(summary, doc_id="table")

    # Use AI model for image captioning
    for img_b64 in images:
        caption = get_summary_from_image(img_b64)
        add_to_faiss(caption, doc_id="image")

    st.info("Summaries and Captions indexed in FAISS!")

user_query = st.text_input("Ask a question:")

if user_query:
    retrieved_chunks = search_faiss(user_query, top_k=3)

    context = "\n\n".join(retrieved_chunks)

    full_prompt = f"""
    You are an AI assistant. Answer the question below ONLY based on the following context:

    Context:
    {context}

    Question:
    {user_query}
    """

    # model = ChatGroq(temperature=0.2, model="mixtral-8x7b-32768")
    # prompt = ChatPromptTemplate.from_template(full_prompt)
    # chain = prompt | model | StrOutputParser()

    # final_answer = chain.invoke({"element": context})
    st.subheader("Answer:")
    st.write(full_prompt)
