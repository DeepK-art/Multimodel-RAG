# 📚 Multimodal PDF RAG

A Streamlit app for **Multimodal Retrieval-Augmented Generation (RAG)** that extracts and reasons over **Text**, **Tables**, and **Images** from PDFs.  
It combines **CLIP**, **Google Gemini**, and **FAISS** for powerful multimodal retrieval and question answering.

---

## 🚀 Features

- Extract **text** from PDFs using `pdfplumber`
- Extract **tables** as structured **JSON** using `fitz` + `Google Gemini`
- Extract and **caption images** from PDFs using `Google Gemini`
- Build a **FAISS** vector database with CLIP embeddings
- Perform **semantic search** across all extracted content
- Generate detailed answers using **Google Gemini Pro 1.5**
- Streamlit **UI** for easy interaction and exploration

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/)
- [Hugging Face Transformers (CLIP)](https://huggingface.co/openai/clip-vit-base-patch32)
- [Google Generative AI (Gemini 1.5 Pro)](https://ai.google.dev/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [pdfplumber](https://github.com/jsvine/pdfplumber)
- [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/en/latest/)
- [nltk](https://www.nltk.org/)

---

## 📥 Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/multimodal-pdf-rag.git
   cd multimodal-pdf-rag
