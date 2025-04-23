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

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run the app:**

```bash
streamlit run app.py
```

### 🚀 Usage

Upload a PDF containing text, tables, and/or images.

Enter your Google Gemini API Key.

Ask any question about the document!

Retrieved context and generated answer will appear.

Explore extracted content and visualized embeddings!

### 💚 Example Queries

"Summarize the financial performance shown in the tables."

"Describe the key findings mentioned in the third page images."

"What is the value of sales in 2022?"

"Summarize the pipeline maintenance steps."

### 🛠️ Requirements

Python 3.9+

API access to Google Gemini (https://aistudio.google.com/)

### 👋 Contributions

Feel free to open issues or pull requests if you'd like to improve this project! ✨


