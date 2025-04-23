from PyPDF2 import PdfReader, PdfWriter

def extract_last_n_pages(input_pdf_path, output_pdf_path, n_pages=5):
    reader = PdfReader(input_pdf_path)
    writer = PdfWriter()
    
    total_pages = len(reader.pages)
    start_page = max(total_pages - n_pages, 0)  # if n_pages > total_pages, start from 0

    for i in range(4):
        writer.add_page(reader.pages[i])
    
    with open(output_pdf_path, "wb") as f:
        writer.write(f)

# Example usage
input_pdf = "1706.03762v7.pdf"         # Input full pdf file
output_pdf = "first_4_pages.pdf"          # Output trimmed pdf
extract_last_n_pages(input_pdf, output_pdf, n_pages=5)
