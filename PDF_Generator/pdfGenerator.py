from fpdf import FPDF
import matplotlib.pyplot as plt

# Step 1: Create a bar plot and save it
categories = ['Apples', 'Bananas', 'Cherries', 'Dates']
values = [23, 17, 35, 29]

plt.figure(figsize=(6, 4))
plt.bar(categories, values, color='skyblue')
plt.title('Fruit Sales Report')
plt.xlabel('Fruit')
plt.ylabel('Sales (Units)')
graph_path = './fruit_sales_graph.png'
plt.savefig(graph_path)
plt.close()

# Step 2: Create a short PDF with FPDF
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Sample Report: Fruit Sales Analysis', 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_table(self, data):
        self.set_font('Arial', '', 10)  # Smaller font for more columns
        num_cols = len(data[0])  # Number of columns
        col_width = self.w / (num_cols + 1)  # Dynamic column width
        th = self.font_size

        # Header
        headers = ['Fruit', 'Sales (Units)', 'Revenue ($)', 'Region', 'Growth (%)']
        for header in headers:
            self.cell(col_width, th * 2, header, border=1, align='C')
        self.ln(th * 2)

        # Data rows
        for row in data:
            for item in row:
                self.cell(col_width, th * 2, str(item), border=1, align='C')
            self.ln(th * 2)

# Create PDF
pdf = PDF()
pdf.add_page()

# Add text
intro_text = (
    "This report provides an overview of the fruit sales for the current quarter. "
    "The data represents sales figures for four major fruits across different regions, "
    "including units sold, revenue, and growth percentages. Insights derived from "
    "the table and graph help to determine inventory and marketing strategies."
)
pdf.chapter_title('Introduction')
pdf.chapter_body(intro_text)

# Add complex table
table_data = [
    ['Apples', 23, 1150, 'North', 5.2],
    ['Bananas', 17, 680, 'South', -2.1],
    ['Cherries', 35, 2450, 'East', 8.7],
    ['Dates', 29, 1740, 'West', 3.4]
]
pdf.chapter_title('Sales Table')
pdf.add_table(table_data)

# Add graph
pdf.chapter_title('Sales Graph')
pdf.image(graph_path, w=150)

# Save the PDF
pdf_output_path = './sample_fruit_sales_report.pdf'
pdf.output(pdf_output_path)

print(f"PDF saved at: {pdf_output_path}")