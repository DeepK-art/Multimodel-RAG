from fpdf import FPDF

# Create a simple PDF for the research paper
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'A Study on Battery Performance in Electric Vehicles', ln=True, align='C')
        self.ln(5)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, ln=True, align='L')
        self.ln(2)

    def chapter_body(self, text):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 8, text)
        self.ln()

    def table(self, header, data):
        self.set_font('Arial', 'B', 11)
        for col_name in header:
            self.cell(50, 8, col_name, border=1, align='C')
        self.ln()
        self.set_font('Arial', '', 11)
        for row in data:
            for item in row:
                self.cell(50, 8, str(item), border=1, align='C')
            self.ln()
        self.ln(5)

# Create PDF
pdf = PDF()
pdf.add_page()

# Abstract
pdf.chapter_title("Abstract")
pdf.chapter_body("This paper examines the performance metrics of lithium-ion batteries used in electric vehicles (EVs). "
                 "Factors like temperature, cycle count, and discharge rate were analyzed. Results show a significant drop "
                 "in battery capacity at higher temperatures and after repeated charge cycles.")

# Introduction
pdf.chapter_title("1. Introduction")
pdf.chapter_body("Electric vehicles rely heavily on battery performance for range and reliability. "
                 "Lithium-ion batteries, while efficient, degrade under certain conditions. This study investigates "
                 "how key parameters affect battery capacity.")

# Methodology
pdf.chapter_title("2. Methodology")
pdf.chapter_body("We conducted tests across different:\n- Temperatures: 0°C, 25°C, 45°C\n- Charge cycles: 0, 500, 1000 cycles\n"
                 "- Discharge rates: 1C, 2C, 3C")

# Results
pdf.chapter_title("3. Results")

# Table 1
pdf.chapter_body("Table 1: Battery Capacity (%) at Different Temperatures")
header = ['Temperature (°C)', 'Initial Capacity (%)']
data = [['0', '92'], ['25', '100'], ['45', '87']]
pdf.table(header, data)

# Table 2
pdf.chapter_body("Table 2: Capacity Retention After Charge Cycles")
header2 = ['Charge Cycles', 'Capacity (%)']
data2 = [['0', '100'], ['500', '91'], ['1000', '83']]
pdf.table(header2, data2)

# Table 3
pdf.chapter_body("Table 3: Capacity at Different Discharge Rates")
header3 = ['Discharge Rate (C)', 'Capacity (%)']
data3 = [['1C', '100'], ['2C', '95'], ['3C', '89']]
pdf.table(header3, data3)

# Conclusion
pdf.chapter_title("4. Conclusion")
pdf.chapter_body("Temperature, cycle count, and discharge rate all significantly impact EV battery performance. "
                 "Keeping operational temperatures moderate and using slower charge/discharge rates can prolong battery life.")

# Save PDF
output_path = './battery_performance_ev.pdf'
pdf.output(output_path)

output_path
