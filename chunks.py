import json
from PyPDF2 import PdfReader
import textwrap

#  Correct path to your PDF file
pdf_path = "C:/Users/hp/Downloads/Dataset_pdf/data_science_interview_questions.pdf"

# Load the PDF
reader = PdfReader(pdf_path)

# Extract all text from the PDF
all_text = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        all_text += text + "\n"

# Chunk the text into ~500-word segments
word_chunks = textwrap.wrap(all_text, 500)

# Prepare chunks for saving as JSON
chunk_data = [{"id": i, "text": chunk} for i, chunk in enumerate(word_chunks)]

# Save chunks to a JSON file
output_path = "C:/Users/hp/Downloads/Dataset_pdf/chunks.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(chunk_data, f, indent=2, ensure_ascii=False)

print(f"Saved {len(chunk_data)} chunks to chunks.json")
