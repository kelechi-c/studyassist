from docx2pdf import convert
import os

doc_path = "pdfs/docs"

for filename in os.listdir(doc_path):
    if filename.endswith(".docx"):
        convert(f"{doc_path}/{filename}")
        os.remove(f"{doc_path}/{filename}")