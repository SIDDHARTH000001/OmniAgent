import fitz  # PyMuPDF
import os
import shutil
import json
from PIL import Image
import numpy as np
import docx  # For DOCX file processing
from time import sleep
import google.generativeai as genai
import os
import configparser

# Load configuration
config = configparser.ConfigParser()
config.read('config.properties')

genai.configure(api_key=config['google']['api_key'])

# Set up Google Vision model
def load_google_vision_model():
    return genai.GenerativeModel("gemini-1.5-flash")

# Utility function to extract images from PDF pages
def extract_pages_as_images(pdf_path, output_folder):
    doc = fitz.open(pdf_path)
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        image_list = page.get_images(full=True)

        for image_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"{output_folder}/page_{page_number + 1}_image_{image_index + 1}.{image_ext}"
            with open(image_filename, "wb") as image_file:
                image_file.write(image_bytes)

# Utility function to extract text from DOCX files
def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    text_content = [paragraph.text for paragraph in doc.paragraphs]
    return "\n".join(text_content)

# Main function to process files and perform OCR using Google Vision
def process_file_with_google_vision(file_path, output_folder, verbose=False):
    file_extension = os.path.splitext(file_path)[-1].lower()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    ocr_results = []
    model = load_google_vision_model()

    if file_extension == ".pdf":
        
        with open(file_path, "rb") as f:
            sample_pdf = genai.upload_file(f, mime_type="application/pdf")
        response = model.generate_content(["extract all the text word by word from this pdf file.", sample_pdf])
        return response.text

    elif file_extension == ".docx":
        docx_text = extract_text_from_docx(file_path)
        ocr_results.append({"text": docx_text})

        json_output_path = os.path.join(output_folder, "docx_output.json")
        with open(json_output_path, "w") as json_file:
            json.dump({"text": docx_text}, json_file, indent=4)

    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

    if verbose:
        print("Processing complete. Cleaned up temporary files.")

    return ocr_results
