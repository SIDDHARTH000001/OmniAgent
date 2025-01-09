import fitz  # PyMuPDF
import os
import shutil
import numpy as np
import torch
import json
from time import sleep
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_imports
from unittest.mock import patch
import docx  # For DOCX file processing

# Set up device and dtype
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def load_model_and_processor():
    def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
        if not str(filename).endswith("modeling_florence2.py"):
            return get_imports(filename)
        imports = get_imports(filename)
        return imports

    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", torch_dtype=torch_dtype, trust_remote_code=True).to(device)

    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
    return model, processor

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
    text_content = []
    for paragraph in doc.paragraphs:
        text_content.append(paragraph.text)
    return "\n".join(text_content)

# Main function to process files and perform OCR using Florence 2B
def process_file_with_florence(file_path, output_folder, verbose=False):
    # Determine file type
    file_extension = os.path.splitext(file_path)[-1].lower()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    ocr_results = []

    if file_extension == ".pdf":
        # Extract images from the PDF
        extract_pages_as_images(file_path, output_folder)

        # Get the list of images
        image_files = [os.path.join(output_folder, file) for file in os.listdir(output_folder) if file.endswith(("png", "jpg", "jpeg"))]

        # Define the task prompt for OCR
        task_prompt = "<OCR>"

        # Process each image with the Florence 2B model
        for idx, image_file in enumerate(image_files):
            if verbose:
                print(f"Processing image {idx + 1} of {len(image_files)}: {image_file}")

            # Load the image
            image = Image.open(image_file)

            # Run the OCR using the Florence model
            result = run_example(image, task_prompt)

            # Append the result to the OCR results list
            ocr_results.append(result)

            # Save the JSON response for each image (optional)
            json_output_path = os.path.join(output_folder, f"image_{idx + 1}_output.json")
            with open(json_output_path, "w") as json_file:
                json.dump(result, json_file, indent=4)

            if verbose:
                print(f"Output saved: {json_output_path}")

            sleep(5)  # To avoid hitting API rate limits

        # Clean up the image files
        shutil.rmtree(output_folder, ignore_errors=True)

    elif file_extension == ".docx":
        # Extract text from the DOCX file
        docx_text = extract_text_from_docx(file_path)
        ocr_results.append({"text": docx_text})

        # Optionally, save the DOCX text to a JSON file
        json_output_path = os.path.join(output_folder, "docx_output.json")
        with open(json_output_path, "w") as json_file:
            json.dump({"text": docx_text}, json_file, indent=4)

    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

    if verbose:
        print("Processing complete. Cleaned up temporary files.")

    # Return the consolidated OCR results
    return ocr_results

# Define the run_example function to utilize the Florence model
def run_example(image, task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    model, processor = load_model_and_processor()

    if task_prompt == '<OCR>':
        image = image.convert("RGB")

    image = np.array(image)

    inputs = processor(text=prompt, images=image, return_tensors="pt").to('cpu', torch.float32)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"].cpu(),
        pixel_values=inputs["pixel_values"].cpu(),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    print("Generated Text:", generated_text)
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.shape[1], image.shape[0])
    )

    return parsed_answer

# Example usage:
# process_file_with_florence("your_file.pdf", "output_folder", verbose=True)
# process_file_with_florence("your_file.docx", "output_folder", verbose=True)
