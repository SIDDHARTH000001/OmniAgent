import boto3
import json
import base64
import pprint
import fitz  # PyMuPDF
import os
import docx  # For DOCX file processing
import configparser


# Load configuration
config = configparser.ConfigParser()
config.read('config.properties')


# Define AWS credentials and setup session
AWS_ACCESS_KEY_ID = config['aws']['access_key_id']
AWS_SECRET_ACCESS_KEY = config['aws']['secret_access_key']
REGION_NAME = config['aws']['region']


boto3.setup_default_session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=REGION_NAME
)

# Initialize Bedrock client
bedrock_client = boto3.client(service_name='bedrock-runtime', region_name=REGION_NAME)

# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Utility function to extract images from PDF pages
def extract_pages_as_images(pdf_path, output_folder):
    doc = fitz.open(pdf_path)
    images = []
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
            images.append(image_filename)  # Collect image paths
    return images

# Utility function to extract text from DOCX files
def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    text_content = [paragraph.text for paragraph in doc.paragraphs]
    return "\n".join(text_content)

# Function to perform OCR using Claude
def perform_ocr_with_claude(image_path):
    base64_image = encode_image(image_path)

    # Define the request payload for Bedrock
    payload = {
        "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
        "contentType": "application/json",
        "accept": "application/json",
        "body": json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": "Perform OCR on the image and provide the extracted text."
                        }
                    ]
                }
            ]
        })
    }

    # Make the API call to Bedrock
    response = bedrock_client.invoke_model(**payload)
    response_content = response['body'].read().decode('utf-8')
    response_json = json.loads(response_content)
    
    return response_json['content'][0]['text']

# Main function to process files and perform OCR using Claude
def process_file_with_claude(file_path, output_folder, verbose=False):
    file_extension = os.path.splitext(file_path)[-1].lower()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    ocr_results = []

    if file_extension == ".pdf":
        # Extract images from PDF and perform OCR
        images = extract_pages_as_images(file_path, output_folder)
        for image_path in images:
            extracted_text = perform_ocr_with_claude(image_path)
            ocr_results.append({"text": extracted_text})

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
