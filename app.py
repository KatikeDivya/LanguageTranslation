from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from bs4 import BeautifulSoup
import requests
import PyPDF2
import langdetect
from nltk.translate.bleu_score import sentence_bleu
import torch

app = Flask(__name__)

# Load the fine-tuned model and tokenizer
model_name_translation = "facebook/mbart-large-50-many-to-many-mmt"  # Path to your fine-tuned model
tokenizer_translation = AutoTokenizer.from_pretrained(model_name_translation, use_fast=False,local_files_only=True)
model_translation = AutoModelForSeq2SeqLM.from_pretrained(model_name_translation,local_files_only=True)

# Predefined list of supported target languages for output
target_languages = {
    "en_XX": "English",
    "ta_IN": "Tamil",
    "te_IN": "Telugu",
    "ml_IN": "Malayalam",
    "mr_IN": "Marathi",
    "hi_IN": "Hindi"
}

def detect_language(text):
    """Detect the language of the given text."""
    return langdetect.detect(text)

def extract_webpage_text(web_url):
    """Extract text content from a web page."""
    headers = {'User-Agent': "Mozilla/5.0"}
    response = requests.get(web_url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch the URL: {response.status_code}")
    soup = BeautifulSoup(response.content, 'html.parser')
    full_text = soup.get_text(separator=' ', strip=True)
    return full_text

def extract_pdf_text(pdf_path):
    """Extract text content from a PDF file."""
    text = ''
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def split_text_into_chunks(text, max_length=512):
    """Split the text into smaller chunks of a defined maximum length."""
    words = text.split()
    chunks = []
    chunk = []
    for word in words:
        chunk.append(word)
        if len(' '.join(chunk)) > max_length:
            chunks.append(' '.join(chunk))
            chunk = []
    if chunk:
        chunks.append(' '.join(chunk))
    return chunks

def calculate_bleu_score(reference_translation, generated_translation):
    """Calculate BLEU score for evaluating the quality of the translation."""
    reference_tokens = [reference_translation.split()]
    generated_tokens = generated_translation.split()
    bleu_score = sentence_bleu(reference_tokens, generated_tokens)
    return bleu_score

def translate_text(text, target_language):
    """Translate the text using the fine-tuned model."""
    inputs_translation = tokenizer_translation(text, return_tensors="pt", padding=True, truncation=True)
    outputs_translation = model_translation.generate(inputs_translation["input_ids"], forced_bos_token_id=tokenizer_translation.lang_code_to_id[target_language], max_length=150, num_beams=5)
    translated_text = tokenizer_translation.decode(outputs_translation[0], skip_special_tokens=True)
    return translated_text

@app.route('/')
def home():
    """Home route for rendering the HTML form."""
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    """Translate text from the selected input (URL, PDF, or direct text)."""
    input_type = request.form['input_type']
    
    # Process the input based on the type (URL, PDF, or text)
    if input_type == 'url':
        web_url = request.form['web_url']
        full_text = extract_webpage_text(web_url)
    elif input_type == 'pdf':
        pdf_path = request.form['pdf_path']
        full_text = extract_pdf_text(pdf_path)
    elif input_type == 'text':
        full_text = request.form['text']
    else:
        return "Invalid input type"

    # Detect the language of the input text
    detected_language = detect_language(full_text)

    # Get the target language for translation
    target_language = request.form['target_language']
    if target_language not in target_languages:
        return f"Sorry, we only support translations to these languages: {', '.join(target_languages.values())}"

    # Split the text into smaller chunks for translation
    text_chunks = split_text_into_chunks(full_text)
    
    # Translate each chunk
    translated_chunks = [translate_text(chunk, target_language) for chunk in text_chunks]
    final_translation = ' '.join(translated_chunks)

    # Render the result in the HTML template
    return render_template('result.html', 
                           original_text=full_text, 
                           translated_text=final_translation, 
                           detected_language=detected_language, 
                           target_language=target_languages[target_language])

if __name__ == '__main__':
    app.run(debug=True)

