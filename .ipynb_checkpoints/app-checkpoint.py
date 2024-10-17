from flask import Flask, request, jsonify, render_template
from transformers import MBart50Tokenizer, MBartForConditionalGeneration
import requests
from bs4 import BeautifulSoup
import PyPDF2
import langdetect

app = Flask(__name__)

# Load the pre-trained model and tokenizer for translation
model_name_translation = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer_translation = MBart50Tokenizer.from_pretrained(model_name_translation, src_lang="en_XX")
model_translation = MBartForConditionalGeneration.from_pretrained(model_name_translation)

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
    return langdetect.detect(text)

def extract_webpage_text(web_url):
    headers = {'User-Agent': "Mozilla/5.0"}
    response = requests.get(web_url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch the URL: {response.status_code}")
    
    soup = BeautifulSoup(response.content, 'html.parser')
    full_text = soup.get_text(separator=' ', strip=True)
    return full_text

def extract_pdf_text(pdf_path):
    text = ''
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def split_text_into_chunks(text, max_length=512):
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

def translate_text(text, target_language):
    inputs_translation = tokenizer_translation(text, return_tensors="pt", padding=True, truncation=True)
    
    # Set the target language for the model
    tokenizer_translation.src_lang = "en_XX"  # Assuming the source text is in English
    model_translation.config.forced_bos_token_id = tokenizer_translation.lang_code_to_id[target_language]
    
    # Generate the translation
    outputs_translation = model_translation.generate(inputs_translation["input_ids"], max_length=150, num_beams=5)
    
    translated_text = tokenizer_translation.decode(outputs_translation[0], skip_special_tokens=True)
    return translated_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    text = data.get('text', '')
    target_language = data.get('target_language', 'en_XX')
    
    # Detect the input language
    detected_language = detect_language(text)

    # Split the text into chunks if needed
    text_chunks = split_text_into_chunks(text)

    # Translate each chunk
    translated_chunks = [translate_text(chunk, target_language) for chunk in text_chunks]
    final_translation = ' '.join(translated_chunks)

    return jsonify({
        'detected_language': detected_language,
        'translated_text': final_translation
    })

if __name__ == "__main__":
    app.run(debug=True)

