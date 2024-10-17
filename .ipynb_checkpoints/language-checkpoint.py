from langchain.llms import GooglePalm
from bs4 import BeautifulSoup
import requests
import pdfplumber
import PyPDF2

# Initialize the language model with a temperature parameter
api_key = "AIzaSyDyCF6Bw7I_ZIszWsWvb-eSESWzWVXsjbo"
llm = GooglePalm(google_api_key=api_key, temperature=0.7)

# Function to make a web request and extract HTML content
def fetch_webpage(url):
    headers = {'User-Agent': "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch the URL: {response.status_code}")
    
    return response.content

# Function to parse and extract text from a webpage
def extract_webpage_text(web_url):
    # Step 1: Fetch the webpage HTML
    content = fetch_webpage(web_url)
    
    # Step 2: Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(content, 'html.parser')
    
    # Step 3: Extract all the text from the webpage
    full_text = soup.get_text(separator=' ', strip=True)
    
    return full_text

# Function to extract text from a PDF using PyPDF2
def extract_pdf_text(pdf_path):
    text = ''
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)
    
        # Loop through each page and extract text
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    
    return text

# Function to remove special characters from input text
def remove_special_characters(text):
    return ''.join(e for e in text if e.isalnum() or e.isspace())

# Function to convert text to lowercase
def to_lowercase(text):
    return text.lower()

# Function to remove extra whitespace from text
def remove_extra_whitespace(text):
    return ' '.join(text.split())

# Function to handle numbers in text
def handle_numbers(text):
    # Example: Convert '123' to 'one hundred and twenty-three'
    return text  # Placeholder for number handling logic

# Function to correct spelling in text
def correct_spelling(text):
    # Placeholder for spelling correction logic
    return text

# Function to detect the language of the text
def detect_language(text):
    # Placeholder for language detection logic
    return 'en'  # Defaulting to English

# Function to tokenize the text
def tokenize_text(text):
    return text.split()

# Function to remove stop words from tokenized text
def remove_stop_words(tokens):
    stop_words = {'the', 'is', 'in', 'at', 'of'}  # Placeholder stop words
    return [token for token in tokens if token not in stop_words]

# Preprocessing function that chains all the cleaning steps
def preprocess_translation_input(user_input):
    # Step 1: Remove special characters
    cleaned_text = remove_special_characters(user_input)
    
    # Step 2: Convert to lowercase
    cleaned_text = to_lowercase(cleaned_text)
    
    # Step 3: Remove extra whitespace
    cleaned_text = remove_extra_whitespace(cleaned_text)
    
    # Step 4: Handle numbers
    cleaned_text = handle_numbers(cleaned_text)
    
    # Step 5: Correct spelling (placeholder)
    cleaned_text = correct_spelling(cleaned_text)
    
    # Step 6: Detect the language
    detected_lang = detect_language(cleaned_text)
    
    # Step 7: Tokenize the text
    tokens = tokenize_text(cleaned_text)
    
    # Step 8: Remove stop words
    filtered_tokens = remove_stop_words(tokens)
    
    return cleaned_text, detected_lang, filtered_tokens

# Function to split text into smaller chunks
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

# Main chatbot function to handle user interaction
def chatbot():
    print("Welcome to the translation chatbot!")

    while True:
        try:
            # Step 1: Get the input type from the user
            print("\nHow can I help you? (Enter 'url', 'pdf', 'text' to translate or 'exit' to quit):")
            input_type = input().strip().lower()
            
            if input_type == 'exit':
                print("Goodbye!")
                break

            # Step 2: Extract the appropriate text based on the input type
            if input_type == 'url':
                print("Enter the URL to translate:")
                web_url = input().strip()
                full_text = extract_webpage_text(web_url)
            elif input_type == 'pdf':
                print("Enter the PDF file path to translate:")
                pdf_path = input().strip()
                full_text = extract_pdf_text(pdf_path)
            elif input_type == 'text':
                print("Enter the text to translate:")
                full_text = input().strip()
            else:
                print("Invalid input type. Please choose 'url', 'pdf', 'text', or 'exit'.")
                continue

            # Step 3: Preview the extracted text
            print(f"Extracted Text: {full_text[:500]}...")  # Show the first 500 characters as preview

            # Step 4: Ask for the target language
            print("Enter the target language code (e.g., 'hi' for Hindi, 'bn' for Bengali, 'ta' for Tamil):")
            target_language = input().strip()

            # Step 5: Preprocess the extracted text
            preprocessed_data = preprocess_translation_input(full_text)
            cleaned_text, detected_lang, filtered_tokens = preprocessed_data
            print(f"Preprocessed Text: {cleaned_text[:500]}...")  # Show cleaned version

            # Step 6: Split the full text into chunks if needed
            text_chunks = split_text_into_chunks(cleaned_text)

            # Step 7: Translate each chunk
            translated_chunks = [llm(f"Translate {chunk} to {target_language}") for chunk in text_chunks]
            final_translation = ' '.join(translated_chunks)

            # Step 8: Print the translated text
            print(f"Translated Text in {target_language}:")
            print(final_translation)

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    chatbot()
