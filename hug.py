import os
import requests
from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
import json
from dotenv import load_dotenv
from flask_cors import CORS
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load environment variables
load_dotenv()

# Flask app setup
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Load Hugging Face GPT-2 model and tokenizer
model_name = "gpt2"  # You can try other models like "gpt2-medium", "gpt2-large"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Predefined responses for specific keywords
KEYWORD_RESPONSES = {
    "hi": "Hello! How can I assist you today?",
    "hello": "Hi there! How can I help you?",
    "hey": "Hey! What can I do for you?",
    "address": "Our office address is 123 Wallingford St, Wallingford, USA.",
    "contact": "You can contact us via email at support@wallingford.com or call us at +123456789.",
    "email": "You can reach us at support@wallingford.com.",
    "phone": "Our contact number is +123456789.",
    "call": "Please feel free to give us a call at +123456789."
}

# Fetch selected pages from the API
def get_selected_pages():
    try:
        # Make a GET request to the API that provides the selected pages
        api_url = "https://wallingford.devstage24x7.com/wp-json/chatbox/v1/selected-pages"
        response = requests.get(api_url)
        
        if response.status_code == 200:
            return response.json()  # Return the JSON response with selected pages
        else:
            return {"error": f"Failed to fetch selected pages: {response.status_code}"}
    except Exception as e:
        return {"error": f"Error fetching selected pages: {str(e)}"}

# Function to fetch specific content from a URL
def fetch_website_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract headers (h1 to h6) and paragraphs (p)
        content = {
            "h1": [header.get_text(strip=True) for header in soup.find_all('h1')],
            "h2": [header.get_text(strip=True) for header in soup.find_all('h2')],
            "h3": [header.get_text(strip=True) for header in soup.find_all('h3')],
            "h4": [header.get_text(strip=True) for header in soup.find_all('h4')],
            "h5": [header.get_text(strip=True) for header in soup.find_all('h5')],
            "h6": [header.get_text(strip=True) for header in soup.find_all('h6')],
            "p": [para.get_text(strip=True) for para in soup.find_all('p')]
        }
        
        # Convert content to JSON format
        return json.dumps(content)
    except Exception as e:
        return json.dumps({"error": f"Error fetching content: {str(e)}"})

# Function to generate a refined prompt using JSON content
def generate_prompt(user_input, json_content):
    return (
        f"Here is some content from our website (structured in JSON format):\n{json_content}\n\n"
        f"User query: {user_input}\n\n"
        "Please respond as a knowledgeable support assistant for Wallingford Financial, based on the above content."
    )

# Function to interact with Hugging Face GPT-2 model
def ask_gpt2(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Flask route for chatbot
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "Message is required"}), 400

    # Check if the user input matches any keyword for predefined responses
    for keyword, response in KEYWORD_RESPONSES.items():
        if keyword.lower() in user_input.lower():  # Case-insensitive match
            return jsonify({"response": response})

    # Fetch the selected pages dynamically from the API if no keyword matched
    selected_pages = get_selected_pages()

    if "error" in selected_pages:
        return jsonify({"error": selected_pages["error"]}), 500

    # Fetch website content from selected pages
    content_from_pages = {}
    for page_name, page_url in selected_pages.items():
        json_content = fetch_website_content(page_url)
        
        if "error" in json.loads(json_content):
            return jsonify({"error": json.loads(json_content)["error"]}), 500

        # Store content from each page
        content_from_pages[page_name] = json.loads(json_content)
    
    # Convert content into a single JSON string (flattened if necessary)
    json_content = json.dumps(content_from_pages)

    # Create a refined prompt using user input and fetched JSON content
    prompt = generate_prompt(user_input, json_content)

    # Ask GPT-2 for a response
    response = ask_gpt2(prompt)
    return jsonify({"response": response})

@app.route('/feedback', methods=['POST'])
def feedback():
    user_feedback = request.json.get("feedback")
    user_response = request.json.get("response")

    if not user_feedback or not user_response:
        return jsonify({"error": "Feedback and response are required"}), 400

    if user_feedback == "thumbs_up":
        return jsonify({"response": "Thank you for your feedback! Glad you liked it!"})

    elif user_feedback == "thumbs_down":
        refined_response = refine_response(user_response)
        return jsonify({"response": "Thank you for your feedback. Here's a refined response:", "response": refined_response})

    else:
        return jsonify({"error": "Invalid feedback value. Please use 'thumbs_up' or 'thumbs_down'."}), 400

# Function to refine the response
def refine_response(original_response):
    try:
        prompt = f"Refine the following response to make it more clear and helpful: {original_response}"
        refined_response = ask_gpt2(prompt)
        return refined_response
    except Exception as e:
        return f"Error refining response: {str(e)}"

# Run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
