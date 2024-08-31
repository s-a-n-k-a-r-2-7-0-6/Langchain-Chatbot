from flask import Flask, request, jsonify
import os
import requests
import time
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
import langchain_community
from langchain_core import prompt_template

# Load environment variables from .env file
load_dotenv()
token = os.getenv('HUGGINGFACE_API_TOKEN')
login(token, add_to_git_credential=True)

app = Flask(__name__)

def search_and_scrape_articles(query, max_results=5):
    serpapi_key = os.getenv('SERPAPI_API_KEY')
    if not serpapi_key:
        print("Error: SERPAPI_API_KEY not found in environment variables.")
        return []

    search_url = f"https://serpapi.com/search.json?q={query}&api_key={serpapi_key}"
    search_response = requests.get(search_url)
    if search_response.status_code != 200:
        print("Error: Failed to retrieve search results.")
        return []

    search_results = search_response.json()
    article_urls = [result.get('link') for result in search_results.get('organic_results', []) if result.get('link')]

    articles_content = []
    for url in article_urls[:max_results]:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            content = ""
            for heading in soup.find_all(['h1', 'h2', 'h3', 'h4']):
                content += heading.get_text() + "\n"
            for paragraph in soup.find_all('p'):
                content += paragraph.get_text() + "\n"

            if content:
                articles_content.append(content.strip())

            time.sleep(1)

        except requests.RequestException as e:
            print(f"Failed to scrape {url}: {e}")

    return articles_content

def concatenate_articles(articles_content):
    concatenated_content = "\n\n".join(articles_content)
    return concatenated_content

def chunk_text(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, truncation=True, device=0)
    return generator

def generate_answer(concatenated_content, query):
    generator = load_llm()
    prompt_template = """
    Based on the following information:

    {concatenated_content}

    Answer the following question:

    {query}
    """
    chunks = chunk_text(concatenated_content, chunk_size=1000)
    answers = []
    for chunk in chunks:
        prompt = prompt_template.format(concatenated_content=chunk, query=query)
        answer = generator(prompt, max_length=512)[0]["generated_text"]
        answers.append(answer)

    final_answer = " ".join(answers)
    return final_answer

@app.route('/query', methods=['POST'])
def query():
    """
    Handles the POST request to '/query'. Extracts the query from the request,
    processes it through the search, concatenate, and generate functions,
    and returns the generated answer.
    """
    # get the data/query from streamlit app
    data = request.get_json()
    query = data.get('query')
    print("Received query:", query)
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    # Step 1: Search and scrape articles based on the query
    print("Step 1: searching articles")
    articles_content = search_and_scrape_articles(query)
    if not articles_content:
        return jsonify({"error": "Failed to retrieve articles"}), 500
    
    # Step 2: Concatenate content from the scraped articles
    print("Step 2: concatenating content")
    concatenated_content = concatenate_articles(articles_content)
    
    # Step 3: Generate an answer using the LLM
    print("Step 3: generating answer")
    answer = generate_answer(concatenated_content, query)
    
    # return the jsonified text back to streamlit
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(host='localhost', port=8501)
