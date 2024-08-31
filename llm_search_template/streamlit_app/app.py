import streamlit as st
import requests

st.title("LLM-based RAG Search")

# Input for user query
query = st.text_input("Enter your query:")

if st.button("Search"):
    # Make a POST request to the Flask API
    flask_url = "http://localhost:8501/query"  # Replace with your Flask app URL
    print("accessing", flask_url, "with query", query)

    if query:
        try:
            # Prepare the payload to send to the Flask app
            payload = {"query": query}
            
            # Call the Flask app (POST request)
            response = requests.post(flask_url, json=payload)

            # Check the response status and display the result
            if response.status_code == 200:
                # Display the generated answer
                answer = response.json().get('answer', "No answer received.")
                st.write("Answer:", answer)
            else:
                st.error(f"Error: {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a query to search.")
