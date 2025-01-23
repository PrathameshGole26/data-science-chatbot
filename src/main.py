import os
import json
import requests
import pandas as pd
import streamlit as st
from io import StringIO

# Configuring Gemini API - API Key
working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))
GEMINI_API_KEY = config_data["GEMINI_API_KEY"]
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

# Configuring Streamlit page settings
st.set_page_config(
    page_title="Data Science ChatBot",
    page_icon="📊",
    layout="centered"
)

# Initialize chat session in Streamlit if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit page title
st.title("🤖 Data Science ChatBot")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input field for user's message
user_prompt = st.chat_input("Ask the Data Science Expert...")

if user_prompt:
    # Add user's message to chat and display it
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Predefined system context for the chatbot
    system_prompt = (
        "You are a highly advanced Data Science Expert with expertise in Python, Power BI, SQL, AI/ML, and "
        "libraries like Numpy, Pandas, TensorFlow, OpenCV, Scikit-learn, Tableau, and Excel. "
        "You can also analyze data, provide statistical summaries, and generate Python code "
        "for any task. If numerical data is provided, you must analyze it and compute key statistics. "
        "Always provide code snippets when required."
    )

    # Prepare the payload for the Gemini API
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": f"{system_prompt}\n\nUser Query: {user_prompt}"}
                ]
            }
        ]
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(GEMINI_API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        response_data = response.json()
        try:
            assistant_response = response_data["candidates"][0]["content"]["parts"][0]["text"]
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

            # Display Gemini's response
            with st.chat_message("assistant"):
                st.markdown(assistant_response)

            # If the user provides CSV-like data, analyze it
            if user_prompt.strip().startswith("data:"):
                # Extract CSV-like data from the user prompt
                csv_data = user_prompt.replace("data:", "").strip()
                try:
                    df = pd.read_csv(StringIO(csv_data))
                    
                    # Display dataset preview
                    st.subheader("Dataset Preview")
                    st.dataframe(df)

                    # Compute and display key statistics
                    st.subheader("Descriptive Statistics")
                    st.write(df.describe())

                    # Display Python code to analyze the data
                    code_snippet = f"""
import pandas as pd
from io import StringIO

# Load data
csv_data = \"\"\"{csv_data}\"\"\"
df = pd.read_csv(StringIO(csv_data))

# Display descriptive statistics
print(df.describe())
"""
                    st.subheader("Python Code")
                    st.code(code_snippet, language="python")
                except Exception as e:
                    st.error(f"Failed to process the provided data. Error: {e}")
        except (KeyError, IndexError) as e:
            st.error(f"Unexpected response format: {response_data}")
    else:
        error_message = f"Error {response.status_code}: {response.text}"
        st.error(error_message)
