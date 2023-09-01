# ai-auto


# Ask-DataFrame: Your Interactive DataFrame Assistant
# Introduction
Ask-DataFrame is an interactive Streamlit web application that allows users to ask questions about a provided DataFrame. The application leverages Hugging Face's CodeLlama-34b-Instruct-hf model to generate Python code based on user queries, then executes this code on the DataFrame to provide the answers.

# Visit the live app here.

# Features
Quick DataFrame preview to understand the data before asking questions
Natural Language Processing to convert user questions into Python code
Dynamic code execution to display real-time answers to the questions
Error handling for invalid queries and code

# Requirements
Python 3.x
pandas
numpy
requests
os
re
streamlit
python-dotenv
