import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
password = os.getenv("hugging_face_token")

# API setup
API_URL = "https://api-inference.huggingface.co/models/codellama/CodeLlama-34b-Instruct-hf"
headers = {"Authorization": f"Bearer {password}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def generate_python_code_prompt(df, question):
    START_CODE_TAG = "```"
    END_CODE_TAG = "```"
    num_rows, num_columns = df.shape
    df_head = df.head().to_string()
    prompt = f"""
You are provided with a pandas dataframe (df) with {num_rows} rows and {num_columns} columns.
This is the metadata of the dataframe:
{df_head}.

When asked about the data, your response should include the python code describing the
dataframe `df`.  Do not include sample data. Using the provided dataframe, df, return python code and prefix
the requested python code with {START_CODE_TAG} exactly and suffix the code with {END_CODE_TAG}
exactly to answer the following question:
{question}
"""
    return prompt

def extract_python_code(output):
    match = re.search(r'```python\n(.*?)(```|$)', output, re.DOTALL)
    if match:
        code = match.group(1).strip()
        cleaned_code = '\n'.join([line for line in code.split('\n') if 'read_csv' not in line])
        return cleaned_code
    else:
        raise ValueError("No valid Python code found in the output")

def execute_code(code, df, question, max_retries=5):
    error_message = None
    retries = 0
    
    while retries <= max_retries:
        try:
            modified_code = f"result = {code}"
            exec_locals = {'df': df}
            exec(modified_code, {}, exec_locals)
            result = exec_locals.get('result', None)
            return result, None  # No error, so return result and None for error_message
        except Exception as e:
            error_message = str(e)
            result = None
            df_head = df.head().to_string()
            new_formatted_prompt = f"With this pandas dataframe (df): {df_head}\nAfter asking this question\n'{question}' \nI ran this code '{code}' \nAnd received this error message \n'{error_message}'. \nPlease provide new correct Python code."
            output = query({
                "inputs": new_formatted_prompt,
                "max_length": 10000,
                "top_k": 100,
                "do_sample": True
            })
            output_str = output[0]['generated_text'] if isinstance(output, list) and output else str(output)
            code = extract_python_code(output_str)  # Update code for the next iteration
            retries += 1  # Increment the retry counter
            
    return None, f"Failed to fix the code after {max_retries} retries. Last error: {error_message}"

# Sample dataframe
df = pd.DataFrame({
    'Var1': [1, 2, 3, 4, 5, 6],
    'Var2': [4, 5, 6, 7, 8, 9],
    'Gender': ['M', 'F', 'M', 'M', 'F', "M"],
    'State': ['IN', "NC", 'IN', 'NC', 'IN', 'NC'],
    'Race': ['W', 'B', 'W', 'B', 'W', "B"]
})

# ... (other imports and functions remain the same)

def main():
    st.title("Ask a question about this data")
    st.write("This is a demo.  It can answer simple questions like what is the sum of column A by Gender.  It cannot do more than one 'groupby'")
    st.write("DataFrame Preview (just the first few rows):")
    st.write(df.head())
    
    question = st.text_input("Enter your question about the DataFrame:")
    
    if question:
        formatted_prompt = generate_python_code_prompt(df, question)
        output = query({
            "inputs": formatted_prompt,
            "max_length": 100000,
            "top_k": 100,
            "do_sample": True
        })
        
        # Extract the relevant string from the list
        output_str = output[0]['generated_text'] if isinstance(output, list) and output else str(output)
        
        try:
            extracted_code = extract_python_code(output_str)
            st.write("Generated Python Code:")
            st.code(extracted_code, language='python')
            
            result, error_message = execute_code(extracted_code, df, question)
            if error_message:
                st.write(f"Error: {error_message}")
            else:
                st.write("Result:")
                st.write(result)
        
        except ValueError as e:
            st.write(f"Error: {e}")

if __name__ == "__main__":
    main()
