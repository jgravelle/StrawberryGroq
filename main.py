import streamlit as st
import os
import requests
from pocketgroq import GroqProvider

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''
if 'available_models' not in st.session_state:
    st.session_state.available_models = []

def get_groq_provider():
    if not st.session_state.api_key:
        st.error("Please enter your Groq API key.")
        return None
    return GroqProvider(api_key=st.session_state.api_key)

def fetch_available_models():
    api_key = st.session_state.api_key
    url = "https://api.groq.com/openai/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        models_data = response.json()
        st.session_state.available_models = [model['id'] for model in models_data['data']]
    except requests.RequestException as e:
        st.error(f"Error fetching models: {str(e)}")

def generate_response(prompt: str, use_cot: bool, model: str) -> str:
    groq = get_groq_provider()
    if not groq:
        return "Error: No API key provided."
    
    if use_cot:
        cot_prompt = f"Solve the following problem step by step, showing your reasoning:\n\n{prompt}\n\nSolution:"
        return groq.generate(cot_prompt, max_tokens=1000, temperature=0, model=model)
    else:
        return groq.generate(prompt, temperature=0, model=model)

def main():
    st.title("GroqBerry Chat")
    
    # API Key input
    api_key = st.text_input("Enter your Groq API Key:", type="password")
    if api_key:
        st.session_state.api_key = api_key
        fetch_available_models()
    
    # Model selection
    if st.session_state.available_models:
        selected_model = st.selectbox("Select a model:", st.session_state.available_models)
    else:
        selected_model = "llama2-70b-4096"  # Default model
    
    # CoT toggle
    use_cot = st.checkbox("Use Chain of Thought")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            if use_cot:
                st.write("Thinking step-by-step...")
            
            response = generate_response(prompt, use_cot, selected_model)
            st.write(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
