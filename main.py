import streamlit as st
import os
from pocketgroq import GroqProvider

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''

def get_groq_provider():
    if not st.session_state.api_key:
        st.error("Please enter your Groq API key.")
        return None
    return GroqProvider(api_key=st.session_state.api_key)

def generate_response(prompt: str, use_cot: bool) -> str:
    groq = get_groq_provider()
    if not groq:
        return "Error: No API key provided."
    
    if use_cot:
        cot_prompt = f"Solve the following problem step by step, showing your reasoning:\n\n{prompt}\n\nSolution:"
        return groq.generate(cot_prompt, max_tokens=4096, temperature=0)
    else:
        return groq.generate(prompt, temperature=0)

def main():
    st.title("Groqberry Chat DEMO")
    st.write("This is a simple chain of thought demonstration modeled on OpenAI's o1 LLM.")
    st.info("Written by Strawberry, powered by [PocketGroq™](https://github.com/jgravelle/pocketgroq)")
    
    # API Key input
    api_key = st.text_input("Enter your Groq API Key:", type="password")
    if api_key:
        st.session_state.api_key = api_key
    
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
            
            response = generate_response(prompt, use_cot)
            st.write(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()