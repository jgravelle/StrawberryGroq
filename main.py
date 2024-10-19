import streamlit as st
import requests
from pocketgroq import GroqProvider
from pocketgroq.autonomous_agent import AutonomousAgent

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''
if 'available_models' not in st.session_state:
    st.session_state.available_models = []
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "llama2-70b-4096"  # Default model

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
        if st.session_state.selected_model not in st.session_state.available_models:
            st.session_state.selected_model = st.session_state.available_models[0]
    except requests.RequestException as e:
        st.error(f"Error fetching models: {str(e)}")

def generate_response(prompt: str, use_cot: bool, model: str) -> str:
    groq = get_groq_provider()
    if not groq:
        return "Error: No API key provided."
    
    # Include chat history in the prompt
    history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
    full_prompt = f"{history}\nUser: {prompt}"
    
    if use_cot:
        agent = AutonomousAgent(groq, max_sources=3, model=model)
        cot_prompt = f"Solve the following problem step by step, showing your reasoning:\n\n{full_prompt}\n\nSolution:"
        
        steps = []
        for step in agent.process_request(cot_prompt):
            if step['type'] == 'research':
                st.write(f"Research: {step['content']}")
            elif step['type'] == 'response':
                steps.append(step['content'])
        
        # Combine steps into a single response
        return "\n".join(steps)
    else:
        return groq.generate(full_prompt, temperature=0, model=model)

def on_model_change():
    st.session_state.selected_model = st.session_state.model_selectbox

def main():
    st.title("Strawberry Groq DEMO")
    st.write("This is a simple demo of the PocketGroq library's enhanced 'Chain of Thought' functionality with Autonomous Agent.")
    st.write("<a href='https://github.com/jgravelle/pocketgroq'>https://github.com/jgravelle/pocketgroq</a> |    <a href='https://www.youtube.com/watch?v=S5dY0DG-q-U'>https://www.youtube.com/watch?v=S5dY0DG-q-U</a>", unsafe_allow_html=True)

    # API Key input
    api_key = st.text_input("Enter your Groq API Key:", type="password")
    if api_key:
        st.session_state.api_key = api_key
        fetch_available_models()
    
    # Model selection
    if st.session_state.available_models:
        st.selectbox(
            "Select a model:", 
            st.session_state.available_models, 
            index=st.session_state.available_models.index(st.session_state.selected_model),
            key="model_selectbox",
            on_change=on_model_change
        )
    
    st.write(f"Current model: {st.session_state.selected_model}")
    
    # CoT toggle
    use_cot = st.checkbox("Use Chain of Thought with Autonomous Agent")
    
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
                st.write("Thinking step-by-step with autonomous research...")
            
            response = generate_response(prompt, use_cot, st.session_state.selected_model)
            st.write(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
