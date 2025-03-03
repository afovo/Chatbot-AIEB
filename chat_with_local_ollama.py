import streamlit as st
import time
from langchain_ollama import OllamaLLM

# Function to get response from local Ollama Mistral LLM using langchain_ollama
model = 'deepseek-r1:14b'

def get_response_from_ollama(prompt):
    ollama = OllamaLLM(model=model)
    response = ollama.invoke(prompt)
    return response

# Streamlit app
st.title(f"💬 Chat with Local Ollama {model} LLM")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input with chat_input
user_input = st.chat_input("Ask something...")

if user_input:
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display the user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get response with spinner
    with st.spinner("Thinking..."):
        response = get_response_from_ollama(user_input)
    
    # Display assistant response with simulated streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        # Simulate streaming with an existing string
        for chunk in response.split():
            full_response += chunk + " "
            message_placeholder.markdown(full_response)
            time.sleep(0.01)  # Small delay to simulate streaming
    
    # Store assistant response in session state
    st.session_state.messages.append({"role": "assistant", "content": response})
