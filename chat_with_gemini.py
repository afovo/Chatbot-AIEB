import streamlit as st
import time
import os
from langchain_google_genai import ChatGoogleGenerativeAI


# Set Google API key (replace with your key or use an env variable)
GOOGLE_API_KEY = 'YOUR_GOOGLE_API_KEY' # "YOUR_GOOGLE_API_KEY"  # Replace with your actual Gemini API key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# Function to get response from Gemini using langchain
def get_response_from_gemini(prompt):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    response = llm.invoke(prompt)
    return response

# Streamlit app
st.title("💬 Chat with Google Gemini")

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
        response = get_response_from_gemini(user_input)
    
    # Display assistant response with simulated streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        # Get the content from the AIMessage object and split it
        assistant_response = response.content
        for chunk in assistant_response.split():
            full_response += chunk + " "
            message_placeholder.markdown(full_response)
            time.sleep(0.01)  # Small delay to simulate streaming
    
    # Store assistant response in session state
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
