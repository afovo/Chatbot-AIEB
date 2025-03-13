import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import tempfile
import time

OPENAI_API_KEY = ""

# Custom prompt template for better answers
CUSTOM_PROMPT_TEMPLATE = """
You are a helpful assistant that answers questions based on the provided documents.
Please answer the question based on the following context information.
If the answer is not in the documents, clearly state: "I don't have enough information to answer this question."
Cite specific parts of the documents when possible.

Context:
{context}

Question: {question}
Answer:
"""

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

st.set_page_config(page_title="Chat with Your PDFs (OpenAI)")

st.title("📄💬 Chat with Your PDFs (OpenAI)")

# File uploader
uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    documents = []
    
    # Process PDFs if vector store doesn't exist in session state
    if "vector_store" not in st.session_state:
        with st.spinner("Processing your PDFs..."):
            # Create a temporary directory to save the uploaded files
            with tempfile.TemporaryDirectory() as temp_dir:
                for file in uploaded_files:
                    # Save the uploaded file to a temporary file
                    temp_file_path = os.path.join(temp_dir, file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    # Load the PDF
                    loader = PyPDFLoader(temp_file_path)
                    documents.extend(loader.load())

                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)  # Increased overlap
                docs = text_splitter.split_documents(documents)

                # Generate embeddings and store in FAISS
                embeddings = OpenAIEmbeddings()
                st.session_state.vector_store = FAISS.from_documents(docs, embeddings)

        st.success("✅ PDFs uploaded and processed! You can now start chatting.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Create memory component for ConversationalRetrievalChain
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    user_input = st.chat_input("Ask a question about your PDFs...")
    
    if user_input:
        # Immediately add user message to chat history and display it
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display the user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Configure retriever with MMR search for more diverse results
        retriever = st.session_state.vector_store.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance
            search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.7}  # Adjust these parameters for better results
        )
        
        # Create custom prompt template
        PROMPT = PromptTemplate(
            template=CUSTOM_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        # Use ConversationalRetrievalChain instead of RetrievalQA for better conversation handling
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name="gpt-4o", temperature=0.5),
            retriever=retriever,
            memory=st.session_state.memory,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=True,
            verbose=True
        )
        
        # Get response from the chatbot with spinner
        with st.spinner("Thinking..."):
            response = qa_chain({"question": user_input})
            response_text = response["answer"]
            source_docs = response["source_documents"]
            
            # Display retrieved chunks in an expander
            with st.expander("View Retrieved Chunks (Context)"):
                for i, doc in enumerate(source_docs):
                    st.markdown(f"**Chunk {i+1}**")
                    st.markdown(f"**Content:** {doc.page_content}")
                    st.markdown(f"**Source:** Page {doc.metadata.get('page', 'unknown')}")
                    st.markdown("---")
        
        # Display assistant response with streaming simulation
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            # Simulate streaming with an existing string
            for chunk in response_text.split():
                full_response += chunk + " "
                message_placeholder.markdown(full_response)
                time.sleep(0.05)  # Small delay to simulate streaming
        
        # Store assistant response in session state
        st.session_state.messages.append({"role": "assistant", "content": response_text})
else:
    st.info("Please upload PDF files to begin.")