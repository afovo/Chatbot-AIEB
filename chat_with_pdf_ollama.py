import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import tempfile
import time


# make sure you have the following models pulled already with ollama.
# ollama pull mistral
# ollama pull nomic-embed-text

# Set Ollama model names
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"  # Embedding model for Ollama
OLLAMA_LLM_MODEL = "mistral" # "deepseek-r1:14b"  # LLM model for Ollama

llm = OllamaLLM(model=OLLAMA_LLM_MODEL, temperature=0.5)
embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)



st.set_page_config(page_title="Chat with Your PDFs (Ollama)")

st.title("📄💬 Chat with Your PDFs (Ollama)")

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
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                docs = text_splitter.split_documents(documents)

                # Generate embeddings and store in FAISS
                # ------------------------------------------------- #
                
                # use Ollama embeddings model
                ############## building your vector store ##############
                st.session_state.vector_store = FAISS.from_documents(docs, embeddings)
                # ------------------------------------------------- #


        st.success("✅ PDFs uploaded and processed! You can now start chatting.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

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
        
        # ------------------------------------------------- #
        # Create QA chain with return_source_documents=True to get retrieved chunks

        retriever = st.session_state.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.7}
        )
        
        # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

        qa_chain = RetrievalQA.from_chain_type(
            ## use Ollama LLM model
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,  # This will return the source documents
            verbose=True,  # Enable verbose mode to see the full prompt
            chain_type_kwargs={
                "verbose": True  # Enable verbose mode in the chain itself
                # "prompt": You can add your custom prompt here.
            }
        )
        # ------------------------------------------------- #
        
        # Get response from the chatbot with spinner
        with st.spinner("Thinking..."):
            response = qa_chain.invoke({"query": user_input})

            # print([i for i in response])

            # Display retrieved chunks in an expander
            with st.expander("View Retrieved Chunks"):
                for i, doc in enumerate(response["source_documents"]):
                    st.markdown(f"**Chunk {i+1}**")
                    st.markdown(f"**Content:** {doc.page_content}")
                    st.markdown(f"**Source:** Page {doc.metadata.get('page', 'unknown')}")
                    st.markdown("---")
            
            # Display the final prompt sent to the LLM
            # with st.expander("View Final Prompt to LLM"):
            #     # Since return_generated_question is not supported, we'll just show the user query
            #     st.markdown(response["query"])
            #     # st.markdown(user_input)
            #     # st.markdown("**Context:** (Combined with retrieved chunks)")

            st.markdown(response)
                    
            response_text = response["result"]
            
        
        # Display assistant response
        # with st.chat_message("assistant"):
        #     st.markdown(response_text)

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