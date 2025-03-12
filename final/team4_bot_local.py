import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
import tempfile
import time
import re
import logging
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# No need for OpenAI API key anymore
# OPENAI_API_KEY = ""
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Set Ollama model names
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_LLM_MODEL = "deepseek-r1:8b-max"

persona = '''
You are a helpful assistant that answers questions based on the provided documents.
Answer the question with detailed information from the documents. If the answer is not in the documents, 
say "I don't have enough information to answer this question." Cite specific parts of the documents when possible.
Consider the chat history for context when answering, but prioritize information from the documents.
'''

template = """
{persona}
        
Chat History:
<history>
{chat_history}
</history>

Given the context information and not prior knowledge, answer the following question:
Question: {user_input}
"""

# Define financial table processing function
def process_financial_tables(table_chunks, query, llm):
    """
    Process financial table data with specialized handling
    
    Args:
        table_chunks: List of document chunks with financial table data
        query: The user's original query
        llm: The language model to use for processing
        
    Returns:
        Processed table information as a string and analysis result
    """
    if not table_chunks:
        return "", {}
    
    # Create specialized prompt for table processing
    table_prompt = """
    You are a financial data expert analyzing tabular data from financial documents. 
    The data below appears in raw text format extracted from a PDF financial document.
    
    First, analyze the structure of this tabular data:
    1. Identify headers, columns, rows, and what kind of financial table this is
    2. Reconstruct the table's structure to understand the relationships between items
    3. Determine the financial metrics, time periods, and any important trends or values
    
    Then, answer this specific question using ONLY the information in this table data:
    {question}
    
    Table data:
    {table_data}
    
    Provide a clear, concise answer with specific numbers and metrics from the table.
    Include any relevant calculations clearly explained.
    If you cannot answer the question based SOLELY on this table data, state that clearly.
    """
    
    # Combine all table chunks into a single context
    combined_table_data = "\n\n".join([
        f"--- TABLE FROM {chunk.metadata.get('source_file', 'Unknown')}, " +
        f"PAGE {chunk.metadata.get('page', 'Unknown')}, " +
        f"SECTION {chunk.metadata.get('section', 'Unknown Section')} ---\n" +
        chunk.page_content
        for chunk in table_chunks
    ])
    
    logging.info(f"Processing {len(table_chunks)} financial table chunks")
    
    # Create a more specific query for table analysis
    table_specific_query = f"Based on the financial tables provided, {query}"
    
    # Get response from LLM for table processing
    prompt_template = PromptTemplate(
        input_variables=["question", "table_data"],
        template=table_prompt
    )
    
    table_chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        verbose=True
    )
    
    try:
        table_response = table_chain.invoke({
            "question": table_specific_query,
            "table_data": combined_table_data
        })
        
        table_analysis = table_response.get('text', '')
        return f"""
        Financial Table Analysis:
        {table_analysis}
        
        """, table_response
    except Exception as e:
        logging.error(f"Error processing financial tables: {str(e)}")
        return f"Error processing financial tables: {str(e)}", {}


st.set_page_config(page_title="Chat with Your PDFs (Ollama)")

st.title("📄💬 Chat with Your PDFs (Ollama)")

# File uploader
uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

def is_toc_page(text):
    """Check if a page contains a Table of Contents or Index with parts in Roman numerals"""
    # Check for TOC indicators
    toc_indicators = ["table of contents", "index", "contents"]
    has_toc_indicator = any(indicator in text.lower() for indicator in toc_indicators)
    
    # Check for PART + Roman numerals
    part_pattern = r'PART\s*[IVX]+'
    has_parts = bool(re.search(part_pattern, text, re.IGNORECASE))
    item_pattern = r'Item\s*\d+'
    has_items = bool(re.search(item_pattern, text, re.IGNORECASE))
    return has_toc_indicator and has_parts and has_items

def extract_toc_structure(text):
    """Extract the table of contents structure from 10-K documents"""
    structure = {}
    
    # Split the text into lines for easier processing
    lines = text.split('\n')
    
    # Find all part sections and their line numbers
    part_indices = []
    current_part = None
    
    for i, line in enumerate(lines):
        part_match = re.search(r'PART\s+([IVX]+)', line, re.IGNORECASE)
        if part_match:
            if current_part:
                # Close the previous part
                part_indices[-1]['end'] = i
            
            current_part = part_match.group(1)
            part_indices.append({
                'part': current_part,
                'start': i,
                'end': len(lines)  # Default to end of text
            })
    
    # Process each part section
    for part_info in part_indices:
        part_num = part_info['part']
        part_start = part_info['start']
        part_end = part_info['end']
        
        part_text = '\n'.join(lines[part_start:part_end])
        
        # Extract items within this part
        item_pattern = r'Item\s+(\d+[A-Z]?)\.?\s+(.*?)(?=\s+(\d+)\s*$)'
        item_matches = re.finditer(item_pattern, part_text, re.MULTILINE)
        
        part_items = {}
        for item_match in item_matches:
            item_num = item_match.group(1)
            item_title = item_match.group(2).strip()
            page_num = int(item_match.group(3)) if item_match.group(3) else None
            
            part_items[item_num] = {"title": item_title, "page": page_num}
        
        # If no items found, try alternative pattern for items with page numbers at end of line
        if not part_items:
            # Look for items in multiple lines 
            for i in range(part_start, part_end):
                line = lines[i]
                item_match = re.search(r'Item\s+(\d+[A-Z]?)\.?\s+(.*)', line)
                if item_match:
                    item_num = item_match.group(1)
                    item_title = item_match.group(2).strip()
                    
                    # Look for page number in the same line
                    page_match = re.search(r'(\d+)\s*$', line)
                    if page_match:
                        page_num = int(page_match.group(1))
                    else:
                        # Check next line for page number
                        if i + 1 < len(lines):
                            next_line = lines[i + 1]
                            page_match = re.search(r'^\s*(\d+)\s*$', next_line)
                            if page_match:
                                page_num = int(page_match.group(1))
                            else:
                                page_num = None
                        else:
                            page_num = None
                    
                    part_items[item_num] = {"title": item_title, "page": page_num}
        
        structure[part_num] = part_items
    
    return structure

def is_potential_table(text):
    """
    Use regex patterns to check if text might contain a table structure
    
    Args:
        text: The text to check for table indicators
        
    Returns:
        Boolean indicating if text likely contains a table
    """
    # Pattern 1: Multiple dollar signs with numbers (financial tables)
    dollar_pattern = r'\$\s*[\d,]+(\.\d+)?'
    if len(re.findall(dollar_pattern, text)) > 3:
        logging.info("Table pattern detected: Financial data with dollar signs")
        return True
    
    # Pattern 2: Row-like structures with consistent separators
    row_pattern = r'(.*?\s{2,}.*?\s{2,}.*?){3,}'
    if re.search(row_pattern, text, re.MULTILINE):
        logging.info("Table pattern detected: Row-like structure with aligned columns")
        return True
    
    # Pattern 3: Year or date headers in tables
    year_pattern = r'(19|20)\d{2}\s*(\(.*?\))?'
    date_headers = re.findall(year_pattern, text)
    if len(date_headers) >= 2:
        logging.info("Table pattern detected: Multiple year/date references")
        return True
    
    # Pattern 4: Headers with "Total" or financial terms
    financial_terms = ['total', 'revenue', 'cost', 'gross', 'net', 'income', 'profit', 'loss', 'assets', 'liabilities']
    term_count = sum(1 for term in financial_terms if re.search(r'\b' + term + r'\b', text.lower()))
    if term_count >= 3:
        logging.info(f"Table pattern detected: Contains {term_count} financial terms")
        return True
    
    # Pattern 5: Contains "Table" or "in millions" keywords
    if re.search(r'\btable\b|\bin millions\b|\bin thousands\b|\b\(\$\s*in\b', text.lower()):
        logging.info("Table pattern detected: Contains explicit table indicators")
        return True
    
    # Pattern 6: Numerical grid pattern - rows of numbers aligned in columns
    num_columns_pattern = r'(\d+[.,]?\d*\s+){3,}'
    if re.search(num_columns_pattern, text):
        logging.info("Table pattern detected: Grid of numbers")
        return True
    
    # Pattern 7: Securities, assets, or financial instrument listings
    securities_pattern = r'\b(securities|assets|cash|equity|debt|government)\b'
    if re.search(securities_pattern, text.lower()) and re.search(r'\d+[.,]\d+', text):
        logging.info("Table pattern detected: Financial instruments with values")
        return True
    
    logging.info("No table patterns detected in text")
    return False

def add_toc_metadata(pages, toc_structure=None, page_offset=0):
    """Assign section metadata to pages based on TOC structure without chunking"""
    # Return original pages if no TOC structure is provided
    if not toc_structure:
        return pages
    
    # Flatten the TOC structure for easier lookup
    flat_toc = {}
    for part, items in toc_structure.items():
        for item_num, item_data in items.items():
            key = f"Part {part} - Item {item_num}"
            if item_data["page"] is not None:
                flat_toc[item_data["page"]] = {
                    "key": key,
                    "title": item_data["title"]
                }
    # Set current section to track which section each page belongs to
    current_section = "Introduction"
    current_section_metadata = {"section": current_section}
    
    # Iterate through pages and add section metadata
    for page in pages:
        pdf_page_num = page.metadata.get('page', 0)  # PDF pages are 0-indexed
        # Convert to displayed page number
        displayed_page_num = pdf_page_num + page_offset
        
        # Check if this page starts a new section
        if displayed_page_num in flat_toc:
            current_section = f"{flat_toc[displayed_page_num]['key']}: {flat_toc[displayed_page_num]['title']}"
            current_section_metadata = {
                "section": current_section,
                "part": flat_toc[displayed_page_num]["key"].split(" - ")[0],
                "item": flat_toc[displayed_page_num]["key"].split(" - ")[1],
                "title": flat_toc[displayed_page_num]["title"]
            }
        # Add the section metadata to the page
        for key, value in current_section_metadata.items():
            page.metadata[key] = value
    # Return the pages with added metadata
    return pages

def extract_headers_as_markdown(doc: Document) -> tuple[str, dict]:
    """
    Extract headers and content from document and format as markdown
    
    Args:
        doc: Document to extract headers from
        
    Returns:
        Tuple of (markdown-formatted content, header metadata)
    """
    content = doc.page_content
    
    # Simple heuristic to identify potential headers
    lines = content.split('\n')
    markdown_content = ""
    
    # Track the most recent headers at each level
    header_metadata = {
        "h1": None,
        "h2": None,
        "h3": None
    }
    
    # Track the current position for possible headers
    current_line_num = 0
    
    for line in lines:
        line = line.strip()
        current_line_num += 1
        if not line:
            markdown_content += "\n"
            continue
        
        # Check for Item headers (10-K specific)
        if re.match(r'^Item\s+\d+[A-Z]?\.', line, re.IGNORECASE):
            header_metadata["h1"] = line
            header_metadata["h2"] = None  # Reset lower level headers
            header_metadata["h3"] = None
            markdown_content += f"# {line}\n\n"
        
        # Check for Part headers
        elif re.match(r'^PART\s+[IVX]+', line):
            header_metadata["h1"] = line
            header_metadata["h2"] = None  # Reset lower level headers
            header_metadata["h3"] = None
            markdown_content += f"# {line}\n\n"
        
        # Potential subheader detection
        elif line.endswith(':'):
            header_metadata["h2"] = line
            header_metadata["h3"] = None  # Reset lower level headers
            markdown_content += f"## {line}\n\n"
        
        # All caps as potential subheaders
        elif line.isupper():
            header_metadata["h2"] = line
            header_metadata["h3"] = None  # Reset lower level headers
            markdown_content += f"## {line}\n\n"
        
        # Numbered sections
        elif re.match(r'^\d+(\.\d+)*\s+', line):
            header_metadata["h3"] = line
            markdown_content += f"### {line}\n\n"
        
        # Regular content
        else:
            markdown_content += f"{line}\n"
    
    return markdown_content, header_metadata

def structure_splitting(documents: List[Document]) -> List[Document]:
    """
    Split PDFs with awareness of document structure and markdown headers
    
    Args:
        documents: List of documents to split
        
    Returns:
        List of split documents
    """
    # Define headers
    headers_to_split_on = [
        ("#", "h1"),     # Map markdown # to h1 in metadata
        ("##", "h2"),    # Map markdown ## to h2 in metadata
        ("###", "h3"),   # Map markdown ### to h3 in metadata
    ]
    
    md_header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    structured_docs = []
    for doc in documents:
        # Preserve existing TOC metadata from the document
        toc_metadata = {k: v for k, v in doc.metadata.items() 
                       if k in ['section', 'part', 'item', 'title']}
        # Convert to markdown-like format and get header metadata
        md_content, original_headers = extract_headers_as_markdown(doc)
        
        # Try to split by headers
        try:
            header_split_docs = md_header_splitter.split_text(md_content)
            
            # Process header split docs
            for header_doc in header_split_docs:
                # Start with existing TOC metadata
                combined_metadata = toc_metadata.copy()
                # Add header path to metadata
                header_path = []
                for level in ["h1", "h2", "h3"]:
                    if level in header_doc.metadata and header_doc.metadata[level]:
                        header_path.append(header_doc.metadata[level])
                        # Also add individual header level to combined metadata
                        combined_metadata[f"header_{level}"] = header_doc.metadata[level]
                
                if header_path:
                    combined_metadata["header_path"] = " > ".join(header_path)
            
                # Further split large sections
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=100,
                    separators=["\n\n", "\n", ".", " ", ""],
                    keep_separator=True
                )
                
                chunks = text_splitter.split_text(header_doc.page_content)
                
                for i, chunk in enumerate(chunks):
                    # Create comprehensive metadata for this chunk
                    chunk_metadata = {
                        "chunk_index": i,
                        "split_method": "header_then_text"
                    }
                    
                    # Add combined TOC and header metadata
                    chunk_metadata.update(combined_metadata)
                    
                    # Check if this chunk contains a financial table
                    is_financial_table = is_potential_table(chunk)
                    if is_financial_table:
                        chunk_metadata["is_financial_table"] = True
                    
                    # Add original document metadata
                    for key, value in doc.metadata.items():
                        if key not in chunk_metadata:
                            chunk_metadata[key] = value
                    
                    # Add this chunk to results
                    structured_doc = Document(
                        page_content=chunk,
                        metadata=chunk_metadata
                    )
                    structured_docs.append(structured_doc)
                
        except Exception as e:
            # Fallback for documents that can't be split by headers
            logging.warning(f"Header splitting failed: {str(e)}. Using fallback method.")
            
            # Combine TOC metadata with original header metadata from extraction
            fallback_metadata = doc.metadata.copy()
            for key, value in original_headers.items():
                if value:  # Only add non-None headers
                    fallback_metadata[f"header_{key}"] = value
            
            # Perform regular text splitting
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                separators=["\n\n", "\n", ".", " ", ""],
                keep_separator=True
            )
            
            chunks = text_splitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                chunk_metadata = fallback_metadata.copy()
                chunk_metadata["chunk_index"] = i
                chunk_metadata["split_method"] = "fallback_text_only"
                
                # Check if this chunk contains a financial table
                is_financial_table = is_potential_table(chunk)
                if is_financial_table:
                    chunk_metadata["is_financial_table"] = True
                
                structured_doc = Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                )
                structured_docs.append(structured_doc)
                
    return structured_docs

if uploaded_files:
    documents = []
    
    # Process PDFs if vector store doesn't exist in session state
    if "vector_store" not in st.session_state:
        with st.spinner("Processing your PDFs..."):
            # Create a temporary directory to save the uploaded files
            with tempfile.TemporaryDirectory() as temp_dir:
                all_pages = []
                toc_structures = {}
                page_offsets = {}
                
                for file in uploaded_files:
                    # Save the uploaded file to a temporary file
                    temp_file_path = os.path.join(temp_dir, file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    # Load the PDF
                    loader = PyPDFLoader(temp_file_path)
                    pdf_pages = loader.load()
                    
                    # Find TOC pages and extract structure
                    toc_found = False
                    for i, page in enumerate(pdf_pages):
                        if is_toc_page(page.page_content):
                            toc_found = True
                            try:
                                logging.info(f"TOC found in {file.name} on page {i+1}")
                                toc_structure = extract_toc_structure(page.page_content if file.name!="MSFT 10-K.pdf" else page.page_content+pdf_pages[i+1].page_content)
                                toc_structures[file.name] = toc_structure
                                logging.info(f"Extracted TOC structure: {toc_structure}")
                                os.makedirs('./temp', exist_ok=True)
                                with open(f'./temp/{file.name}_TOC.txt', 'w', encoding='utf-8') as file1:
                                    json.dump(toc_structure, file1, ensure_ascii=False, indent=4)
                                break
                            except Exception as e:
                                logging.error(f"Error extracting TOC: {e}")
                    
                    page_offset = (2-i) if toc_found else 0
                    if file.name=="MSFT 10-K.pdf": # brute force :(
                        page_offset -= 1
                    page_offsets[file.name] = page_offset
                    logging.info(f"Page offset for {file.name}: {page_offset}")
                    
                    # Store all pages with file information
                    for page in pdf_pages:
                        page.metadata['source_file'] = file.name
                        all_pages.append(page)
                
                # Process each file with its TOC if available
                for file_name in {page.metadata['source_file'] for page in all_pages}:
                    file_pages = [p for p in all_pages if p.metadata['source_file'] == file_name]
                    toc_structure = toc_structures.get(file_name)
                    page_offset = page_offsets.get(file_name, 0)
                    file_docs = add_toc_metadata(file_pages, toc_structure, page_offset)
                    documents.extend(file_docs)
                # Split documents into chunks with structure awareness
                structured_docs = structure_splitting(documents)
                
                logging.info(f"Created {len(structured_docs)} document chunks from {len(uploaded_files)} files")
                # Generate embeddings and store in FAISS
                embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
                st.session_state.vector_store = FAISS.from_documents(structured_docs, embeddings)

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
        
        # Configure retriever with advanced parameters
        retriever = st.session_state.vector_store.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance
            search_kwargs={"k": 12, "fetch_k": 50, "lambda_mult": 0.7}
        )
        
        # Get chat history for context
        chat_history = ""
        if len(st.session_state.messages) > 1:  # If there are previous messages
            for i, msg in enumerate(st.session_state.messages[:-1]):  # Exclude the current user message
                role = "User" if msg["role"] == "user" else "Assistant"
                chat_history += f"{role}: {msg['content']}\n\n"
        
        # First retrieve documents and separate financial tables from regular text
        with st.spinner("Retrieving relevant information..."):
            retrieved_docs = retriever.get_relevant_documents(user_input)
            
            # Separate financial table chunks from regular chunks
            financial_table_chunks = []
            regular_chunks = []
            
            for doc in retrieved_docs:
                if doc.metadata.get("is_financial_table", False):
                    financial_table_chunks.append(doc)
                else:
                    regular_chunks.append(doc)
            
            logging.info(f"Retrieved {len(retrieved_docs)} docs: {len(financial_table_chunks)} financial tables and {len(regular_chunks)} regular chunks")
        
        # Initialize Ollama LLM
        llm = OllamaLLM(model=OLLAMA_LLM_MODEL, temperature=0.5)
        
        # Special handling for financial tables if present
        table_analysis = ""
        if financial_table_chunks:
            with st.spinner("Analyzing financial tables..."):
                table_analysis, table_data = process_financial_tables(financial_table_chunks, user_input, llm)
                logging.info(f"Financial table analysis completed")
        
        # Create enhanced prompt with table analysis if available
        enhanced_prompt = f"""
        {persona}
        
        Chat History:
        <history>
        {chat_history}
        </history>
        
        {table_analysis if financial_table_chunks else ""}
        
        Given the context information and not prior knowledge, answer the following question:
        Question: {user_input}
        """
        
        # Create a QA chain to handle the regular chunks
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",  # "stuff" chain type puts all retrieved documents into the prompt context
            return_source_documents=True,  # Return source documents for reference
            verbose=True,
            chain_type_kwargs={
                "verbose": True  # Enable verbose mode to see the full prompt
            }
        )
        
        # Get response with enhanced prompt that includes table analysis
        with st.spinner("Generating response..."):
            response = qa_chain.invoke({
                "query": enhanced_prompt,
            })
            
            # Display retrieved chunks in an expander if source documents are available
            if "source_documents" in response:
                with st.expander("View Retrieved Chunks (Context)"):
                    # First show financial table chunks if present
                    if financial_table_chunks:
                        st.markdown("### Financial Table Chunks")
                        for i, doc in enumerate(financial_table_chunks):
                            st.markdown(f"**Financial Table {i+1}**")
                            st.markdown(f"**Content:** {doc.page_content}")
                            st.markdown(f"**Source:** {doc.metadata.get('source_file', 'Unknown')}, Page {doc.metadata.get('page', 'Unknown')}")
                            if 'section' in doc.metadata:
                                st.markdown(f"**Section:** {doc.metadata.get('section', 'Unknown')}")
                            if 'header_path' in doc.metadata:
                                st.markdown(f"**Path:** {doc.metadata.get('header_path', 'Unknown')}")
                            st.markdown("---")
                    
                    # Then show regular chunks
                    st.markdown("### Regular Document Chunks")
                    for i, doc in enumerate(response["source_documents"]):
                        if not doc.metadata.get("is_financial_table", False):  # Skip table chunks here
                            st.markdown(f"**Chunk {i+1}**")
                            st.markdown(f"**Content:** {doc.page_content}")
                            st.markdown(f"**Source:** {doc.metadata.get('source_file', 'Unknown')}, Page {doc.metadata.get('page', 'Unknown')}")
                            if 'section' in doc.metadata:
                                st.markdown(f"**Section:** {doc.metadata.get('section', 'Unknown')}")
                            if 'header_path' in doc.metadata:
                                st.markdown(f"**Path:** {doc.metadata.get('header_path', 'Unknown')}")
                            st.markdown("---")
            
            response_text = response["result"]
            
        # Display assistant response
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