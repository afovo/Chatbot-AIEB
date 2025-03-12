import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import re
import tempfile
import time
import streamlit as st
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
import pdfplumber
import pandas as pd
import re
from langchain.schema import Document

OPENAI_API_KEY = ""
# Set Openai API key (replace with your key or use an env variable)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# Define persona and template (unchanged)
persona = '''
You are a helpful assistant that answers questions based on the provided documents.
Answer the question with detailed information from the documents. If the answer is not in the documents, 
say "I don't have enough information to answer this question." Cite specific parts of the documents when possible.
Consider the chat history for context when answering, but prioritize information from the documents.
When discussing financial data or numbers from tables, always mention the exact figures with their context.
'''

template = """
{persona}
        
Chat History:
<history>
{chat_history}
</history>

Documents:
<documents>
{context}
</documents>

Given the context information and not prior knowledge, answer the following question:
Question: {user_input}
"""

def detect_toc_page(document):
    """
    Detect if the current document is a table of contents page
    Returns True if the page contains both 'For the Fiscal Year Ended' and ('INDEX' or 'TABLE OF CONTENTS')
    """
    content = document.page_content.upper()
    has_fiscal_year = 'FOR THE FISCAL YEAR ENDED' in content
    has_index = 'INDEX' in content or 'TABLE OF CONTENTS' in content
    
    return has_fiscal_year and has_index

def extract_toc_structure(toc_pages):
    """
    Extract table of contents structure from detected TOC pages
    Returns a dictionary mapping item numbers to their page numbers
    """
    toc_structure = {}
    page_offset = 0
    
    # Combine all TOC pages
    combined_content = ""
    for page in toc_pages:
        combined_content += page.page_content + "\n"
    
    # Look for patterns like "Item X. Title .... PageNumber"
    item_pattern = r'Item\s+(\d+[A-Z]?)\.?\s+([^\d]+?)\s+(\d+)'
    
    # Look for other patterns that might indicate sections
    section_pattern = r'PART\s+([IVX]+)'
    
    # Find all matches
    items = re.findall(item_pattern, combined_content, re.MULTILINE)
    
    # Extract page offset if present
    # If there's a clearly defined page number in the TOC itself (like page 2)
    # we can use that to calculate the offset
    first_page_match = re.search(r'Page\s+(\d+)', combined_content, re.IGNORECASE)
    if first_page_match:
        first_listed_page = int(first_page_match.group(1))
        # Assuming the first listed page is the next after TOC
        page_offset = first_listed_page - len(toc_pages) - 1
    
    # Process each item
    for item_number, title, page_number in items:
        # Clean up the title
        title = title.strip()
        # Add to structure, adjusting page number if needed
        toc_structure[f"Item {item_number}"] = {
            'title': title,
            'raw_page': int(page_number),
            'adjusted_page': int(page_number) - page_offset if page_offset else int(page_number)
        }
    
    # Find section divisions
    sections = re.findall(section_pattern, combined_content)
    section_positions = {}
    
    for section in sections:
        match = re.search(rf'PART\s+{section}', combined_content)
        if match:
            pos = match.start()
            section_positions[f"PART {section}"] = pos
    
    # Assign sections to items
    current_section = None
    for item in sorted(toc_structure.keys(), key=lambda x: combined_content.find(x)):
        # Find which section this item belongs to
        item_pos = combined_content.find(item)
        for section, pos in sorted(section_positions.items(), key=lambda x: x[1]):
            if item_pos > pos:
                current_section = section
        
        if current_section:
            toc_structure[item]['section'] = current_section
    
    return toc_structure

def extract_headers_as_markdown(doc):
    """
    Extract headers and content from document and format as markdown
    """
    content = doc.page_content
    
    # Simple heuristic to identify potential headers
    lines = content.split('\n')
    markdown_content = ""
    
    for line in lines:
        line = line.strip()
        # Skip empty lines
        if not line:
            markdown_content += "\n"
            continue
            
        # Check for Item headers (10-K specific)
        if re.match(r'^Item\s+\d+[A-Z]?\.', line, re.IGNORECASE):
            markdown_content += f"# {line}\n\n"
        # Check for Part headers
        elif re.match(r'^PART\s+[IVX]+', line):
            markdown_content += f"# {line}\n\n"
        # Potential header detection heuristics
        elif len(line) < 100 and line.endswith(':'):
            markdown_content += f"## {line}\n\n"
        elif len(line) < 80 and line.isupper():
            markdown_content += f"## {line}\n\n"
        elif re.match(r'^\d+(\.\d+)*\s+', line):  # Numbered sections
            markdown_content += f"### {line}\n\n"
        else:
            markdown_content += f"{line}\n"
    
    return markdown_content

def is_multirow_header(table_data, max_header_rows=3):
    """Check if table has multi-row headers"""
    if len(table_data) < 2:
        return False
        
    # Check if first rows are potential headers
    # Heuristic: headers are shorter than data rows
    header_row_lengths = [len(''.join(row)) for row in table_data[:max_header_rows]]
    data_row_lengths = [len(''.join(row)) for row in table_data[max_header_rows:max_header_rows+3]]
    
    if not data_row_lengths:  # Not enough data rows to compare
        return False
        
    avg_header_len = sum(header_row_lengths) / len(header_row_lengths)
    avg_data_len = sum(data_row_lengths) / len(data_row_lengths)
    
    return avg_header_len < avg_data_len * 0.7  # Headers are typically shorter

def merge_header_rows(table_data, max_header_rows=2):
    """Merge multiple header rows into single header"""
    if len(table_data) < max_header_rows + 1:
        return table_data
        
    merged_header = []
    for col_idx in range(len(table_data[0])):
        header_text = ' '.join([
            table_data[row_idx][col_idx] 
            for row_idx in range(max_header_rows)
            if col_idx < len(table_data[row_idx]) and table_data[row_idx][col_idx].strip()
        ])
        merged_header.append(header_text.strip())
    
    return [merged_header] + table_data[max_header_rows:]

def extract_table_caption(page, table_data):
    """Try to extract caption for a table"""
    import re
    
    # Get table bounding box
    table_bbox = get_table_bbox(table_data, page)
    if not table_bbox:
        return None
        
    # Look for text above or below the table that might be a caption
    text_above = page.crop((0, max(0, table_bbox[1] - 100), page.width, table_bbox[1])).extract_text()
    text_below = page.crop((0, table_bbox[3], page.width, min(page.height, table_bbox[3] + 100))).extract_text()
    
    # Look for caption patterns
    caption_patterns = [
        r'Table\s+\d+[\.:]?\s*(.*?)(?:\n|$)',
        r'(?:Figure|Fig\.)\s+\d+[\.:]?\s*(.*?)(?:\n|$)',
        r'(?:^\s*|[\.\n]\s*)([^\.]+?)\s+\(.*?\d+.*?\)\s*(?:\n|$)'  # Financial metrics often have years in parentheses
    ]
    
    for text in [text_above, text_below]:
        if not text:
            continue
            
        for pattern in caption_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
                
    return None

def get_table_bbox(table_data, page):
    """Get approximate bounding box of table on page"""
    # This is a simplified implementation
    # In a real scenario, you'd use the actual table coordinates from pdfplumber
    
    # Look through all tables on the page and find the one matching our data
    for table in page.find_tables():
        if table.extract() == table_data:
            return table.bbox
    return None

def segment_documents_by_toc(documents, toc_structure):
    """
    Segment documents based on table of contents structure
    Returns a list of documents with enhanced metadata
    """
    segmented_docs = []
    
    # Create a map of page numbers to document indices
    page_to_doc_idx = {}
    for i, doc in enumerate(documents):
        page = doc.metadata.get("page", 0)
        page_to_doc_idx[page] = i
    
    # Sort TOC items by page number (to process in order)
    sorted_items = sorted(toc_structure.items(), key=lambda x: x[1]['adjusted_page'])
    
    # Process each TOC item
    for i, (item_key, item_data) in enumerate(sorted_items):
        start_page = item_data['adjusted_page']
        
        # Determine end page (either the next item's page or the end of the document)
        if i < len(sorted_items) - 1:
            end_page = sorted_items[i+1][1]['adjusted_page']
        else:
            end_page = max(page_to_doc_idx.keys()) + 1
        
        # Find documents that belong to this section
        for page_num in range(start_page, end_page):
            if page_num in page_to_doc_idx:
                doc_idx = page_to_doc_idx[page_num]
                doc = documents[doc_idx]
                
                # Create a new document with enhanced metadata
                new_doc = Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "item_number": item_key,
                        "item_title": item_data['title'],
                        "section": item_data.get('section', ''),
                        "toc_page_range": f"{start_page}-{end_page-1}"
                    }
                )
                segmented_docs.append(new_doc)
    
    # Add documents that weren't covered by TOC (including TOC pages themselves)
    processed_pages = set(page for doc in segmented_docs for page in [doc.metadata.get("page", 0)])
    for i, doc in enumerate(documents):
        page = doc.metadata.get("page", 0)
        if page not in processed_pages:
            segmented_docs.append(doc)
    
    return segmented_docs

def structure_aware_pdf_splitting(documents):
    """Split PDFs with awareness of document structure and TOC"""
    # Define headers
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    md_header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    # Detect TOC pages
    toc_pages = []
    for doc in documents:
        if detect_toc_page(doc):
            print(1)
            toc_pages.append(doc)
    
    # If TOC pages found, extract structure
    toc_structure = {}
    if toc_pages:
        try:
            toc_structure = extract_toc_structure(toc_pages)
            
            # Save TOC structure to file for reference
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f"toc_structure_{timestamp}.txt", "w", encoding="utf-8") as f:
                f.write("EXTRACTED TABLE OF CONTENTS STRUCTURE:\n\n")
                for item, data in toc_structure.items():
                    f.write(f"{item}: {data['title']}\n")
                    f.write(f"  Raw Page: {data['raw_page']}\n")
                    f.write(f"  Adjusted Page: {data['adjusted_page']}\n")
                    if 'section' in data:
                        f.write(f"  Section: {data['section']}\n")
                    f.write("\n")
            
            # Segment documents based on TOC structure
            documents = segment_documents_by_toc(documents, toc_structure)
        except Exception as e:
            print(f"Error processing TOC structure: {e}")
            # Continue with regular processing if TOC processing fails
    
    structured_docs = []
    for doc in documents:
        # Convert to markdown-like format
        md_content = extract_headers_as_markdown(doc)
        
        # Try to split by headers
        try:
            header_split_docs = md_header_splitter.split_text(md_content)
        except Exception:
            # Fallback if header splitting fails
            header_split_docs = [Document(page_content=md_content, metadata=doc.metadata)]
        
        # Further split large sections
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""],
            keep_separator=True
        )
        
        for header_doc in header_split_docs:
            chunks = text_splitter.split_text(header_doc.page_content)
            for chunk in chunks:
                metadata = {}
                if hasattr(header_doc, 'metadata'):
                    metadata.update(header_doc.metadata)
                
                # Preserve original document metadata
                for key, value in doc.metadata.items():
                    if key not in metadata:
                        metadata[key] = value
                
                structured_doc = Document(
                    page_content=chunk,
                    metadata=metadata
                )
                structured_docs.append(structured_doc)
                
    return structured_docs

def extract_tables_from_pdf(pdf_path):
    """Extract tables from PDF and convert to structured format"""
    tables = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract tables from the page
                page_tables = page.extract_tables()
                
                for table_num, table_data in enumerate(page_tables):
                    if table_data and any(any(cell for cell in row) for row in table_data):
                        # Clean table data
                        cleaned_table = clean_table_data(table_data)
                        
                        # Try to identify table caption
                        table_caption = extract_table_caption(page, table_data)
                        
                        # Create DataFrame with proper header handling
                        if cleaned_table and len(cleaned_table) > 1:
                            headers = cleaned_table[0]
                            # Ensure headers are unique
                            unique_headers = []
                            header_counts = {}
                            
                            for h in headers:
                                if h in header_counts:
                                    header_counts[h] += 1
                                    unique_headers.append(f"{h}_{header_counts[h]}")
                                else:
                                    header_counts[h] = 0
                                    unique_headers.append(h)
                                    
                            # Create DataFrame
                            df = pd.DataFrame(cleaned_table[1:], columns=unique_headers)
                            
                            # Create context document
                            table_context = create_table_context(df, table_caption, page_num, table_num)
                            tables.append(table_context)
    except Exception as e:
        print(f"Error extracting tables: {e}")
        
    return tables

def clean_table_data(table_data):
    """Clean extracted table data"""
    if not table_data:
        return []
        
    cleaned_table = []
    for row in table_data:
        if not row or all(cell is None or cell == "" for cell in row):
            continue
            
        # Replace None with empty string and strip whitespace
        cleaned_row = [(str(cell).strip() if cell is not None else "") for cell in row]
        cleaned_table.append(cleaned_row)
    
    # Handle multi-row headers if needed
    if len(cleaned_table) >= 3 and is_multirow_header(cleaned_table):
        cleaned_table = merge_header_rows(cleaned_table)
        
    return cleaned_table

def create_table_context(df, caption, page_num, table_num):
    """Convert table to textual representation for embedding"""
    # Create a descriptive text representation of the table
    table_desc = f"Table {table_num+1} on page {page_num+1}"
    if caption:
        table_desc += f": {caption}"
    
    # Get table dimensions
    num_rows, num_cols = df.shape
    table_desc += f"\nThis table has {num_rows} rows and {num_cols} columns.\n"
    
    # Add column names
    table_desc += f"Columns: {', '.join(df.columns.tolist())}\n\n"
    
    # Try to identify what the table is about based on patterns
    financial_indicators = identify_financial_indicators(df)
    if financial_indicators:
        table_desc += f"This table appears to contain financial data: {', '.join(financial_indicators)}\n"
    
    # Add summary of numeric columns
    numeric_summary = ""
    for col in df.columns:
        try:
            # Convert to numeric, coerce errors
            numeric_data = pd.to_numeric(df[col], errors='coerce')
            # Check if we have enough numeric values
            if numeric_data.notna().sum() > len(df) * 0.5:  # If more than half are numeric
                numeric_summary += f"Column '{col}' statistics: "
                numeric_summary += f"Min: {numeric_data.min():.2f}, Max: {numeric_data.max():.2f}, "
                numeric_summary += f"Mean: {numeric_data.mean():.2f}, Median: {numeric_data.median():.2f}\n"
        except:
            continue
            
    if numeric_summary:
        table_desc += "\nNumeric column statistics:\n" + numeric_summary + "\n"
    
    # Add full table content in a structured text format
    table_desc += "\nTable content:\n"
    
    # Convert DataFrame to string representation
    table_string = df.to_string(index=False)
    table_desc += table_string
    
    # Add row-by-row description for better semantic understanding
    table_desc += "\n\nRow-by-row description:\n"
    for i, row in df.iterrows():
        row_desc = " | ".join([f"{col}: {row[col]}" for col in df.columns])
        table_desc += f"Row {i+1}: {row_desc}\n"
    
    # Return as a Document object
    return Document(
        page_content=table_desc,
        metadata={
            "page": page_num + 1,
            "type": "table",
            "caption": caption if caption else f"Table {table_num+1}",
            "table_num": table_num + 1,
            "source": f"Table {table_num+1} on page {page_num+1}"
        }
    )

def identify_financial_indicators(df):
    """Try to identify what financial indicators this table contains"""
    # Common financial terms to look for
    financial_terms = {
        "revenue": ["revenue", "sales", "income", "turnover"],
        "profit": ["profit", "earnings", "ebit", "ebitda", "net income", "gross margin"],
        "assets": ["assets", "liabilities", "equity", "balance sheet", "capital"],
        "cash flow": ["cash flow", "operating cash", "investing cash", "financing cash"],
        "ratios": ["ratio", "percentage", "growth", "margin", "return on", "roe", "roa"]
    }
    
    found_indicators = []
    
    # Check column names
    all_text = ' '.join(df.columns).lower()
    
    # Also check first row and first column as they often contain labels
    if len(df) > 0:
        first_row = ' '.join(str(x).lower() for x in df.iloc[0])
        all_text += ' ' + first_row
        
    first_col = ' '.join(str(x).lower() for x in df.iloc[:, 0] if len(df.columns) > 0)
    all_text += ' ' + first_col
    
    # Look for financial indicators
    for category, terms in financial_terms.items():
        for term in terms:
            if term in all_text:
                found_indicators.append(category)
                break
    
    return list(set(found_indicators))  # Remove duplicates

def save_extracted_headers(documents, output_dir):
    """Save extracted headers to a text file"""
    headers_text = ""
    toc_headers_text = "DOCUMENT STRUCTURE FROM TABLE OF CONTENTS:\n\n"
    
    # Check if we have TOC segmented documents
    toc_docs = [doc for doc in documents if "item_number" in doc.metadata]
    has_toc = len(toc_docs) > 0
    
    # First add TOC-derived structure if available
    if has_toc:
        # Group by item number
        items = {}
        for doc in toc_docs:
            item_num = doc.metadata.get("item_number")
            if item_num not in items:
                items[item_num] = {
                    "title": doc.metadata.get("item_title", ""),
                    "section": doc.metadata.get("section", ""),
                    "page_range": doc.metadata.get("toc_page_range", "")
                }
        
        # Add to text output, grouped by section
        sections = {}
        for item_num, item_data in items.items():
            section = item_data.get("section", "Other")
            if section not in sections:
                sections[section] = []
            sections[section].append((item_num, item_data))
        
        # Output by section
        for section, section_items in sorted(sections.items()):
            toc_headers_text += f"{section}:\n"
            for item_num, item_data in sorted(section_items):
                toc_headers_text += f"  {item_num}: {item_data['title']}\n"
                toc_headers_text += f"    Pages: {item_data['page_range']}\n"
            toc_headers_text += "\n"
    
    # Now add detected headers from content
    headers_text += "HEADERS DETECTED IN CONTENT:\n\n"
    for doc in documents:
        if 'Header' in doc.metadata.get('type', '') or any(key in doc.metadata for key in ['Header 1', 'Header 2', 'Header 3']):
            level = doc.metadata.get('level', 'unknown')
            page = doc.metadata.get('page', 'unknown')
            
            # Add TOC information if available
            toc_info = ""
            if "item_number" in doc.metadata:
                toc_info = f" [{doc.metadata['item_number']}: {doc.metadata['item_title']}]"
                
            headers_text += f"[Level: {level}] [Page: {page}]{toc_info} {doc.page_content}\n\n"
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save TOC structure if available
    if has_toc:
        toc_file = os.path.join(output_dir, f"toc_structure_{timestamp}.txt")
        with open(toc_file, 'w', encoding='utf-8') as f:
            f.write(toc_headers_text)
    
    # Save content headers
    output_file = os.path.join(output_dir, f"extracted_headers_{timestamp}.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        if has_toc:
            f.write(toc_headers_text + "\n" + headers_text)
        else:
            f.write(headers_text)
    
    return output_file

def save_extracted_tables(table_documents, output_dir):
    """Save extracted tables to text files and create a summary text file"""
    table_summary = []
    saved_files = []
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_text = "TABLE SUMMARY\n\n"
    
    for i, doc in enumerate(table_documents):
        if doc.metadata.get('type') == 'table':
            # Extract table content from document
            table_content = doc.page_content
            
            # Create summary entry
            page = doc.metadata.get('page', 'unknown')
            caption = doc.metadata.get('caption', f"Table {i}")
            table_num = doc.metadata.get('table_num', i)
            
            summary_text += f"Table {i+1}:\n"
            summary_text += f"  Page: {page}\n"
            summary_text += f"  Caption: {caption}\n"
            summary_text += f"  Table Number: {table_num}\n\n"
            
            # Save table content to text file
            table_filename = f"table_{i}_{timestamp}.txt"
            table_path = os.path.join(output_dir, table_filename)
            
            with open(table_path, 'w', encoding='utf-8') as f:
                f.write(table_content)
            
            saved_files.append(table_path)
            table_summary.append(table_path)
    
    # Save summary to text file
    summary_file = os.path.join(output_dir, f"table_summary_{timestamp}.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    return summary_file, saved_files

# Main Streamlit app code
def main():  
    st.set_page_config(page_title="Enhanced RAG for Annual Reports")

    st.title("📄💬 Enhanced RAG for Annual Reports")

    # File uploader
    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

    if uploaded_files:
        documents = []
        table_documents = []
        
        # Process PDFs if vector store doesn't exist in session state
        if "vector_store" not in st.session_state:
            with st.spinner("Processing your PDFs with enhanced structure and table recognition..."):
                # Create a temporary directory to save the uploaded files
                with tempfile.TemporaryDirectory() as temp_dir:
                    for file in uploaded_files:
                        # Save the uploaded file to a temporary file
                        temp_file_path = os.path.join(temp_dir, file.name)
                        with open(temp_file_path, "wb") as f:
                            f.write(file.getbuffer())
                        
                        # Load the PDF for text content
                        loader = PyPDFLoader(temp_file_path)
                        documents.extend(loader.load())
                        
                        # Extract tables separately
                        table_documents.extend(extract_tables_from_pdf(temp_file_path))

                    # Split documents into chunks with structure awareness
                    structured_docs = structure_aware_pdf_splitting(documents)
                    
                    # Combine with table documents
                    all_docs = structured_docs + table_documents
                    
                    # Set output directory to current directory
                    output_dir = "./monica/temp"  # Current directory
                    
                    # Save extracted headers and tables
                    try:
                        # Identify headers from structured docs
                        header_docs = [doc for doc in structured_docs 
                                      if any(key in doc.metadata for key in ['Header 1', 'Header 2', 'Header 3'])]
                        
                        # Save headers to file
                        headers_file = save_extracted_headers(header_docs, output_dir)
                        st.session_state.headers_file = headers_file
                        
                        # Save tables to files
                        tables_summary_file, table_files = save_extracted_tables(table_documents, output_dir)
                        st.session_state.tables_summary_file = tables_summary_file
                        st.session_state.table_files = table_files
                        
                        # Store output directory in session state
                        st.session_state.output_dir = output_dir
                    except Exception as e:
                        st.warning(f"Error saving extracted data: {str(e)}")
                    
                    # Generate embeddings and store in FAISS
                    embeddings = OpenAIEmbeddings()
                    st.session_state.vector_store = FAISS.from_documents(all_docs, embeddings)

            # Display success message with information about saved files
            success_msg = "✅ PDFs processed with enhanced structure and table recognition!"
            
            # Add information about saved files if available
            if hasattr(st.session_state, 'headers_file') and os.path.exists(st.session_state.headers_file):
                success_msg += f"\n\n**提取的标题已保存到:** `{os.path.basename(st.session_state.headers_file)}`"
                
            if hasattr(st.session_state, 'tables_summary_file') and os.path.exists(st.session_state.tables_summary_file):
                success_msg += f"\n\n**表格摘要已保存到:** `{os.path.basename(st.session_state.tables_summary_file)}`"
                
            if hasattr(st.session_state, 'table_files') and st.session_state.table_files:
                success_msg += f"\n\n**已保存 {len(st.session_state.table_files)} 个表格文件**"
                
            # Add download buttons
            if hasattr(st.session_state, 'headers_file') or hasattr(st.session_state, 'table_files'):
                with st.expander("查看和下载提取的数据"):
                    st.markdown("**所有文件已保存在当前目录下**")
                    
                    # List of all extracted files
                    all_files = []
                    if hasattr(st.session_state, 'headers_file'):
                        all_files.append(('标题数据', st.session_state.headers_file))
                    if hasattr(st.session_state, 'tables_summary_file'):
                        all_files.append(('表格摘要', st.session_state.tables_summary_file))
                    if hasattr(st.session_state, 'table_files'):
                        for i, file in enumerate(st.session_state.table_files):
                            all_files.append((f'表格 {i+1}', file))
                    
                    # Create download buttons for each file
                    for file_name, file_path in all_files:
                        if os.path.exists(file_path):
                            with open(file_path, 'r', encoding='utf-8') as f:
                                file_content = f.read()
                            st.download_button(
                                label=f"下载 {file_name}",
                                data=file_content,
                                file_name=os.path.basename(file_path),
                                mime="text/plain"
                            )
                
            st.success(success_msg)
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # User input
        user_input = st.chat_input("Ask a question about the annual report...")
        
        if user_input:
            # Add user message to chat history and display it
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Configure retriever with optimized parameters
            retriever = st.session_state.vector_store.as_retriever(
                search_type="mmr",  # Maximum Marginal Relevance
                search_kwargs={
                    "k": 6,  # Retrieve more documents
                    "fetch_k": 30,  # Consider more candidates for diversity
                    "lambda_mult": 0.8,  # Balance between relevance and diversity
                    "filter": None  # Optional metadata filtering
                }
            )
            
            # Improved filter capability for TOC-segmented documents
            def filter_by_toc_item(query, retriever_obj):
                """Allow filtering by Item number (e.g., 'Item 7A')"""
                item_match = re.search(r'Item\s+(\d+[A-Z]?)', query, re.IGNORECASE)
                if item_match:
                    item_num = f"Item {item_match.group(1)}"
                    st.info(f"Detected reference to {item_num}. Focusing search on that section.")
                    # Update retriever with filter
                    retriever_obj.search_kwargs["filter"] = {"item_number": item_num}
                else:
                    # Reset filter if no item detected
                    retriever_obj.search_kwargs["filter"] = None
                
                return query
            
            # Apply TOC filter if applicable
            try:
                # Check if TOC structure is available
                sample_docs = retriever.get_relevant_documents(user_input[:10])
                if sample_docs and len(sample_docs) > 0 and "item_number" in sample_docs[0].metadata:
                    user_input = filter_by_toc_item(user_input, retriever)
            except Exception as e:
                st.warning(f"Error during metadata filtering: {e}")
                # Continue without filtering
            
            # Get chat history for context
            chat_history = ""
            if len(st.session_state.messages) > 1:
                for i, msg in enumerate(st.session_state.messages[:-1]):
                    role = "User" if msg["role"] == "user" else "Assistant"
                    chat_history += f"{role}: {msg['content']}\n\n"
            
            # Create the QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model_name="gpt-4o", temperature=0.5),
                retriever=retriever,
                chain_type="stuff",
                return_source_documents=True,
                verbose=True,
                chain_type_kwargs={
                    "verbose": True
                }
            )
            
            # Get response from the chatbot with spinner
            with st.spinner("Analyzing annual report data..."):
                response = qa_chain.invoke({
                    "query": template.format(
                        persona=persona,
                        user_input=user_input,
                        chat_history=chat_history,
                        context="{{context}}"  # This will be filled by RetrievalQA
                    ),
                })
                
                # For debugging/transparency
                if "source_documents" in response:
                    with st.expander("View Retrieved Information"):
                        for i, doc in enumerate(response["source_documents"]):
                            st.markdown(f"**Chunk {i+1}**")
                            if doc.metadata.get("type") == "table":
                                st.markdown(f"**Source:** Table from page {doc.metadata.get('page', 'unknown')}")
                            else:
                                st.markdown(f"**Source:** Text from page {doc.metadata.get('page', 'unknown')}")
                                
                            # Show snippet of content
                            content_preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                            st.markdown(f"**Content Preview:** {content_preview}")
                            st.markdown("---")
                
                response_text = response["result"]
            
            # Display assistant response with streaming effect
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for chunk in response_text.split():
                    full_response += chunk + " "
                    message_placeholder.markdown(full_response)
                    time.sleep(0.05)
                    
            # Store assistant response
            st.session_state.messages.append({"role": "assistant", "content": response_text})
    else:
        st.info("Please upload annual report PDFs to begin.")

if __name__ == "__main__":
    main()