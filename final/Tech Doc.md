# Technical Documentation - Team4

**Architecture Overview**

The system is designed to parse, analyze, and enable question-answering from financial documents (10-K reports) through a Streamlit interface. The architecture incorporates specialized document processing for financial tables, document structure awareness, and conversational context management.

**Tech Stack Components**

1. **Frontend**: Streamlit
2. **Document Processing**: PyPDFLoader, Custom splitters
3. **Embedding & Retrieval**: OpenAI Embeddings, FAISS vector store
4. **LLM Integration**: ChatOpenAI (GPT-4o)
5. **Chain Management**: LangChain's RetrievalQA, LLMChain

## Model Choices

### 1. Final Choice

- **Embedding Model**: OpenAI Embeddings
- **Primary LLM**: GPT-4o (temperature=0.5)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Retrieval Method**: Maximum Marginal Relevance (MMR)

### 2. Local Experiment

//Todo [wjf] [ght]

| Model          |                       |      |
| -------------- | --------------------- | ---- |
|                |                       |      |
|                |                       |      |
| deepseek-r1:8b | 跑lrz的问题跑了10分钟 |      |

///ppt challenge

## Document Processing and Embedding

### 1. Structure Detection & Metadata Enhancement

**Table of Contents Processing**

- Function: 

  ```python
  is_toc_page()
  ```

  - Identifies TOC pages through regex patterns
  - Detects "table of contents," "index," "PART [Roman numerals]," and item numbers

- Function: 

  ```
  extract_toc_structure()
  ```

  - Extracts hierarchical structure (Parts → Items)
  - Maps page numbers to document sections and handles page number offset between displayed and actual indices
  - Function `add_toc_metadata()`Assigns section metadata to pages based on TOC structure

**Header Extraction & Markdown Conversion**

- Function: 

  ```
  extract_headers_as_markdown()
  ```

  - Identifies in-page sections using regex
  - Recognizes headers like "Item X.X", "PART X", lines ending with ":", and all-caps sections
  - Converts document structure to markdown format
  - Returns both markdown content and header metadata dictionary

**Table Structure Identification**

- Function: 

  ```
  is_potential_table()
  ```

  - Uses multiple regex patterns to identify financial tables:
    - Multiple dollar signs with numbers
    - Row-like structures with consistent separators
    - Year/date headers in tables
    - Financial term concentration (total, revenue, etc.)
    - Keywords like "Table" or "in millions"
    - Numerical grid patterns
    - Securities and financial instrument listings
  - Tags identified tables with `is_financial_table=True`

### 2. Document Chunking & Metadata Preservation

- Function: `add_toc_metadata()`
  - Assigns section metadata to pages based on TOC structure
  - Tracks which section each page belongs to
- Function: `structure_splitting()`
  - Primary chunking method with structure awareness
  - Uses `MarkdownHeaderTextSplitter` for header-based chunking
  - Falls back to `RecursiveCharacterTextSplitter` if header splitting fails
  - Preserves both TOC and header metadata across chunks
  - Sets chunk size (500 chars) and overlap (100 chars)

### **3. Advanced Retrieval Configuration**

- Retrieves more candidates (fetch_k=50) and filters to top results (k=12)
- Lambda multiplier (0.7) balances relevance and diversity

## System Prompts

1. Separates financial table chunks from regular text chunks
2. Processes financial tables through specialized analysis chain

Function: 

```
process_financial_tables()
```

- Creates specialized prompts for table analysis
- Combines related table chunks with source information
- Uses `LLMChain` with detailed financial analysis prompt
- Returns structured table analysis as a supplementary context

### 1. Base System Prompt

```
You are a helpful assistant that answers questions based on the provided documents.
Answer the question with detailed information from the documents. If the answer is not in the documents, 
say "I don't have enough information to answer this question." Cite specific parts of the documents when possible.
Consider the chat history for context when answering, but prioritize information from the documents.
```

### 2. Financial Table Analysis Prompt

```
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
```

## Insights

### 1. Hallucination Experiment

- **Potential Causes of Hallucinations**
fact Detailed questions/calculations 
Logical consistency testing 
Ask for something that does not appear in the local file
Deliberately Induce model misunderstanding promblems - increase or decrease the scope of the problem 

- **Over Over-reliance on prompts**
"If you can't find it, just say you don't know" prompt is  also a built-in prompt in LangChain.  

### 2. Structure Preservation Challenges

Document maintains parallel structure tracking:
- TOC-based section metadata
- In-text header hierarchy
- Table identification flags

### 3. Financial Table Processing

- Tables require specialized processing due to their structured nature
- Standard text chunking can break table relationships
- Identified tables receive dedicated prompt engineering