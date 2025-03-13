# Technical Documentation - Team4

## Team Members:

Huating Guan,  Yin Wei,  Jingyi Yu,  Monica Wu, Ruizhe Lyu

## Model link:

GitHub Link: https://github.com/afovo/Chatbot-AIEB/blob/main/final/team4_bot.py

## 1. **Architecture Overview**

The system is designed to parse, analyze, and enable question-answering from financial documents (10-K reports) through a Streamlit interface. The architecture incorporates specialized document processing for financial tables, document structure awareness, and conversational context management.

**Tech Stack Components**

1. **Frontend**: Streamlit
2. **Document Processing**: PyPDFLoader, Custom splitters
3. **Embedding & Retrieval**: OpenAI Embeddings, FAISS vector store
4. **LLM Integration**: ChatOpenAI (GPT-4o)
5. **Chain Management**: LangChain's RetrievalQA, LLMChain

## 2. Model Choices

### 2.1. Sample Question 1

Q: How much CASH does Amazon have at the end of 2024?

DeepSeek's Question: Wrong!

![image-20250312235947099](https://charisplace2.oss-cn-shanghai.aliyuncs.com/typora2/202503130958132.png)

OpenAI's answer: Correct!

![image-20250313095933442](https://charisplace2.oss-cn-shanghai.aliyuncs.com/typora2/202503130959618.png)

The correct answer: $78779

It can be found in Amazon pdf as follows:

![image-20250313000836141](https://charisplace2.oss-cn-shanghai.aliyuncs.com/typora2/202503131000335.png)

**Analysis for the reason of wrong answer:**

Deekseep is distributed by some related information.On page 55 of Amazon pdf, it mentions"During 2024, we completed acquisition activity for aggregate cash consideration of $780 million, net of cash acquired." It is the acquisition amount, but DeepSeek regard it as  cash.

### 2.2. Sample Question2

Compared to 2023, does Amazon's liquidity decrease or increase?

DeepSeek: Wrong!

![image-20250313101955663](https://charisplace2.oss-cn-shanghai.aliyuncs.com/typora2/202503131019845.png)

OpenAI: Correct but not good enough

![image-20250313102036576](https://charisplace2.oss-cn-shanghai.aliyuncs.com/typora2/202503131020648.png)

The correct answer: It increases.

The cite from Amazon pdf:

![image-20250313102123590](https://charisplace2.oss-cn-shanghai.aliyuncs.com/typora2/202503131021786.png)

Evaluation for the answer：

 Deepseek fail to find the right answer, since the question need some reasoning. OpenAI also don‘t find the right place, but it’s clever and do some correct reasoning like campare cash, marketable securities, which is reasonable.

#### Summary：

From the sample questions above, we can see that both models have some limitations in answering the questions. In short, OpenAI handles implicit information and calculations better, while DeepSeek struggles with incomplete retrievals.

### 2.3. Final Choice

- **Embedding Model**: OpenAI Embeddings
- **Primary LLM**: GPT-4o (temperature=0.5)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Retrieval Method**: Maximum Marginal Relevance (MMR)


## 3. Document Processing and Embedding

### 3.1. Structure Detection & Metadata Enhancement

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

### 3.2. Document Chunking & Metadata Preservation

- Function: `add_toc_metadata()`
  - Assigns section metadata to pages based on TOC structure
  - Tracks which section each page belongs to
- Function: `structure_splitting()`
  - Primary chunking method with structure awareness
  - Uses `MarkdownHeaderTextSplitter` for header-based chunking
  - Falls back to `RecursiveCharacterTextSplitter` if header splitting fails
  - Preserves both TOC and header metadata across chunks
  - Sets chunk size (500 chars) and overlap (100 chars)

### **3.3. Advanced Retrieval Configuration**

- Retrieves more candidates (fetch_k=50) and filters to top results (k=12)
- Lambda multiplier (0.7) balances relevance and diversity

## 4. System Prompts

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

### 4.1. Base System Prompt

```
You are a helpful assistant that answers questions based on the provided documents.
Answer the question with detailed information from the documents. If the answer is not in the documents, 
say "I don't have enough information to answer this question." Cite specific parts of the documents when possible.
Consider the chat history for context when answering, but prioritize information from the documents.
```

### 4.2. Financial Table Analysis Prompt

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

## 5. Insights

### 5.1. Hallucination Experiment

- **Potential Causes of Hallucinations**
fact Detailed questions/calculations 
Logical consistency testing 
Ask for something that does not appear in the local file
Deliberately Induce model misunderstanding promblems - increase or decrease the scope of the problem 

- **Over Over-reliance on prompts**
"If you can't find it, just say you don't know" prompt is  also a built-in prompt in LangChain. 

### 5.2. Structure Preservation Challenges

#### 1) Fixed Chunking leads to **semantic fragmentation** 

Chunking into 500 characters results in semantic fragmentation and the loss of information structure.

Using fixed-length chunks instead of splitting by headers can lead to the following drawbacks:

- **Context Breaks**: Fixed-length splitting may cut off paragraphs or sentences mid-way, causing semantic discontinuity and hindering the model’s overall comprehension.
- **Loss of Information Structure**: The original chapter hierarchy and header details are lost, removing natural logic and structure. This makes it harder to locate key information during retrieval or answer generation.
- **Scattered Key Information**: Important content might end up split across multiple chunks, resulting in fragmented retrieval and difficulty piecing together a complete answer.
- **Increased Noise**: Arbitrary splitting can mix irrelevant or peripheral details with core information, introducing extra noise and reducing answer accuracy.
- **Underutilized Metadata**: Titles and headers often provide valuable context and cues. Ignoring them weakens the system’s grasp of the document’s structure and impairs Q&A performance

#### 2) Context limitations & Input Truncation

​	When using Large Language Models (LLMs), each model has its own “context window size,” which is the maximum amount of input (usually measured in tokens) it can handle in a single interaction. If the input text exceeds this limit, the model will truncate or discard any excess, causing some content to be ignored.

- **GPT-4o**: Has a context limit of about 2K tokens (roughly a few thousand characters). Any input beyond this limit is truncated.
- **Gemini-Flash**: Has a context limit of up to 128K tokens, allowing it to process far more text at once compared to GPT-4o.

When we split a document into many small chunks or include a large amount of contextual information in the prompt, the total token count can quickly approach or exceed the model’s context limit. For models with a smaller context window (e.g., 2K tokens), this often leads to:

- **Input Truncation**: Some chunks or prompts get cut off, leaving the model unable to see all the information.
- **Missing Context**: Truncated information may contain crucial details, resulting in incomplete or inaccurate answers.

### 5.3. Financial Table Processing

- Tables require specialized processing due to their structured nature
- Standard text chunking can break table relationships
- Identified tables receive dedicated prompt engineering

## 6.  Challenging Question:

Extract the total 'cash and cash equivalents' of the three companies as of the end of 2024 from the balance charts and notes to the financial statements, respectively, and explain any differences that may exist.
