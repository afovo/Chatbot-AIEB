# Tech Doc

## Model Choices

## System Prompts
1. Basic persona prompt
```python
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
```

2. Recursive QA for formatting table data

```python
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
```



## Embeddings

1. **Table of content as metadata**
   - Page number offset between displayed and index
2. **Extract page by page headers as markdown**
   - Identify in page sections using Regular Expressions (e.g. includes "part", less than 120 words and end with":")
   - Convert them into markdown layers
   - Use `MarkdownHeaderTextSplitter` to split
3. **Identify and mark table structure**
   - Regular Expressions maches >=3  (e.g. Multiple dollar signs with numbers, Row-like structures with consistent separators, Year or date headers in tables, Headers with "Total" or financial terms, Contains "Table" or "in millions" keywords, Numerical grid pattern - rows of numbers aligned in columns)
   - Mark "has_financial_table"=True

## Insights

Chunking is important, struggling in preserving the table information and the paper structure.

