# 1. Install Environment 

You will create a new conda environment and install the packages.
The environment will be named `chatbot`.

```bash
# Create a new conda environment
conda create -n chatbot python=3.11

# Activate the environment
conda activate chatbot

# Install packages available in conda-forge
conda install -c conda-forge streamlit faiss-cpu pdf2image pytesseract pillow

# Install the remaining packages using pip
# Option 1: Install using requirements.txt
pip install -r requirements.txt

# Option 2: Install packages individually if you don't have requirements.txt
pip install "langchain>=0.1.0"
pip install "langchain-community>=0.0.10"
pip install "langchain-google-genai>=0.0.5"
pip install "google-generativeai>=0.3.0"
pip install "langchain-openai>=0.0.2"
pip install "langchain-ollama"
pip install "openai>=1.3.0"
pip install "pypdf>=3.15.1"
```


# 2. Access the LLM

## 2.1 Download the Ollama 

https://ollama.com/download



## 2.2 Get Gemini API Key 

You can get the Gemini API key from the Google Cloud Console: https://aistudio.google.com/apikey.

Put the API key in the `GOOGLE_API_KEY` variable in the `chat_with_pdf_gemini.py` and  `chat_with_gemini.py` file.

```python
# find and replace the GOOGLE_API_KEY in chat_with_pdf_gemini.py
GOOGLE_API_KEY = 'YOUR_GOOGLE_API_KEY'

# find and replace the GOOGLE_API_KEY in chat_with_gemini.py
GOOGLE_API_KEY = 'YOUR_GOOGLE_API_KEY'
```


## 2.3 Get OpenAI API Key 

Put the API key in the `OPENAI_API_KEY` variable in the `chat_with_pdf_openai.py` file.

```python
# find and replace the OPENAI_API_KEY in chat_with_pdf_openai.py
OPENAI_API_KEY = 'YOUR_OPENAI_API_KEY'
```




# 3. Run the app

```bash

python test_with_ollama.py

streamlit run chat_with_pdf_openai.py
streamlit run chat_with_pdf_gemini.py
streamlit run chat_with_local_ollama.py
```


