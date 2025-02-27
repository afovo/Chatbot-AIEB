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
pip install "openai>=1.3.0"
pip install "pypdf>=3.15.1"
```

# 2. Run the app

```bash
streamlit run chat_with_pdf_openai.py
streamlit run chat_with_pdf_gemini.py
```


