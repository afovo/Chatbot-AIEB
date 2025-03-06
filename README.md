# 1. Prepare your tools

Feel free to skip these steps if you have already installed the tools.

## 1.1 Install Visual Studio Code (VSCode)

1. Go to the [VSCode download page](https://code.visualstudio.com/download)
2. Download the appropriate version for your operating system (Windows, macOS, or Linux)
3. Follow the installation instructions for your platform
4. Launch VSCode after installation

## 1.2 Download and Open This Repository

1. Download this repository by clicking the green "Code" button on the GitHub page and selecting "Download ZIP"
2. Extract the ZIP file to a location on your computer
3. In VSCode, go to File > Open Folder and select the extracted folder

## 1.3 Opening the Terminal in VSCode

1. In VSCode, press `` Ctrl+` `` (Windows/Linux) or `` Cmd+` `` (macOS) to open the integrated terminal
2. Alternatively, go to View > Terminal from the menu bar

## 1.4 Install Miniconda

1. Go to the [Miniconda download page](https://www.anaconda.com/docs/getting-started/miniconda/install)
2. Download the appropriate installer for your operating system
3. Run the installer and follow the installation instructions
4. Verify the installation by opening a new terminal and typing `conda --version`



# 2. Install Environment 


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
pip install "langchain>=0.1.0"  # Core LangChain framework for building LLM applications
pip install "langchain-community>=0.0.10"  # Community integrations for LangChain, like RAG tools
pip install "langchain-google-genai>=0.0.5"  # Google Generative AI (Gemini) integration for LangChain
pip install "google-generativeai>=0.3.0"  # Google's official Python SDK for Gemini models
pip install "langchain-openai>=0.0.2"  # OpenAI integration for LangChain
pip install "langchain-ollama"  # Ollama integration for running local LLMs with LangChain
pip install "openai>=1.3.0"  # OpenAI's official Python SDK for GPT models
pip install "pypdf>=3.15.1"  # Library for reading and extracting text from PDF files
```

In the vscode terminal, use following command to confirm the environment is working:

```bash
conda activate chatbot
which python # should return /Users/your_username/miniconda3/envs/chatbot/bin/python
```


# 2. Run Chatbot with Local LLM

## 2.1 Download and Install Ollama

Ollama allows you to run large language models locally on your computer. To install Ollama:

1. Visit the official Ollama website at [https://ollama.com/download](https://ollama.com/download)
2. Download the appropriate installer for your operating system (Windows, macOS, or Linux)
3. Run the installer and follow the on-screen instructions
4. After installation, Ollama will run as a service in the background
5. Verify the installation by opening a terminal (you can open it in VSCode) and running `ollama --version`

Once installed, you can download and run LLM models using commands like `ollama pull mistral` or `ollama run mistral`.

Mistral here is a small and fast model, you can try other models like `llama3.1` or `llama3.1:8b`. For more information about the models, you can visit the [Ollama website](https://ollama.com/models).

Use the following command in the vscode terminal to make sure it's working:

```bash
ollama --version # should return a version number like 0.5.12
```


Pull the model you want to use:

```bash
ollama pull mistral
```

You can start to chat with the model by running the following command:

```bash
ollama run mistral # should return a prompt to enter a message # entry /bye to exit
```

Besides the LLM, we also need to install the embedding model. Here we use the `nomic-embed-text` model.

```bash
ollama pull nomic-embed-text
```

## 2.2 Test the ollama model

**use the ollama model in the python code**

You can also try whether the ollama can work with the `test_with_ollama.py` file.

If you check the python code, you will find that it will call the ollama model to answer the question.
```python
# in the python script.
ollama = OllamaLLM(model="mistral")
response = ollama.invoke("What's the capital of France?")
print(response)
```

In the vscode terminal, use following commands to confirm the ollama is working:

```bash
# in the terminal, run the following command
python test_with_ollama.py # the expected output is: The capital city of France is Paris.
```


## 2.3 Run the Chatbot App 

Now you have the ollama model and the embedding model. 


You can run the chatbot.


To run the chatbot with Ollama, you have three options:

**Basic Ollama Chatbot**
```bash
streamlit run chat_with_local_ollama.py
```

**Basic Ollama Chatbot with PDF**
```bash
streamlit run chat_with_pdf_ollama.py
```

**Basic Ollama Chatbot with PDF and Historical Context**
```bash
streamlit run chat_with_pdf_ollama_with_history.py
```






# 3. Run Chatbot with Remote LLM

## 3.1 Get Gemini API Key 

You can get the Gemini API key from the Google Cloud Console: https://aistudio.google.com/apikey.

Put the API key in the `GOOGLE_API_KEY` variable in 
* `chat_with_pdf_gemini.py` 
* `chat_with_gemini.py` file.
* `chat_with_pdf_gemini_with_history.py` file.

```python
# find and replace the GOOGLE_API_KEY in chat_with_pdf_gemini.py
GOOGLE_API_KEY = 'YOUR_GOOGLE_API_KEY'

# find and replace the GOOGLE_API_KEY in chat_with_gemini.py
GOOGLE_API_KEY = 'YOUR_GOOGLE_API_KEY'
```

## 3.2 Run the Chatbot App 

To run the chatbot with Ollama, you have three options:

**Basic Gemini Chatbot**
```bash
streamlit run chat_with_pdf_gemini.py
```

**Basic Gemini Chatbot with PDF**
```bash
streamlit run chat_with_pdf_gemini.py
```

**Basic Gemini Chatbot with PDF and Historical Context**
```bash
streamlit run chat_with_pdf_gemini_with_history.py
```


## 3.3 Other Options

Put the API key in the `OPENAI_API_KEY` variable in the `chat_with_pdf_openai.py` file.

```python
# find and replace the OPENAI_API_KEY in chat_with_pdf_openai.py
OPENAI_API_KEY = 'YOUR_OPENAI_API_KEY'
```

