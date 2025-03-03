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

In the vscode terminal, you can try the following command to make sure the environment is working:

```bash

conda activate chatbot
which python # should return /Users/your_username/miniconda3/envs/chatbot/bin/python
```


# 2. Access the LLM

## 2.1 Download and Install Ollama

Ollama allows you to run large language models locally on your computer. To install Ollama:

1. Visit the official Ollama website at [https://ollama.com/download](https://ollama.com/download)
2. Download the appropriate installer for your operating system (Windows, macOS, or Linux)
3. Run the installer and follow the on-screen instructions
4. After installation, Ollama will run as a service in the background
5. Verify the installation by opening a terminal (you can open it in VSCode) and running `ollama --version`

Once installed, you can download and run various models using commands like `ollama pull mistral` or `ollama run mistral`.

Mistral here is a small and fast model, you can try other models like `llama3.1` or `llama3.1:8b`. For more information about the models, you can visit the [Ollama website](https://ollama.com/models).

You can also try the following command in the vscode terminal to make sure it's working:

```bash
ollama --version # should return a version number like 0.5.12
```

You can start to chat with the model by running the following command:

```bash
ollama run mistral # should return a prompt to enter a message # entry /bye to exit
```

You can also try whether the ollama can work with the `test_with_ollama.py` file.

If you check the python code, you will find that it will call the ollama model to answer the question.
```python
ollama = OllamaLLM(model="mistral")
response = ollama.invoke("What's the capital of France?")
print(response)
```

In the vscode terminal, you can try the following command to make sure the ollama is working:

```bash
python test_with_ollama.py # the expected output is: The capital city of France is Paris.
```



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


