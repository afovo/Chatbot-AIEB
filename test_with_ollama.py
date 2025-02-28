from langchain_ollama import OllamaLLM
# ollama = OllamaLLM(model="llama3")
ollama = OllamaLLM(model="mistral")
response = ollama.invoke("What's the capital of France?")
print(response)
