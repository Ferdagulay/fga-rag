# FGA-RAG

A CV question answering RAG pipeline that uses:

* Langchain

* FAISS in-memory vector db

* DeepSeek-R1-Distill-Qwen-1.5B (via Ollama or Huggingface API)

# Running locally

## Create a virtual python env
```sh
python -m venv .venv
.\.venv\Scripts\activate
```

## Install the requirements
```sh
pip install -r requirements.txt
```

## Run Ollama

You can run ollama easily with Docker. Below script runs ollama and downloads deepseek-r1 & bge-m3 models (tested on a pc with 16gb ram).

```sh
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

docker exec -it ollama ollama run deepseek-r1
docker exec -it ollama ollama run bge-m3
```

## Run
```sh
python main.py
```