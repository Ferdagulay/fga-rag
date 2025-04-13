from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
from langchain_ollama import OllamaLLM


def create_llm(name):
    if name == "deepseek-r1":
        return OllamaLLM(model=name, temperature=0)
    elif name == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B":
        llm = HuggingFaceEndpoint(
            endpoint_url=f"https://api-inference.huggingface.co/models/{name}",
            huggingfacehub_api_token="write_your_token",
            temperature=0.01,
        )
        llm.timeout = 300
        return llm
    raise ValueError(f"LLM model {name} not supported.")


def create_chain(vectorstore, llm):
    return RetrievalQA.from_chain_type(
        # use k= 20 for OLLAMA
        # use k=5 for Hugginface
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    )
