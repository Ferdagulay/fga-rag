from pathlib import Path

from document_processing.pdf_parser import extract_text_from_pdf
from model_factory.llm_chain import create_chain, create_llm
from vectorstore_factory.vector_utils import create_embeddings, create_vectorstore

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    pdf_file = DATA_DIR / "FerdaGulAydin_AIEngineer_cv.pdf"
    # HuggingFace API
    # deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B  -> HuggingFace API LLM model
    # BAAI/bge-small-en-v1.5 -> HuggingFace API embedding model

    # Ollama
    # deepseek-r1 -> Ollama LLM model
    # bge-m3-> Ollama embedding model

    llm = create_llm("deepseek-r1")
    embedding = create_embeddings("bge-m3")
    docs = extract_text_from_pdf(pdf_file)
    vectorstore = create_vectorstore(docs, embedding)
    qa_chain = create_chain(vectorstore, llm)

    response = qa_chain.invoke(
        {
            "query": "Summarize the projects she involved in  and give details about her contribution"
        }
    )

    print(response["result"])


if __name__ == "__main__":
    main()
