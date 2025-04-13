import fitz
from langchain.docstore.document import Document


def split_into_paragraphs(text):
    paragraphs = []
    current_paragraph = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            current_paragraph.append(line)
        else:
            if current_paragraph:
                paragraphs.append(" ".join(current_paragraph))
                current_paragraph = []
    if current_paragraph:
        paragraphs.append(" ".join(current_paragraph))
    return paragraphs


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        doc = fitz.open(stream=f.read(), filetype="pdf")
    docs = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        for para in split_into_paragraphs(text):
            docs.append(Document(page_content=para, metadata={"page": page_num + 1}))
    return docs
