import os
import warnings
from pypdf import PdfReader
from langchain.schema import Document

warnings.filterwarnings("ignore", category=UserWarning, module="pypdf")


def extract_pages(pdf_path: str):
    """Extracts raw text pages from a PDF as LangChain Documents."""
    reader = PdfReader(pdf_path)
    docs = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        metadata = {
            "source_file": os.path.basename(pdf_path),
            "page": i + 1,
        }
        docs.append(Document(page_content=text, metadata=metadata))

    return docs
