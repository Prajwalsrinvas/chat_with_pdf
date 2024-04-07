import platform

if platform.system() != "Darwin":
    __import__("pysqlite3")
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import os
import re
import sys
from typing import Callable, Dict, List, Tuple

from dotenv import load_dotenv
from langchain.docstore.document import Document


def extract_metadata_from_pdf(file_path: str) -> dict:
    """
    Extracts the PDF file metadata.

    :param file_path: The path to the PDF file.
    :return: A dictionary containing the title and creation_date.
    """


def extract_pages_from_pdf(file_path: str) -> List[Tuple[int, str]]:
    """
    Extracts the text from each page of the PDF.

    :param file_path: The path to the PDF file.
    :return: A list of tuples containing the page number and the extracted text.
    """


def parse_pdf(file_path: str) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
    """
    Extracts the title and text from each page of the PDF.

    :param file_path: The path to the PDF file.
    :return: A tuple containing the title and a list of tuples with page numbers and extracted text.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    metadata = extract_metadata_from_pdf(file_path)
    pages = extract_pages_from_pdf(file_path)

    return pages, metadata


def merge_hyphenated_words(text: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def fix_newlines(text: str) -> str:
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)


def remove_multiple_newlines(text: str) -> str:
    return re.sub(r"\n{2,}", "\n", text)


def clean_text(
    pages: List[Tuple[int, str]], cleaning_functions: List[Callable[[str], str]]
) -> List[Tuple[int, str]]:
    cleaned_pages = []
    for page_num, text in pages:
        for cleaning_function in cleaning_functions:
            text = cleaning_function(text)
        cleaned_pages.append((page_num, text))
    return cleaned_pages


def text_to_docs(
    text: List[Tuple[int, str]], metadata: Dict[str, str]
) -> List[Document]:
    """
    Converts a list of tuples of page numbers and extracted text to a list of Documents.
    """


def store_chunks(document_chunks: List[Document], collection_name: str, directory: str):
    """
    Creates a Chroma vector store and persists the documents to disk.

    :param document_chunks: A list of Documents to be stored.
    :param collection_name: Name of this collection.
    :param directory: Location on disk to persist the data.
    """


if __name__ == "__main__":
    load_dotenv()

    # Step 1: Parse the PDF
    file_path = "data/jan-2024.pdf"
    raw_pages, metadata = parse_pdf(file_path)

    # Step 2: Create text chunks
    cleaning_functions = [
        merge_hyphenated_words,
        fix_newlines,
        remove_multiple_newlines,
    ]
    cleaned_text_pdf = clean_text(raw_pages, cleaning_functions)
    document_chunks = text_to_docs(cleaned_text_pdf, metadata)

    # Step 3: Generate embeddings and store them in the DB
    store_chunks(document_chunks, "jan-2024-economic", "data/chroma")
