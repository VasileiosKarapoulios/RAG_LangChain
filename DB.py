from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai
import os
import shutil


# Set the environment variable OpenAI API key
os.environ["OPENAI_API_KEY"] = ""  # Key needs to be filled

# Read it
openai.api_key = os.getenv("OPENAI_API_KEY")

CHROMA_PATH = "chromaDB"
DATA_PATH = "data"


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = PyPDFLoader(f"{DATA_PATH}/vaska579.pdf")
    documents = loader.load_and_split()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the document
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
