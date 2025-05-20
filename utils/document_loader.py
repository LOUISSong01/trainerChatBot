import os
from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.docstore.document import Document

def load_documents(directory: str) -> List[Document]:
    """
    Load documents from a directory recursively
    
    Args:
        directory (str): Path to the directory containing documents
        
    Returns:
        List[Document]: List of loaded documents
    """
    loader = DirectoryLoader(
        directory,
        glob="**/*.md",  # Load markdown files
        loader_cls=TextLoader,
        show_progress=True
    )
    
    documents = loader.load()
    return documents

def process_documents(documents: List[Document]) -> List[Document]:
    """
    Process documents to clean and prepare them for embedding
    
    Args:
        documents (List[Document]): List of documents to process
        
    Returns:
        List[Document]: Processed documents
    """
    processed_docs = []
    for doc in documents:
        # Remove empty lines
        content = "\n".join([line for line in doc.page_content.split("\n") if line.strip()])
        
        # Create new document with processed content
        processed_doc = Document(
            page_content=content,
            metadata=doc.metadata
        )
        processed_docs.append(processed_doc)
    
    return processed_docs 