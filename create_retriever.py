import dotenv 
import os
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

NEWS_CSV_PATH = "data/news_data.csv"
NEWS_CHROMA_PATH = "chroma_data"

dotenv.load_dotenv()
if not os.path.exists(NEWS_CHROMA_PATH):
    loader = CSVLoader(file_path=NEWS_CSV_PATH, source_column="article_summary")
    articles = loader.load()

    articles_vector_db = Chroma.from_documents(
        articles, OpenAIEmbeddings(), persist_directory=NEWS_CHROMA_PATH
    )
    print("Vectorstore created and persisted.")
else:
    print("Vectorstore already exists, loading from disk.")
    # Load the vectorstore from disk
    articles_vector_db = Chroma(
        embedding_function=OpenAIEmbeddings(), persist_directory=NEWS_CHROMA_PATH
    )  

    

    question = "I just got promoted"
    relevant_docs = articles_vector_db.similarity_search(question, k=1)

    print(f"Articles: {relevant_docs[0].page_content}")
# you may add text splitters for larger data