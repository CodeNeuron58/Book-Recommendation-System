# embedding_search.py
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from src.data_ingestion import load_documents

def generate_embeddings(raw_documents, persist_directory="embeddings/books_embeddings_db"):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(persist_directory):
        print("Loading saved embeddings...")
        db_books = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        print("Generating new embeddings and saving them...")
        db_books = Chroma.from_documents(
            documents, embeddings, persist_directory=persist_directory
        )
    return db_books

def retrieve_semantic_recommendations(db_books, query, books, category=None, tone=None, initial_top_k=50, final_top_k=16):
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs
