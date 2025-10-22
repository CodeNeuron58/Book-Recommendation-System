import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

def load_books_data(csv_path: str):
    books = pd.read_csv(csv_path)
    books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
    books["large_thumbnail"] = np.where(
        books["large_thumbnail"].isna(),
        "images.jpeg",
        books["large_thumbnail"],
    )
    return books

def load_documents(txt_path: str):
    from langchain_community.document_loaders import TextLoader
    raw_documents = TextLoader(txt_path, encoding="utf-8").load()
    return raw_documents
