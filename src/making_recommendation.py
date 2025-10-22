# recommendation.py
import pandas as pd
from src.embedding import retrieve_semantic_recommendations
from src.data_ingestion import load_books_data

# making_recommendation.py

def recommend_books(query, category, tone, db_books):
    """Generate book recommendations based on the given query, category, and tone."""
    books = load_books_data("data/books_with_emotions.csv")
    recommendations = retrieve_semantic_recommendations(db_books, query, books, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        # Handle missing description
        description = row.get("description", "")
        if pd.isna(description) or description == "":
            truncated_description = "No description available"
        else:
            truncated_desc_split = description.split()
            truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        # Handle authors formatting
        authors = row.get("authors", "Unknown Author")
        if pd.isna(authors):
            authors_str = "Unknown Author"
        else:
            authors_split = authors.split(";")
            if len(authors_split) == 2:
                authors_str = f"{authors_split[0]} and {authors_split[1]}"
            elif len(authors_split) > 2:
                authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
            else:
                authors_str = authors

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    
    return results
