# ui.py
import pandas as pd
import gradio as gr
from src.making_recommendation import recommend_books
from src.embedding import generate_embeddings
from src.data_ingestion import load_documents

# Create Gradio dashboard
def create_dashboard(db_books):
    categories = ["All"] + sorted(pd.read_csv("data/books_with_emotions.csv")["simple_categories"].unique())
    tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

    with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
        gr.Markdown("# Semantic book recommender")

        with gr.Row():
            user_query = gr.Textbox(label="Please enter a description of a book:", placeholder="e.g., A story about forgiveness")
            category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
            tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All")
            submit_button = gr.Button("Find recommendations")

        gr.Markdown("## Recommendations")
        output = gr.Gallery(label="Recommended books", columns=8, rows=2)

        # Correct the click function call to match the signature
        submit_button.click(fn=recommend_books, inputs=[user_query, category_dropdown, tone_dropdown, db_books], outputs=output)

    return dashboard

if __name__ == "__main__":
    # Load documents and generate embeddings
    raw_documents = load_documents("data/tagged_descriptions.txt")
    db_books = generate_embeddings(raw_documents)

    # Create and launch the Gradio dashboard
    dashboard = create_dashboard(db_books)
    dashboard.launch()

