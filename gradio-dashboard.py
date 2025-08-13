import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

raw_documents = TextLoader("tagged_description.txt").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db_books = Chroma.from_documents(documents, gemini_embeddings)


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
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


def get_default_books():
    """Get initial books to display on startup"""
    default_books = books.head(8)  # Show first 8 books

    gallery_results = []
    description_html = "<div class='book-descriptions'>"

    for i, (_, row) in enumerate(default_books.iterrows()):
        # Gallery items (just images with minimal captions)
        gallery_results.append((row["large_thumbnail"], f"{row['title']}"))

        # Detailed descriptions for the right panel
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        # Get emotional scores for display
        emotions = {
            'Joy': row.get('joy', 0),
            'Surprise': row.get('surprise', 0),
            'Anger': row.get('anger', 0),
            'Fear': row.get('fear', 0),
            'Sadness': row.get('sadness', 0)
        }
        top_emotion = max(emotions, key=emotions.get)

        description_html += f"""
        <div class='book-card' id='book-{i}'>
            <div class='book-header'>
                <h3 class='book-title'>{row['title']}</h3>
                <p class='book-author'>by {authors_str}</p>
                <span class='emotion-badge'>{top_emotion}</span>
            </div>
            <div class='book-description'>
                {row['description']}
            </div>
            <div class='book-meta'>
                <span class='category'>{row.get('simple_categories', 'Unknown')}</span>
                <span class='rating'>★ {row.get('average_rating', 'N/A')}</span>
            </div>
        </div>
        """

    description_html += "</div>"
    return gallery_results, description_html


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)

    if recommendations.empty:
        return [], "<div class='book-descriptions'><p style='color: #666; text-align: center; margin-top: 40px;'>No books found matching your criteria.</p></div>"

    gallery_results = []
    description_html = "<div class='book-descriptions'>"

    for i, (_, row) in enumerate(recommendations.iterrows()):
        # Gallery items (just images with minimal captions)
        gallery_results.append((row["large_thumbnail"], f"{row['title']}"))

        # Detailed descriptions for the right panel
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        # Get emotional scores for display
        emotions = {
            'Joy': row.get('joy', 0),
            'Surprise': row.get('surprise', 0),
            'Anger': row.get('anger', 0),
            'Fear': row.get('fear', 0),
            'Sadness': row.get('sadness', 0)
        }
        top_emotion = max(emotions, key=emotions.get)

        description_html += f"""
        <div class='book-card' id='book-{i}'>
            <div class='book-header'>
                <h3 class='book-title'>{row['title']}</h3>
                <p class='book-author'>by {authors_str}</p>
                <span class='emotion-badge'>{top_emotion}</span>
            </div>
            <div class='book-description'>
                {row['description']}
            </div>
            <div class='book-meta'>
                <span class='category'>{row.get('simple_categories', 'Unknown')}</span>
                <span class='rating'>★ {row.get('average_rating', 'N/A')}</span>
            </div>
        </div>
        """

    description_html += "</div>"
    return gallery_results, description_html


categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Custom CSS for beautiful black minimalist theme with split layout
custom_css = """
/* Global dark theme */
:root {
    --primary-bg: #0a0a0a;
    --secondary-bg: #1a1a1a;
    --card-bg: #111111;
    --border-color: #333333;
    --text-primary: #ffffff;
    --text-secondary: #b0b0b0;
    --accent-color: #404040;
    --hover-accent: #505050;
}

.gradio-container {
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Header styling */
.markdown h1 {
    color: #ffffff !important;
    font-size: 2.5rem !important;
    font-weight: 300 !important;
    text-align: center !important;
    margin: 2rem 0 3rem 0 !important;
    letter-spacing: -0.025em !important;
}

/* Input containers */
.input-container {
    background: rgba(17, 17, 17, 0.8) !important;
    border: 1px solid #333333 !important;
    border-radius: 12px !important;
    backdrop-filter: blur(10px) !important;
}

/* Textbox styling */
.gr-textbox {
    background: #111111 !important;
    border: 1px solid #333333 !important;
    border-radius: 8px !important;
    color: #ffffff !important;
}

.gr-textbox:focus {
    border-color: #505050 !important;
    box-shadow: 0 0 0 3px rgba(80, 80, 80, 0.1) !important;
}

/* Dropdown styling */
.gr-dropdown {
    background: #111111 !important;
    border: 1px solid #333333 !important;
    border-radius: 8px !important;
    color: #ffffff !important;
}

.gr-dropdown-menu {
    background: #111111 !important;
    border: 1px solid #333333 !important;
    border-radius: 8px !important;
}

.gr-dropdown-option {
    color: #ffffff !important;
    background: #111111 !important;
}

.gr-dropdown-option:hover {
    background: #404040 !important;
}

/* Button styling */
.gr-button {
    background: linear-gradient(135deg, #333333 0%, #404040 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    font-weight: 500 !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3) !important;
}

.gr-button:hover {
    background: linear-gradient(135deg, #404040 0%, #505050 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4) !important;
}

/* Labels */
label {
    color: #b0b0b0 !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    margin-bottom: 8px !important;
}

/* Gallery styling - Left side */
.gr-gallery {
    background: #111111 !important;
    border: 1px solid #333333 !important;
    border-radius: 12px !important;
    padding: 24px !important;
    height: 600px !important;
    overflow-y: auto !important;
}

.gr-gallery-item {
    border-radius: 8px !important;
    overflow: hidden !important;
    background: #1a1a1a !important;
    border: 1px solid #333333 !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
    margin: 8px !important;
}

.gr-gallery-item:hover {
    transform: translateY(-4px) !important;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.5) !important;
    border-color: #505050 !important;
}

/* Gallery image styling */
.gr-gallery img {
    border-radius: 6px !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    height: auto !important;
}

/* Gallery caption styling - minimal for left side */
.gr-gallery-caption {
    color: #ffffff !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    padding: 12px !important;
    background: #1a1a1a !important;
    text-align: center !important;
}

/* Book descriptions panel - Right side */
.book-descriptions {
    background: #111111 !important;
    border: 1px solid #333333 !important;
    border-radius: 12px !important;
    padding: 24px !important;
    height: 600px !important;
    overflow-y: auto !important;
    color: #ffffff !important;
}

.book-card {
    background: #1a1a1a !important;
    border: 1px solid #333333 !important;
    border-radius: 8px !important;
    padding: 20px !important;
    margin-bottom: 16px !important;
    transition: all 0.3s ease !important;
}

.book-card:hover {
    border-color: #505050 !important;
    background: #222222 !important;
}

.book-header {
    margin-bottom: 12px !important;
    border-bottom: 1px solid #333333 !important;
    padding-bottom: 12px !important;
}

.book-title {
    color: #ffffff !important;
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    margin: 0 0 4px 0 !important;
    line-height: 1.3 !important;
}

.book-author {
    color: #b0b0b0 !important;
    font-size: 0.875rem !important;
    margin: 0 0 8px 0 !important;
    font-style: italic !important;
}

.emotion-badge {
    background: linear-gradient(135deg, #333333, #404040) !important;
    color: #ffffff !important;
    padding: 4px 12px !important;
    border-radius: 16px !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
}

.book-description {
    color: #e0e0e0 !important;
    font-size: 0.875rem !important;
    line-height: 1.6 !important;
    margin-bottom: 12px !important;
}

.book-meta {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    font-size: 0.75rem !important;
    color: #888888 !important;
}

.category {
    background: #333333 !important;
    padding: 4px 8px !important;
    border-radius: 4px !important;
    color: #b0b0b0 !important;
}

.rating {
    color: #ffd700 !important;
    font-weight: 500 !important;
}

/* Remove default margins and padding */
.gr-block, .gr-form, .gr-box {
    background: transparent !important;
}

/* Custom spacing - Symmetric layout */
.gr-row {
    margin-bottom: 24px !important;
    gap: 24px !important;
}

.gr-column {
    gap: 16px !important;
}

/* Ensure equal spacing in input row */
.gr-row > .gr-column {
    margin: 0 12px !important;
}

.gr-row > .gr-column:first-child {
    margin-left: 0 !important;
}

.gr-row > .gr-column:last-child {
    margin-right: 0 !important;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #1a1a1a;
}

::-webkit-scrollbar-thumb {
    background: #404040;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #505050;
}

/* Subtle animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.gradio-container > * {
    animation: fadeIn 0.6s ease-out;
}

/* Input focus effects */
.gr-textbox:focus, .gr-dropdown:focus {
    outline: none !important;
    border-color: #666666 !important;
    box-shadow: 0 0 0 3px rgba(102, 102, 102, 0.1) !important;
}

/* Placeholder text */
.gr-textbox::placeholder {
    color: #666666 !important;
    font-style: italic !important;
}

/* Loading state */
.loading {
    opacity: 0.7 !important;
    pointer-events: none !important;
}

/* Split layout styling */
.split-container {
    display: flex !important;
    gap: 20px !important;
    height: 600px !important;
}

.books-panel {
    flex: 1 !important;
}

.descriptions-panel {
    flex: 1 !important;
}
"""

with gr.Blocks(theme=gr.themes.Base(), css=custom_css) as dashboard:
    gr.Markdown("# Discover Your Next Great Read")

    with gr.Row():
        with gr.Column(scale=3):
            user_query = gr.Textbox(
                label="What kind of story are you looking for?",
                placeholder="A tale of redemption, mysteries in small towns, adventures in space...",
                lines=2
            )
        with gr.Column(scale=1):
            category_dropdown = gr.Dropdown(
                choices=categories,
                label="Category",
                value="All"
            )
        with gr.Column(scale=1):
            tone_dropdown = gr.Dropdown(
                choices=tones,
                label="Mood",
                value="All"
            )
        with gr.Column(scale=1):
            submit_button = gr.Button("✨ Find Books", variant="primary")

    # Get initial books to display
    initial_books, initial_descriptions = get_default_books()

    # Split layout: Books on left, descriptions on right
    with gr.Row():
        with gr.Column(scale=1):
            books_gallery = gr.Gallery(
                label="Books",
                columns=2,
                rows=4,
                height=600,
                show_label=True,
                container=True,
                preview=False,
                value=initial_books
            )
        with gr.Column(scale=1):
            book_descriptions = gr.HTML(
                label="Details",
                value=initial_descriptions
            )

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=[books_gallery, book_descriptions]
    )

if __name__ == "__main__":
    dashboard.launch()