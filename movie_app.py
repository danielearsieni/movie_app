import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import base64

# File path for precomputed embeddings
EMBEDDINGS_FILE = "mock_movie_embeddings.parquet"
CSV_FILE = "suggestions.csv"

# Caching the loading of embeddings
@st.cache_data
def load_embeddings():
    if not os.path.exists(EMBEDDINGS_FILE):
        # Generate mock data
        titles = [f"Movie {i}" for i in range(100)]
        embeddings = [np.random.rand(384).tolist() for _ in range(100)]
        df = pd.DataFrame({"title": titles, "embedding": embeddings})
        df.to_parquet(EMBEDDINGS_FILE)
    else:
        df = pd.read_parquet(EMBEDDINGS_FILE)
    return df

# Generate a mock embedding from a keyword (random seed based on text)
def mock_embed(text):
    np.random.seed(abs(hash(text)) % 10**6)
    return np.random.rand(384)

# Create a download link for a CSV file
def get_csv_download_link(df, filename):
    csv_data = df.to_csv(index=False)
    b64 = base64.b64encode(csv_data.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download CSV file</a>'
    return href

# Load embeddings
df = load_embeddings()

# Streamlit UI
st.title("🎬 Movie Finder (Mock Example)")
query = st.text_input("Enter a keyword or phrase:")

if query:
    query_embedding = mock_embed(query).reshape(1, -1)
    embedding_matrix = np.vstack(df["embedding"].to_numpy())

    similarities = cosine_similarity(query_embedding, embedding_matrix)[0]
    df["similarity"] = similarities

    top_results = df.sort_values("similarity", ascending=False).head(5)

    st.subheader("Top 5 Similar Movies:")
    for _, row in top_results.iterrows():
        st.markdown(f"**{row['title']}** - Similarity: `{row['similarity']:.4f}`")

    # Save to CSV and offer download link
    top_results[["title", "similarity"]].to_csv(CSV_FILE, index=False)
    st.markdown(get_csv_download_link(top_results[["title", "similarity"]], CSV_FILE), unsafe_allow_html=True)

# Optional button to clear cache (and delete parquet file)
if st.button("🧹 Clear Local Embedding Cache"):
    if os.path.exists(EMBEDDINGS_FILE):
        os.remove(EMBEDDINGS_FILE)
        st.success("Local embedding cache cleared. Please reload the page.")
    else:
        st.info("No cached embeddings found.")

