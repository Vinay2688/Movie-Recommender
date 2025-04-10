import os
os.environ["TORCH_USE_RTLD_GLOBAL"] = "1"
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"

import torch


import pandas as pd
import faiss
import numpy as np
import google.generativeai as genai
import streamlit as st
from sentence_transformers import SentenceTransformer
import pickle



# Streamlit UI
st.set_page_config(page_title="RAG Movie Recommender", layout="wide")

# Configure Gemini AI


genai.configure(api_key=os.getenv("GEMINI_API_KEY")) # Replace with your actual API key

# Load Movies Dataset
movies = pd.read_csv("movies.csv")

# Load Sentence Transformer Model
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embed_model()

# Paths
FAISS_INDEX_PATH = "faiss_index.bin"
EMBEDDINGS_PATH = "movie_embeddings.pkl"

# ✅ Deployment-safe version: Only load prebuilt index and embeddings
@st.cache_resource
def load_faiss_index():
    with open(EMBEDDINGS_PATH, "rb") as f:
        movie_embeddings = pickle.load(f)
    index = faiss.read_index(FAISS_INDEX_PATH)
    return index, movie_embeddings

index, movie_embeddings = load_faiss_index()

# Get similar movies
def get_similar_movies(query, top_n=5):
    query_embedding = embed_model.encode(query, convert_to_numpy=True)
    query_embedding = np.expand_dims(query_embedding, axis=0)
    distances, indices = index.search(query_embedding, top_n)
    retrieved_movies = movies.iloc[indices[0]]["title"].tolist()
    return retrieved_movies

# Gemini AI response
def generate_response(user_query, retrieved_movies):
    prompt = f"""
    User asked: {user_query}
    Based on our database, here are some recommended movies: {', '.join(retrieved_movies)}
    Provide a friendly, engaging response.
    """
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    response = model.generate_content(prompt)
    return response.text

# Streamlit layout
st.markdown("<h1 style='text-align: center;'>🎥 RAG-Powered Movie Recommender</h1>", unsafe_allow_html=True)

query = st.text_input("Type your query here...", placeholder="E.g., Suggest me a sci-fi movie")
chat_history = st.container()

if query:
    recommended_movies = get_similar_movies(query, top_n=5)
    response = generate_response(query, recommended_movies)

    with chat_history:
        st.markdown(f"**🧑‍💻 You:** {query}")
        st.markdown(f"**🤖 AI:** Here are some great movie recommendations! 🍿")
        st.write(", ".join(recommended_movies))
        st.markdown(f"**🤖 AI:** {response}")
