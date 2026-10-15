# Movie Recommender

A conversational movie assistant. Ask for movies in plain language and it answers from movies it actually retrieves, and politely declines questions that aren't about movies.

The project started as a content-based recommender on the TMDB 5000 Movies dataset (CountVectorizer + cosine similarity) and was rebuilt in 2025 as a retrieval-augmented generation (RAG) chatbot.

## How it works
Question → MiniLM embedding → FAISS similarity search → Gemini 1.5 Flash answers from the retrieved movies → Streamlit chat

- **Embeddings:** `all-MiniLM-L6-v2` (Sentence Transformers), precomputed in `movie_embeddings.pkl`
- **Retrieval:** FAISS index (`faiss_index.bin`) over the movie data in `movies.csv`
- **Generation:** Google Gemini answers using only the retrieved context
- **Guardrail:** out-of-domain (non-movie) questions are rejected

## Tech stack
Python, Sentence Transformers, FAISS, Google Gemini, Streamlit, Pandas, NumPy, PyTorch

## Run it
```bash
git clone https://github.com/Vinay2688/Movie-Recommender.git
cd Movie-Recommender
pip install -r requirements.txt
```
Get a free Gemini API key from Google AI Studio and set it as an environment variable:
```bash
export GEMINI_API_KEY="your-key"        # macOS / Linux
setx GEMINI_API_KEY "your-key"          # Windows (reopen the terminal afterwards)
```
Then start the app:
```bash
streamlit run chatty.py
```

## Data
Movie metadata comes from the TMDB 5000 Movies dataset.
