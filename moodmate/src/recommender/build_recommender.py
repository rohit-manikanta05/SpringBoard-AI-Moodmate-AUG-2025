import os, re, joblib, pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.base import BaseEstimator, TransformerMixin

DATA_CSV = os.path.join("data", "music", "songs.csv")
VECT_PATH = os.path.join("models", "tfidf_vectorizer.joblib")
INDEX_PATH = os.path.join("models", "song_index.joblib")
SONGS_PARQUET = os.path.join("models", "songs_clean.parquet")

class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        cleaned = []
        for t in X:
            t = str(t).lower()
            t = re.sub(r"[^a-z0-9\s]+", " ", t)
            t = re.sub(r"\s+", " ", t).strip()
            cleaned.append(t)
        return cleaned

def build_corpus(df):
    # Combine tags/genre/mood into a single text field
    parts = []
    for _, r in df.iterrows():
        text = " ".join([str(r.get(c, "")) for c in ["tags", "genre", "mood"]])
        parts.append(text)
    return parts

def main():
    assert os.path.exists(DATA_CSV), f"Missing {DATA_CSV}. Please place your songs.csv there."
    df = pd.read_csv(DATA_CSV)
    # Basic cleaning + fill
    for col in ["track_id","title","artist","genre","tags","mood"]:
        if col not in df.columns:
            df[col] = ""
    df = df.fillna("")
    df["search_query"] = (df["artist"].astype(str) + " " + df["title"].astype(str)).str.strip()

    corpus_raw = build_corpus(df)

    pipe = Pipeline([
        ("clean", TextCleaner()),
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2)))
    ])
    X = pipe.fit_transform(corpus_raw)

    joblib.dump(pipe, VECT_PATH)
    joblib.dump(X, INDEX_PATH)
    df.to_parquet(SONGS_PARQUET, index=False)
    print(f"Saved vectorizer → {VECT_PATH}")
    print(f"Saved index      → {INDEX_PATH}")
    print(f"Saved songs      → {SONGS_PARQUET}")

if __name__ == "__main__":
    main()
