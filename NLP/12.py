from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Example documents
docs = [
    "I love machine learning and NLP",
    "NLP and machine learning are amazing",
    "Cooking recipes are fun to try",
]

# TF-IDF representation
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)

# Compute cosine similarity
sim_matrix = cosine_similarity(X)

print("Cosine Similarity Matrix:\n", sim_matrix)