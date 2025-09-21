from gensim.models import Word2Vec

# Sample sentences (tokenized)
sentences = [
    ["i", "love", "natural", "language", "processing"],
    ["language", "processing", "is", "fun"],
    ["deep", "learning", "advances", "artificial", "intelligence"],
    ["python", "is", "great", "for", "machine", "learning"]
]

# Train a Word2Vec model
model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, workers=2, sg=1)

# Explore embeddings
print("Vector for 'language':\n", model.wv["language"])
print("\nMost similar to 'learning':", model.wv.most_similar("learning"))
print("\nSimilarity between 'python' and 'language':", model.wv.similarity("python", "language"))