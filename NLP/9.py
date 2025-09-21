from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Sample documents
docs = [
    "machine learning is fun",
    "deep learning advances machine intelligence",
    "artificial intelligence and machine learning"
]

# Create TF-IDF model
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(docs)

# Convert to DataFrame
tfidf_df = pd.DataFrame(X.toarray(), columns=tfidf.get_feature_names_out())

print("Vocabulary:", tfidf.get_feature_names_out())
print("\nTF-IDF Matrix:")
print(tfidf_df.round(3))