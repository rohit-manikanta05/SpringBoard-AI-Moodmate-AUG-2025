from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Sample corpus
corpus = [
    "I love natural language processing",
    "Language processing is fun",
    "I love coding in Python",
    "Python and NLP are powerful"
]

# Create Bag of Words model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

# Convert to dataframe for readability
bow_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

print("Vocabulary:", vectorizer.get_feature_names_out())
print("\nBag of Words Matrix:")
print(bow_df)