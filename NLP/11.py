from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Tiny sentiment dataset
texts = [
    "I love this movie",
    "This film was awful",
    "Amazing performance and great story",
    "Boring and too long",
    "Fantastic acting",
    "Terrible direction"
]
labels = [1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative

# Split data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)

# Bag of Words + Naive Bayes
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

clf = MultinomialNB()
clf.fit(X_train_bow, y_train)

y_pred = clf.predict(X_test_bow)
print("Classification Report:\n", classification_report(y_test, y_pred))