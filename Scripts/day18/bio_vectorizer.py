from sklearn.feature_extraction.text import TfidfVectorizer

bios = [
    "Expert in Python and Machine Learning for social good.",
    "Professional Chef who loves outdoor Hiking and mountains.",
    "Machine Learning enthusiast and mountain hiker."
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(bios)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("Vector Shape:", tfidf_matrix.toarray().shape)
