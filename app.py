import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = [
    ("NASA confirms Earth is round via satellite images", 1),
    ("Government announces new education policy", 1),
    ("Aliens landed in my backyard last night", 0),
    ("Drink lemon water and you will never get cancer", 0),
]
texts, labels = zip(*data)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
model = LogisticRegression()
model.fit(X, labels)

st.title("Fake News Detector")
news_input = st.text_area("Enter news here:")
if st.button("Check"):
    vec = vectorizer.transform([news_input])
    pred = model.predict(vec)[0]
    if pred == 1:
        st.success("REAL NEWS")
    else:
        st.error("FAKE NEWS")