import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Simple Data
reviews = ["I love this", "Great movie", "Worst ever", "Hated it", "Amazing", "Bad"]
labels = [1, 1, 0, 0, 1, 0] # 1 is Positive, 0 is Negative

# 2. Build the Model (The "Brain")
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(reviews)
model = LogisticRegression().fit(X, labels)

# 3. The Website Part
st.title("🎬 Simple Movie Sentiment AI")
user_text = st.text_input("Type a review (e.g., 'It was great'):")

if st.button("Analyze"):
    prediction = model.predict(tfidf.transform([user_text]))[0]
    if prediction == 1:
        st.success("Positive ✨")
    else:
        st.error("Negative 📉")
