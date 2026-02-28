import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 1. Preprocessing Function
def clean_text(text):
    text = text.lower() # Lowercasing
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    return text

# 2. Build and Train Model
@st.cache_resource
def train_model():
    df = pd.read_csv("mental_health.csv")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Pipeline: TF-IDF + Logistic Regression
    model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression())
    ])
    model.fit(df['cleaned_text'], df['label'])
    return model

# 3. Streamlit Interface
st.set_page_config(page_title="Mental Health AI", page_icon="🧠")
st.title("🧠 Mental Health Sentiment Detector")
st.write("Enter how you feel to identify your current emotional state.")

model = train_model()

user_input = st.text_area("Your Thoughts:", placeholder="I've been feeling really overwhelmed lately...")

if st.button("Analyze Sentiment"):
    if user_input:
        prediction = model.predict([clean_text(user_input)])[0]
        
        # Color-coded results
        if prediction == "Happy":
            st.success(f"Result: {prediction} 😊")
        elif prediction == "Sad":
            st.error(f"Result: {prediction} 😢")
        elif prediction == "Anxiety":
            st.warning(f"Result: {prediction} 😰")
        else:
            st.info(f"Result: {prediction} 😫")
    else:
        st.warning("Please enter some text first!")
