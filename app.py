import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 1. Preprocessing Function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower() # Lowercasing
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    return text

# 2. Build and Train Model
@st.cache_resource
def train_model():
    # Ensure the file is in your GitHub repo root
    df = pd.read_csv("mental_health_dataset.csv")
    
    # Preprocessing the dataset
    df['cleaned_text'] = df['Mood_Description'].apply(clean_text)

    # Pipeline: TF-IDF + Logistic Regression
    model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression())
    ])
    
    # Training on your specific columns
    model.fit(df['cleaned_text'], df['Mental_Health_Status'])
    return model

# 3. Streamlit Interface
st.set_page_config(page_title="Mental Health AI", page_icon="🧠")
st.title("🧠 Mental Health Sentiment Detector")
st.write("Identify your current emotional state using NLP.")

# Initialize Model
try:
    model = train_model()
except FileNotFoundError:
    st.error("Error: 'mental_health_dataset.csv' not found. Please ensure it is uploaded to GitHub.")
    st.stop()

user_input = st.text_area("Your Thoughts:", placeholder="I've been feeling really overwhelmed lately...")

# 4. Feelings Mapping (Ensuring words are clear)
# Adjust these keys to match exactly what is in your CSV 'Mental_Health_Status' column
feeling_map = {
    "Happy": "Happy and Positive",
    "Sad": "Sad or Low Mood",
    "Anxiety": "Anxious or Panicked",
    "Stress": "Stressed and Overwhelmed",
    "Normal": "Stable and Calm"
}

if st.button("Analyze Sentiment"):
    if user_input:
        # Get raw prediction from model
        raw_prediction = model.predict([clean_text(user_input)])[0]
        
        # Get user-friendly words from our map
        feeling_text = feeling_map.get(raw_prediction, raw_prediction)

        st.subheader("Analysis Results:")

        # Color-coded results with words + emojis
        if "Happy" in raw_prediction or "Normal" in raw_prediction:
            st.success(f"**Feeling Detected:** {feeling_text} 😊")
        elif "Sad" in raw_prediction or "Depression" in raw_prediction:
            st.error(f"**Feeling Detected:** {feeling_text} 😢")
        elif "Anxiety" in raw_prediction:
            st.warning(f"**Feeling Detected:** {feeling_text} 😰")
        elif "Stress" in raw_prediction:
            st.info(f"**Feeling Detected:** {feeling_text} 😫")
        else:
            st.write(f"**Feeling Detected:** {feeling_text}")
            
        st.write("---")
        st.caption("Note: This is an AI prediction for educational purposes and not a clinical diagnosis.")
    else:
        st.warning("Please enter some text first!")
