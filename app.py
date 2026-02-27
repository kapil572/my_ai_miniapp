import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 1. Expanded Dataset (More data = better accuracy)
data = {
    'text': [
        "I love this movie", "Great acting and plot", "A masterpiece", "Amazing experience",
        "Worst movie ever", "I hated the plot", "Waste of time", "Terrible acting",
        "It was okay, not great", "Boring and slow", "Simply brilliant", "Highly recommended",
        "Not my cup of tea", "A total disaster", "One of the best", "Poor quality"
    ],
    'label': [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0]
}
df = pd.DataFrame(data)

# 2. Advanced Model Pipeline
# ngram_range=(1,2) lets the model see single words AND pairs of words.
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
    ('classifier', LogisticRegression())
])

# Train the model
model_pipeline.fit(df['text'], df['label'])

# 3. Enhanced UI
st.set_page_config(page_title="Sentiment Pro", page_icon="🧠")
st.title("🧠 Advanced Sentiment Analyzer")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    user_text = st.text_area("Enter a detailed review:", placeholder="e.g., The acting was great but the plot was a bit slow...")
    
with col2:
    st.write("### Stats")
    st.write(f"**Training Samples:** {len(df)}")
    st.write("**Model:** Logistic Regression")

if st.button("Analyze Sentiment", use_container_width=True):
    if user_text.strip():
        # Predict class and probability
        prediction = model_pipeline.predict([user_text])[0]
        proba = model_pipeline.predict_proba([user_text])[0]
        confidence = proba[1] if prediction == 1 else proba[0]
        
        st.markdown("---")
        if prediction == 1:
            st.balloons()
            st.success(f"### Positive Sentiment (Confidence: {confidence:.2%})")
        else:
            st.error(f"### Negative Sentiment (Confidence: {confidence:.2%})")
            
        # Optional: show the "strength" of the sentiment
        st.progress(confidence)
    else:
        st.warning("Please enter some text first!")
