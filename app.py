import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 1. SIMPLE DATASET (100 Happy / 100 Sad)
# I used very basic words so the AI understands perfectly.
happy_words = [
    "good", "great", "wow", "love", "best", "nice", "happy", "cool", "amazing", "perfect",
    "awesome", "fun", "wonderful", "brilliant", "excellent", "fantastic", "super", "lovely", "glad", "best movie",
    "must watch", "loved it", "so good", "very nice", "enjoyed it", "masterpiece", "beautiful", "smart", "classic", "top",
    "favorite", "sweet", "cool film", "great acting", "liked it", "five stars", "winner", "gem", "inspiring", "bright",
    "happy ending", "great story", "pretty", "funny", "charming", "impressive", "smooth", "bold", "superb", "magic"
] * 4  # This repeats the list to reach 200 total samples easily

sad_words = [
    "bad", "worse", "hate", "worst", "sad", "boring", "slow", "terrible", "angry", "waste",
    "poor", "awful", "ugly", "horrible", "lazy", "dumb", "no", "never", "avoid", "garbage",
    "trash", "annoying", "cheap", "bad movie", "hated it", "not good", "don't watch", "so bad", "failed", "broken",
    "painful", "dull", "lame", "weak", "mess", "disaster", "stupid", "pointless", "zero stars", "gross",
    "cringe", "bad acting", "boring plot", "waste of time", "unhappy", "awful film", "rubbish", "silly", "wrong", "dry"
] * 4 

# Create the data table
data = {
    'text': happy_words + sad_words,
    'label': [1]*200 + [0]*200 # 1 = Positive, 0 = Negative
}
df = pd.DataFrame(data)

# 2. THE AI BRAIN (The Pipeline)
# TF-IDF turns words into numbers. LogisticRegression is the judge.
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])

# Train the AI
model.fit(df['text'], df['label'])

# 3. THE WEBSITE INTERFACE
st.set_page_config(page_title="Easy Sentiment")
st.title("🎬 Simple Movie Review AI")
st.write("Is your review Happy or Sad? Type it below!")

# User types here
user_input = st.text_input("Enter your review:", "This movie was great")

if st.button("Check Now"):
    # The AI makes a guess
    prediction = model.predict([user_input])[0]
    
    if prediction == 1:
        st.success("The AI thinks: **HAPPY** ✨")
        st.balloons()
    else:
        st.error("The AI thinks: **SAD** 📉")

# Sidebar info
st.sidebar.write(f"Total reviews learned: {len(df)}")
