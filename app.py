import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

# --- PAGE SETUP ---
st.set_page_config(page_title="Mental Health CSV Analyzer", layout="wide")
st.title("🧠 Custom Mental Health Sentiment Classifier")
st.write("Upload a dataset to train the AI on specific feelings like Happy, Sad, Anxiety, and Stress.")

# --- STEP 1: UPLOAD FILE ---
st.header("/content/Combined Data.csv")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV
    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.write(df.head())

    # --- STEP 2: COLUMN MAPPING ---
    # This allows you to select which columns are the "Text" and the "Label"
    st.header("2. Configure Columns")
    col1, col2 = st.columns(2)

    with col1:
        text_col = st.selectbox("Select the column containing the SENTENCES:", df.columns)
    with col2:
        label_col = st.selectbox("Select the column containing the FEELINGS (Labels):", df.columns)

    # --- STEP 3: TRAINING ---
    if st.button("🚀 Train Model Now"):
        with st.spinner("Processing text and training..."):
            # Prepare data
            X = df[text_col].astype(str)
            y = df[label_col]

            # Build the Machine Learning Pipeline
            # Tfidf turns words into numbers; LinearSVC classifies them
            model = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words='english')),
                ('clf', LinearSVC())
            ])

            model.fit(X, y)

            # Save model to session state so it stays active
            st.session_state['custom_model'] = model
            st.success("Training complete! Your AI is now ready.")

    # --- STEP 4: PREDICTION ---
    if 'custom_model' in st.session_state:
        st.divider()
        st.header("3. Test Your Custom AI")
        user_input = st.text_input("Type how you are feeling to see what the AI thinks:")

        if user_input:
            prediction = st.session_state['custom_model'].predict([user_input])[0]

            # Display result with a bit of flair
            st.info(f"Analysis Result: **{prediction.upper()}**")

else:
    st.info("Please upload a CSV file to get started. You can use the one we created earlier!")
