import streamlit as st
import pickle

# Load model and vectorizer
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Streamlit page configuration
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered",
)

# Main title
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>📩 SMS Spam Detector</h1>", unsafe_allow_html=True)
st.markdown("#### Easily check whether a message is **SPAM** or **Not Spam** using machine learning.", unsafe_allow_html=True)
st.write("---")

# User input
user_input = st.text_area("✉️ Enter your message below:", height=150)

# Predict button
if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message to analyze.")
    else:
        # Preprocess and predict
        input_vector = vectorizer.transform([user_input.lower()])
        prediction = model.predict(input_vector)[0]

        # Display result
        if prediction == "spam":
            st.error("🚫 This message is **SPAM**.")
        else:
            st.success("✅ This message is **NOT SPAM**.")

# Footer
st.write("---")
st.markdown("<small>Built with ❤️ using Streamlit & Naive Bayes</small>", unsafe_allow_html=True)
