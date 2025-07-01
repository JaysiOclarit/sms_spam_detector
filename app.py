import streamlit as st
import pickle

# Load model and vectorizer
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Page configuration
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a more professional UI
st.markdown("""
    <style>
        body {
            background-color: #f5f7fa;
        }
        .main {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 2rem;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);
        }
        h1 {
            font-family: 'Segoe UI', sans-serif;
            color: #2c3e50;
        }
        .stTextArea textarea {
            background-color: #f9f9f9;
            border: 1px solid #ccc;
            border-radius: 6px;
        }
        .footer {
            text-align: center;
            color: #888;
            font-size: 0.85rem;
            margin-top: 3rem;
        }
    </style>
""", unsafe_allow_html=True)

# Container for main content
with st.container():
    st.markdown("<h1 style='text-align: center;'>📩 SMS Spam Detector</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size: 1.1rem; color: #555;'>"
        "Quickly determine if a message is <strong>SPAM</strong> or <strong>NOT SPAM</strong> using machine learning."
        "</p>", unsafe_allow_html=True
    )
    st.write("")

    user_input = st.text_area("Enter your SMS message below:", height=150, help="Type or paste your SMS message here for analysis.")

    col1, col2, _ = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Analyze Message", use_container_width=True):
            if user_input.strip() == "":
                st.warning("⚠️ Please enter a message to analyze.")
            else:
                # Predict
                input_vector = vectorizer.transform([user_input.lower()])
                prediction = model.predict(input_vector)[0]

                # Display result
                if prediction == "spam":
                    st.error("🚫 This message is classified as **SPAM**.")
                else:
                    st.success("✅ This message is classified as **NOT SPAM**.")

# Footer
st.markdown("<div class='footer'>Built with ❤️ using Streamlit & Naive Bayes</div>", unsafe_allow_html=True)
