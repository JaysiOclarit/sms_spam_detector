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

# Custom CSS for a more modern & professional look
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #dfe9f3 0%, #ffffff 100%);
        }
        .main {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.06);
        }
        h1 {
            font-family: 'Segoe UI', sans-serif;
            color: #2c3e50;
        }
        .stTextArea textarea {
            background-color: #ffffff !important;
            color: #000000 !important;
            border: 1px solid #ced4da !important;
            border-radius: 8px !important;
            font-size: 1rem;
        }
        .stButton > button {
            background-color: #4A90E2;
            color: white;
            border: none;
            padding: 0.5rem 1.2rem;
            border-radius: 8px;
            font-size: 1rem;
            transition: background-color 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #3b7ed0;
        }
        .footer {
            text-align: center;
            color: #888;
            font-size: 0.85rem;
            margin-top: 3rem;
        }
    </style>
""", unsafe_allow_html=True)

# Main content
with st.container():
    st.markdown("<div class='main'>", unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center;'>📩 SMS Spam Detector</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size: 1.1rem; color: #555;'>"
        "Quickly determine if a message is <strong>SPAM</strong> or <strong>NOT SPAM</strong> using machine learning."
        "</p>", unsafe_allow_html=True
    )

    user_input = st.text_area("📨 Enter your SMS message below:", height=150, help="Type or paste your SMS message here for analysis.")

    col1, col2, _ = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Analyze Message", use_container_width=True):
            if user_input.strip() == "":
                st.warning("⚠️ Please enter a message to analyze.")
            else:
                input_vector = vectorizer.transform([user_input.lower()])
                prediction = model.predict(input_vector)[0]

                if prediction == "spam":
                    st.error("🚫 This message is classified as **SPAM**.")
                else:
                    st.success("✅ This message is classified as **NOT SPAM**.")

    st.markdown("</div>", unsafe_allow_html=True)

# Collaborators section
st.markdown("---")
st.markdown("### 👥 Project Collaborators")
collaborators = [
    {"name": "Louie Anton Alupay", "github": "https://github.com/Alupay10"},
    {"name": "Jan Christer Oclarit", "github": "https://github.com/JaysiOclarit"}
]
for person in collaborators:
    st.markdown(f"- [{person['name']}]({person['github']})")

# Footer
st.markdown("<div class='footer'>Built with ❤️ using Streamlit & Naive Bayes</div>", unsafe_allow_html=True)
