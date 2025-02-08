import streamlit as st
import string
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Function to analyze sentiment and emotions
def analyze_text(input_text):
    lower_case = input_text.lower()
    cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

    # Tokenize words
    tokenized_words = word_tokenize(cleaned_text, "english")

    # Remove stopwords
    final_words = [word for word in tokenized_words if word not in stopwords.words('english')]

    # Lemmatize words
    lemma_words = [WordNetLemmatizer().lemmatize(word) for word in final_words]

    # Emotion mapping
    emotion_list = []
    try:
        with open('emotion.txt', 'r') as file:
            for line in file:
                clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
                word, emotion = clear_line.split(':')
                if word in lemma_words:
                    emotion_list.append(emotion)
    except FileNotFoundError:
        st.error("The file 'emotion.txt' is missing. Please provide it in the working directory.")
        return None, None

    # Sentiment analysis
    sentiment_analyzer = SentimentIntensityAnalyzer()
    scores = sentiment_analyzer.polarity_scores(cleaned_text)

    sentiment = "Neutral"
    if scores['neg'] > scores['pos']:
        sentiment = "Negative"
    elif scores['neg'] < scores['pos']:
        sentiment = "Positive"

    return Counter(emotion_list), sentiment

# Streamlit UI
st.set_page_config(
    page_title="Sentiment & Emotion Analyzer",
    page_icon="🌀",
    layout="wide"
)

# Styling
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #a8edea, #fed6e3);
        font-family: 'Arial', sans-serif;
        color: #2C3E50;
    }
    .main-title {
        font-size: 4rem;
        font-weight: 800;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 10px;
        background: -webkit-linear-gradient(45deg, #ff7e5f, #feb47b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
    }
    .sub-title {
        font-size: 1.2rem;
        color: #ffffff;
        text-align: center;
        margin-bottom: 40px;
    }
    .btn {
        font-size: 1.2rem;
        color: white;
        background: linear-gradient(to right, #6A82FB, #FC5C7D);
        border: none;
        padding: 10px 20px;
        border-radius: 20px;
        cursor: pointer;
        text-align: center;
    }
    .btn:hover {
        background: linear-gradient(to right, #FC5C7D, #6A82FB);
    }
    footer {
        text-align: center;
        font-size: 0.9rem;
        color: #888;
        margin-top: 40px;
        border-top: 1px solid #ddd;
        padding-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("<div class='main-title'>Sentiment & Emotion Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Discover the sentiment and emotional tone of speech or text with ease.</div>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Choose Input Mode 🛠️")
input_mode = st.sidebar.radio("Input Options:", ["Enter Text", "Upload File"])

# Input Text or File
if input_mode == "Enter Text":
    input_text = st.text_area("Enter your text below:", placeholder="Type or paste the text here...", height=200)
else:
    uploaded_file = st.file_uploader("Upload a .txt file:", type=["txt"])
    if uploaded_file is not None:
        input_text = uploaded_file.read().decode("utf-8")
    else:
        input_text = ""

# Analyze Button
if st.button("🔍 Analyze"):
    if input_text.strip():
        emotions, sentiment = analyze_text(input_text)

        if emotions:
            # Sentiment Display
            st.markdown(
                f"<div style='background: linear-gradient(to right, #6A82FB, #FC5C7D); color: white; border-radius: 10px; padding: 15px; font-size: 1.5rem; text-align: center;'>"
                f"The sentiment is: <b>{sentiment}</b></div>",
                unsafe_allow_html=True,
            )

            # Emotion Chart
            st.markdown("<h3 style='color: #34495E;'>Emotion Analysis Chart 📊</h3>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(emotions.keys(), emotions.values(), color=['#6A82FB', '#FC5C7D', '#D4FC79'], edgecolor='black')
            plt.title("Emotion Analysis", fontsize=16, color="#34495E")
            plt.xlabel("Emotions", fontsize=14)
            plt.ylabel("Count", fontsize=14)
            plt.xticks(rotation=45)
            st.pyplot(fig)
    else:
        st.warning("⚠️ Please provide text or upload a file for analysis.")

# Footer
st.markdown("""
    <footer>
        NLP-Based Sentiment and Emotion Analyzer
    </footer>
""", unsafe_allow_html=True)
