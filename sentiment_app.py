import streamlit as st
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
import nltk
import seaborn as sns
import matplotlib.pyplot as plt

# Download VADER Lexicon
nltk.download('vader_lexicon')

# Initialize models
analyzer = SentimentIntensityAnalyzer()
sentiment_pipeline = pipeline("sentiment-analysis")

# Functions for sentiment analysis
def analyze_sentiment_textblob(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, polarity

def analyze_sentiment_vader(text):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound > 0.05:
        sentiment = "Positive"
    elif compound < -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, scores

def analyze_sentiment_transformers(text):
    result = sentiment_pipeline(text)
    label = result[0]['label']
    score = result[0]['score']
    return label, score

# Visualization function
def visualize_sentiment(text):
    textblob_sentiment, textblob_score = analyze_sentiment_textblob(text)
    vader_sentiment, vader_scores = analyze_sentiment_vader(text)
    transformers_label, transformers_score = analyze_sentiment_transformers(text)

    methods = ['TextBlob', 'VADER', 'Transformers']
    scores = [textblob_score, vader_scores['compound'], transformers_score]

    fig, ax = plt.subplots()
    sns.barplot(x=methods, y=scores, palette='viridis', ax=ax)
    ax.set_title("Sentiment Scores")
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    return fig

# Streamlit App
st.title("Sentiment Analysis Tool")
st.write("Analyze the sentiment of your text using three different models: TextBlob, VADER, and Transformers.")

# Input from user
user_text = st.text_area("Enter text for sentiment analysis:")

if user_text:
    st.subheader("Results")

    # Analyze using TextBlob
    textblob_sentiment, textblob_score = analyze_sentiment_textblob(user_text)
    st.write(f"**TextBlob:** Sentiment: {textblob_sentiment}, Polarity: {textblob_score}")

    # Analyze using VADER
    vader_sentiment, vader_scores = analyze_sentiment_vader(user_text)
    st.write(f"**VADER:** Sentiment: {vader_sentiment}, Compound Score: {vader_scores['compound']}")

    # Analyze using Transformers
    transformers_label, transformers_score = analyze_sentiment_transformers(user_text)
    st.write(f"**Transformers:** Sentiment: {transformers_label}, Confidence Score: {transformers_score}")

    # Visualize
    st.subheader("Visualization")
    fig = visualize_sentiment(user_text)
    st.pyplot(fig)
