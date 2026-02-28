import streamlit as st
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import re

# -------------------------
# Safety Disclaimer
# -------------------------
DISCLAIMER = """
⚠️ I’m not a therapist. I provide emotional support, not medical advice.
If you're in crisis, please seek professional help immediately.
India Helpline: AASRA 9820466726
US: Call or text 988
"""

# -------------------------
# Load NLP Models
# -------------------------
@st.cache_resource
def load_models():
    sentiment_model = pipeline("sentiment-analysis")
    vader = SentimentIntensityAnalyzer()
    return sentiment_model, vader

sentiment_model, vader = load_models()

# -------------------------
# Crisis Detection
# -------------------------
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life",
    "self harm", "die", "hopeless", "no reason to live"
]

def detect_crisis(text):
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in CRISIS_KEYWORDS)

# -------------------------
# Mood Detection
# -------------------------
def detect_mood(text):
    if detect_crisis(text):
        return "critical"
    
    vader_score = vader.polarity_scores(text)['compound']
    
    if vader_score >= 0.5:
        return "positive"
    elif vader_score <= -0.5:
        if "anxious" in text.lower() or "panic" in text.lower():
            return "anxious"
        return "sad"
    elif "stress" in text.lower() or "exam" in text.lower():
        return "stressed"
    else:
        return "neutral"

# -------------------------
# Empathetic Responses
# -------------------------
def generate_response(mood, user_text):
    
    if mood == "critical":
        return (
            "I'm really sorry you're feeling this much pain. "
            "You deserve support right now. Please consider reaching out to "
            "AASRA (India: 9820466726) or 988 (US). "
            "If you're in immediate danger, call emergency services. "
            "You're not alone, even if it feels that way."
        )

    elif mood == "sad":
        return (
            "I hear that you're feeling low. It’s completely okay to feel this way. "
            "Would you like to share what’s weighing on you? "
            "Even small steps count. You’ve handled difficult days before."
        )

    elif mood == "anxious":
        return (
            "It sounds like you're feeling anxious. That can be really overwhelming. "
            "Let’s try something simple: inhale for 4 seconds, hold for 7, "
            "exhale for 8. Repeat 3 times. "
            "Your mind deserves calm."
        )

    elif mood == "stressed":
        return (
            "I hear you're feeling stressed—maybe exams or placements? "
            "Break it into small tasks. Study 25 minutes, rest 5. "
            "You’ve tackled tough assignments before—you’ve got this!"
        )

    elif mood == "positive":
        return (
            "That’s great to hear! 😊 "
            "Keep nurturing what’s working for you. "
            "Would you like a quick productivity boost or gratitude exercise?"
        )

    else:
        return (
            "Thanks for sharing. How are you feeling today? "
            "You can talk freely here—this space is for you."
        )

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Mental Health Companion", page_icon="🧠")

st.title("🧠 Student Mental Health Companion")
st.write(DISCLAIMER)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("How are you feeling today?")

if user_input:
    mood = detect_mood(user_input)
    response = generate_response(mood, user_input)
    
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Companion", response))

for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Companion:** {message}")