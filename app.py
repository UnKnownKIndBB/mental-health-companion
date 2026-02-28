import streamlit as st
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import random
import re

# ------------------------- SAFETY DISCLAIMER -------------------------
DISCLAIMER = """
⚠️ **Important**: I am NOT a licensed therapist or doctor.  
I can offer emotional support, listening, and simple coping tips, but **I am not a substitute for professional help**.

If you are in crisis or feel unsafe:
- **India**: AASRA Helpline → 9820466726 (24×7)
- **US**: 988 Suicide & Crisis Lifeline (call or text)
- Emergency: Call 112 / 108 / local emergency services immediately.

You are never alone. Reach out — help is available.
"""

# ------------------------- LOAD MODELS (cached) -------------------------
@st.cache_resource
def load_models():
    st.write("Loading AI models... (only once)")  # visible only on first run
    sentiment_model = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    vader = SentimentIntensityAnalyzer()
    return sentiment_model, vader

sentiment_model, vader = load_models()

# ------------------------- IMPROVED CRISIS DETECTION -------------------------
CRISIS_PATTERNS = [
    r"\b(suicide|suicidal|kill myself|end my life|die|self harm|cutting|overdose)\b",
    r"no (point|reason) (to live|living|exist)",
    r"want to disappear|not wake up|better off dead",
    r"\b(jump|hang|pill|cut).*myself",
    r"i can't do this anymore.*(anymore|life)",
]

def detect_crisis(text: str) -> bool:
    if not text:
        return False
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in CRISIS_PATTERNS)

# ------------------------- MOOD DETECTION (Hybrid VADER + Transformers) -------------------------
def detect_mood(text: str) -> str:
    if detect_crisis(text):
        return "critical"

    # VADER score (great for short emotional text)
    vader_score = vader.polarity_scores(text)['compound']

    # Transformers model (more contextual)
    hf_result = sentiment_model(text)[0]
    hf_label = hf_result['label']      # POSITIVE / NEGATIVE
    hf_score = hf_result['score']

    # Strong positive
    if vader_score >= 0.45 or (hf_label == "POSITIVE" and hf_score > 0.85):
        return "positive"

    # Strong negative
    if vader_score <= -0.45 or (hf_label == "NEGATIVE" and hf_score > 0.88):
        lowered = text.lower()
        if any(word in lowered for word in ["anxious", "panic", "worry", "nervous", "heart racing", "overwhelmed"]):
            return "anxious"
        if any(word in lowered for word in ["stress", "exam", "deadline", "placement", "study", "assignment", "overloaded"]):
            return "stressed"
        return "sad"

    # Mild negative or neutral
    if vader_score <= -0.15:
        return "sad" if "sad" in text.lower() or "low" in text.lower() else "neutral"
    
    return "neutral"

# ------------------------- RESPONSE POOLS (Varied & Natural) -------------------------
RESPONSES = {
    "critical": [
        "I'm really sorry you're feeling this much pain right now. You don't have to go through this alone. Please reach out to AASRA (9820466726) or 988 immediately. I'm here to listen until you can talk to them.",
        "This sounds incredibly heavy. Your life matters, and help is available 24/7. Call AASRA at 9820466726 right now — they are trained for exactly this. I'm staying with you."
    ],
    "sad": [
        "I hear how heavy this feels right now. It's okay to not be okay. Would you like to tell me what's been weighing on you?",
        "That sounds really painful. You've survived every difficult day so far — I'm proud of you for that. I'm here with you.",
        "Feeling low is exhausting. Even sharing it here is a small brave step. What's one thing that's been hardest lately?",
        "I'm really sorry you're carrying this. You are not alone in this moment."
    ],
    "anxious": [
        "Anxiety can feel like everything is too loud inside. Let's try a quick calming breath together: Inhale 4 seconds… hold 7… exhale 8. Repeat 3 times whenever you're ready.",
        "That racing feeling is really uncomfortable. You're safe here. Would you like a 60-second grounding exercise?",
        "I can hear the worry in your words. Your nervous system is just trying to protect you. Let's slow it down together."
    ],
    "stressed": [
        "Exams, placements, assignments — it's a lot. You've handled tough phases before. Let's break today into tiny manageable pieces.",
        "I hear the stress loud and clear. The 25-minute Pomodoro + 5-minute break technique works wonders for most students. Want me to guide you through one?",
        "You're not lazy or failing — you're just carrying a heavy load. One task at a time. You've got this."
    ],
    "positive": [
        "That's wonderful to hear! 😊 Keep doing whatever is lighting you up right now. Want a quick gratitude boost or productivity tip?",
        "Love seeing this energy! Celebrate the small wins today. Is there anything you'd like to build on?",
        "This positivity looks good on you! What's one thing that made you smile today?"
    ],
    "neutral": [
        "Thanks for checking in. How are you really feeling today? I'm here whenever you want to talk.",
        "I'm listening. What's on your mind today?"
    ]
}

# ------------------------- SMART RESPONSE GENERATOR -------------------------
def generate_response(mood: str, user_text: str) -> str:
    # Check conversation history for escalation
    if "mood_history" in st.session_state and len(st.session_state.mood_history) >= 3:
        recent_moods = st.session_state.mood_history[-3:]
        if recent_moods.count("sad") >= 2 or recent_moods.count("anxious") >= 2:
            return ("It feels like this heaviness has been around for a few days now. "
                    "I'm genuinely concerned about you. Would you feel comfortable talking to someone on campus "
                    "or calling AASRA (9820466726)? I'm here either way.")

    # Normal varied response
    if mood in RESPONSES:
        response = random.choice(RESPONSES[mood])
    else:
        response = "Thanks for sharing. I'm here to listen — no judgment."

    # Add gentle follow-up for most moods
    if mood in ["sad", "anxious", "stressed", "neutral"]:
        response += "\n\nI'm here whenever you want to continue."
    
    return response

# ------------------------- STREAMLIT UI -------------------------
st.set_page_config(page_title="MindMate • Student Mental Health Companion", page_icon="🧠", layout="centered")

st.title("🧠 MindMate")
st.caption("A safe space for students • Built with care")

st.markdown(DISCLAIMER)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "mood_history" not in st.session_state:
    st.session_state.mood_history = []

# Sidebar
with st.sidebar:
    st.header("About MindMate")
    st.write("Detects mood using AI and responds with empathy + practical tips.")
    st.write("Made for students, by students who care.")
    
    if st.button("🗑️ Clear Conversation"):
        st.session_state.chat_history = []
        st.session_state.mood_history = []
        st.rerun()

# Display chat history with nice bubbles
for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.chat_message("user").write(message)
    else:
        st.chat_message("assistant").write(message)

# User input
user_input = st.chat_input("How are you feeling today? (It's okay to say anything)")

if user_input:
    # Save user message
    st.session_state.chat_history.append(("You", user_input))
    
    # Detect mood
    mood = detect_mood(user_input)
    st.session_state.mood_history.append(mood)
    
    # Generate response
    response = generate_response(mood, user_input)
    
    # Save bot message
    st.session_state.chat_history.append(("Companion", response))
    
    # Rerun to show new messages
    st.rerun()