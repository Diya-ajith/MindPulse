import os, random
import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import plotly.express as px
from transformers import BertTokenizer, BertForSequenceClassification

# ================= OFFLINE MODE =================
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="MindPulse",
    page_icon="🧠",
    layout="wide"
)

# ================= NEON DARK‑PURPLE + GLASS UI =================
st.markdown("""
<style>
.stApp {
  background: linear-gradient(135deg,#1a0033,#240046,#3c096c,#5a189a);
  background-size: 400% 400%;
  animation: bgmove 18s ease infinite;
  color: white;
}
@keyframes bgmove {
  0% {background-position: 0% 50%;}
  50% {background-position: 100% 50%;}
  100% {background-position: 0% 50%;}
}
label { color:white !important; font-weight:600; }
textarea {
  color:#000000 !important;
  background: rgba(255,255,255,0.9) !important;
  border-radius:14px !important;
}
textarea::placeholder { color:#555555 !important; }
.glass {
  background: rgba(255,255,255,0.20);
  backdrop-filter: blur(26px);
  border-radius: 20px;
  padding: 20px;
  margin-bottom: 16px;
  box-shadow: 0 20px 45px rgba(0,0,0,0.55);
}
.neon-title {
  text-align:center;
  font-size:52px;
  font-weight:900;
  background: linear-gradient(90deg,#4facfe,#00c6ff,#1de9b6);
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
}
.badge {
  display:inline-block;
  padding:6px 14px;
  border-radius:20px;
  background: linear-gradient(90deg,#f093fb,#f5576c);
  font-weight:600;
  color:white;
}
.chat-user {
  background: linear-gradient(135deg,#667eea,#764ba2);
  padding:14px;
  border-radius:18px;
  margin:10px 0;
  text-align:right;
}
.chat-ai {
  background: linear-gradient(135deg,#43e97b,#38f9d7);
  padding:14px;
  border-radius:18px;
  margin:10px 0;
  color:#0b2e1c;
}
</style>
""", unsafe_allow_html=True)

# ================= EMOTION LABELS =================
LABELS = [
 'admiration','amusement','anger','annoyance','approval','caring',
 'confusion','curiosity','desire','disappointment','disapproval',
 'disgust','embarrassment','excitement','fear','gratitude','grief',
 'joy','love','nervousness','optimism','pride','realization','relief',
 'remorse','sadness','surprise','neutral'
]

# ================= ONE‑LINE IN‑DEPTH EXPLANATIONS =================
EMOTION_DESCRIPTION = {
 "admiration":"Admiration reflects deep respect and inspiration toward someone’s qualities or actions.",
 "amusement":"Amusement represents light‑hearted enjoyment that promotes emotional relaxation.",
 "anger":"Anger arises from frustration, injustice, or unmet emotional expectations.",
 "annoyance":"Annoyance reflects mild irritation caused by repeated or minor disturbances.",
 "approval":"Approval indicates agreement and positive validation of decisions or behavior.",
 "caring":"Caring expresses empathy, emotional warmth, and concern for others’ well‑being.",
 "confusion":"Confusion occurs when information feels unclear or difficult to process.",
 "curiosity":"Curiosity reflects a strong desire to explore, learn, and understand.",
 "desire":"Desire represents strong emotional or motivational longing toward a goal or outcome.",
 "disappointment":"Disappointment arises when expectations are not fulfilled.",
 "disapproval":"Disapproval reflects dissatisfaction or disagreement based on personal values.",
 "disgust":"Disgust represents strong emotional rejection toward unpleasant stimuli.",
 "embarrassment":"Embarrassment occurs due to social discomfort or self‑conscious awareness.",
 "excitement":"Excitement reflects high emotional energy and positive anticipation.",
 "fear":"Fear signals perceived threat or uncertainty requiring caution.",
 "gratitude":"Gratitude expresses appreciation and thankfulness toward others.",
 "grief":"Grief reflects deep emotional pain following loss or separation.",
 "joy":"Joy represents genuine happiness and emotional fulfillment.",
 "love":"Love reflects deep emotional bonding, trust, and affection.",
 "nervousness":"Nervousness indicates anxiety before uncertain or important events.",
 "optimism":"Optimism reflects hopefulness and positive expectations about the future.",
 "pride":"Pride arises from achievement, confidence, and self‑worth.",
 "realization":"Realization reflects sudden understanding or clarity of thought.",
 "relief":"Relief reflects emotional release after stress or worry subsides.",
 "remorse":"Remorse represents guilt or regret over past actions.",
 "sadness":"Sadness reflects emotional pain or loss.",
 "surprise":"Surprise occurs as a reaction to unexpected events.",
 "neutral":"Neutral indicates emotional balance without strong feelings."
}

# ================= 2 REPLIES PER EMOTION =================
REPLIES = {
 "admiration":["That’s truly admirable.","You seem genuinely impressed."],
 "amusement":["That sounds fun 😄","It seems that made you smile."],
 "anger":["I understand why you’re upset.","That sounds really frustrating."],
 "annoyance":["That does sound irritating.","I can see why that bothered you."],
 "approval":["That seems like a good decision.","You sound confident about this."],
 "caring":["That shows a lot of kindness.","You clearly care deeply."],
 "confusion":["It’s okay to feel unsure.","Things may become clearer soon."],
 "curiosity":["That curiosity is a good thing.","Wanting to know more is natural."],
 "desire":["That sounds important to you.","You seem strongly motivated."],
 "disappointment":["I’m sorry it didn’t go as hoped.","That kind of let‑down can hurt."],
 "disapproval":["It seems this doesn’t sit right with you.","You’re standing by your values."],
 "disgust":["That sounds unpleasant.","I understand that reaction."],
 "embarrassment":["That must have felt awkward.","These moments happen to everyone."],
 "excitement":["That’s exciting!","You sound really enthusiastic."],
 "fear":["That sounds scary.","You’re not alone in this."],
 "gratitude":["That’s very thoughtful of you.","Your appreciation shows."],
 "grief":["I’m really sorry you’re going through this.","That sounds deeply painful."],
 "joy":["That’s wonderful to hear!","You sound genuinely happy."],
 "love":["That’s really sweet ❤️","You sound deeply connected."],
 "nervousness":["It’s okay to feel nervous.","That anxiety is understandable."],
 "optimism":["That’s a hopeful outlook.","Your positivity shows."],
 "pride":["You deserve to feel proud.","That’s a real achievement."],
 "realization":["That sounds like an important insight.","Things became clearer for you."],
 "relief":["I’m glad things feel lighter now.","That relief must feel good."],
 "remorse":["It takes courage to admit that.","Wanting to improve really matters."],
 "sadness":["I’m sorry you’re feeling this way.","That sounds really heavy."],
 "surprise":["That must have been unexpected!","Sounds like you didn’t see that coming."],
 "neutral":["I’m listening.","Tell me more when you’re ready."]
}

def pressure_label(conf):
    if conf >= 0.75:
        return "🔴 HIGH"
    elif conf >= 0.45:
        return "🟡 MEDIUM"
    else:
        return "🟢 LOW"

# ================= LOAD MODEL =================
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(BASE, "bert_goemotions")

@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(MODEL, local_files_only=True)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ================= HEADER =================
st.markdown("<div class='neon-title'>MindPulse 🧠</div>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center'>Live Emotional Pressure Detection System</h4>",
            unsafe_allow_html=True)

# ================= LAYOUT =================
left, center, right = st.columns([1.2, 2.2, 1.4])

with center:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    text = st.text_area("Enter the message", height=140,
                        placeholder="Type the message here...")
    analyze = st.button("🚀 Analyze Emotion", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ================= ANALYSIS =================
if analyze and text.strip():
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)[0].tolist()
    idx = int(torch.argmax(torch.tensor(probs)))
    emotion = LABELS[idx]
    conf = probs[idx]

    # LEFT PANEL (CONFIDENCE + PRESSURE)
    with left:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown(f"## {emotion.capitalize()}")
        st.markdown(f"<span class='badge'>Confidence: {round(conf*100,2)}%</span>",
                    unsafe_allow_html=True)
        st.progress(conf)
        st.markdown(f"### Pressure: **{pressure_label(conf)}**")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### 🧠 Emotion Explanation")
        st.write(EMOTION_DESCRIPTION[emotion])
        st.markdown("</div>", unsafe_allow_html=True)

    # CHAT
    with center:
        st.markdown(f"<div class='chat-user'>{text}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='chat-ai'>{random.choice(REPLIES[emotion])}</div>",
            unsafe_allow_html=True
        )

    with right:
        df = pd.DataFrame({"Emotion": LABELS, "Probability": probs})

    # ---- PIE CHART TITLE (WHITE & PROPERLY PLACED) ----
        st.markdown(
        "<h4 style='text-align:center; color:white;'>Emotion Distribution</h4>",
        unsafe_allow_html=True
    )

        pie = px.pie(
        df,
        names="Emotion",
        values="Probability",
        color_discrete_sequence=px.colors.sequential.Plasma,
        height=260
    )
        pie.update_layout(
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        margin=dict(t=10, b=10, l=10, r=10)
    )
        st.plotly_chart(pie, use_container_width=True)

    # ---- BAR GRAPH TITLE (WHITE & CLEAN) ----
        st.markdown(
        "<h4 style='text-align:center; color:white; margin-top:10px;'>Top 5 Emotions</h4>",
        unsafe_allow_html=True
    )

        bar = px.bar(
        df.sort_values("Probability", ascending=False).head(5),
        x="Emotion",
        y="Probability",
        color="Emotion",
        height=240,
        color_discrete_sequence=px.colors.sequential.Plasma
    )
        bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        margin=dict(t=10, b=10, l=10, r=10)
    )
        st.plotly_chart(bar, use_container_width=True)