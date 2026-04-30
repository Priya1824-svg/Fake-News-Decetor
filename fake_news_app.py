
import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

@st.cache_resource
def load_model():
    with open("fake_news_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    return model, tfidf

model, tfidf = load_model()

st.title("📰 Fake News Detector")
st.markdown("Enter a news headline or article to check if it is **Real or Fake!**")
st.divider()

text_input = st.text_area("Enter news headline or article:", 
                           placeholder="Paste any news headline or article here...", 
                           height=150)

if st.button("Check News 🔍"):
    if text_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        with st.spinner("Analyzing..."):
            input_tfidf = tfidf.transform([text_input])
            prediction = model.predict(input_tfidf)[0]
            probability = model.predict_proba(input_tfidf)[0]

        if prediction == 1:
            st.success(f"✅ This news appears to be **REAL** with {probability[1]*100:.2f}% confidence!")
        else:
            st.error(f"🚨 This news appears to be **FAKE** with {probability[0]*100:.2f}% confidence!")

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🟢 Real Probability", f"{probability[1]*100:.2f}%")
        with col2:
            st.metric("🔴 Fake Probability", f"{probability[0]*100:.2f}%")

st.divider()
st.markdown("**Try these examples:**")
col1, col2 = st.columns(2)
with col1:
    if st.button("📰 Real News Example"):
        st.info("As U.S. budget fight looms, Republicans flip their fiscal script")
with col2:
    if st.button("🚨 Fake News Example"):
        st.info("Donald Trump Sends Out Embarrassing New Year Eve Message This is Disturbing")

st.caption("Built by N. Priya Dharshini | PhD Research Scholar | Kalasalingam University")
