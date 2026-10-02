import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="SentimentMind AI",
    page_icon="🧠",
    layout="centered"
)


# =========================================
# LOAD CSS
# =========================================

def load_css():
    with open("style.css", "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )


load_css()


# =========================================
# MODEL PATH
# =========================================

MODEL_PATH = "."


# =========================================
# LOAD MODEL
# =========================================

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


tokenizer, model, device = load_model()


# =========================================
# HEADER
# =========================================

st.title("🧠 SentimentMind AI")

st.write(
    "AI-powered Sentiment Analysis"
)

st.divider()


# =========================================
# INPUT
# =========================================

st.subheader("Enter your text")

text = st.text_area(
    "Text",
    placeholder="Example: I really enjoyed this product!",
    height=160,
    label_visibility="collapsed"
)


# =========================================
# ANALYZE
# =========================================

analyze = st.button(
    "🔍 Analyze Sentiment",
    use_container_width=True
)


if analyze:

    if not text.strip():

        st.warning(
            "Please enter some text first."
        )

    else:

        with st.spinner(
            "Analyzing sentiment..."
        ):

            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128
            )

            # Move tensors to device
            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            # Model prediction
            with torch.no_grad():

                outputs = model(**inputs)

            # Probabilities
            probabilities = torch.softmax(
                outputs.logits,
                dim=-1
            )[0]

            # Predicted class
            predicted_class = torch.argmax(
                probabilities
            ).item()


        # =========================================
        # LABELS
        # =========================================

        labels = {
            0: "Negative",
            1: "Neutral",
            2: "Positive"
        }

        prediction = labels[predicted_class]


        # =========================================
        # SCORES
        # =========================================

        negative = probabilities[0].item()
        neutral = probabilities[1].item()
        positive = probabilities[2].item()


        # =========================================
        # RESULT
        # =========================================

        st.divider()

        st.subheader("Predicted Sentiment")

        if prediction == "Negative":

            st.error(
                f"🔴 {prediction}"
            )

        elif prediction == "Neutral":

            st.warning(
                f"🟡 {prediction}"
            )

        else:

            st.success(
                f"🟢 {prediction}"
            )


        # =========================================
        # CONFIDENCE
        # =========================================

        st.subheader("Confidence Scores")

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "🔴 Negative",
                f"{negative * 100:.2f}%"
            )

            st.progress(
                negative
            )


        with col2:

            st.metric(
                "🟡 Neutral",
                f"{neutral * 100:.2f}%"
            )

            st.progress(
                neutral
            )


        with col3:

            st.metric(
                "🟢 Positive",
                f"{positive * 100:.2f}%"
            )

            st.progress(
                positive
            )


        # =========================================
        # INFO
        # =========================================

        st.divider()

        st.info(
            "SentimentMind AI analyzes your text and "
            "classifies it as Negative, Neutral, or Positive."
        )


# =========================================
# FOOTER
# =========================================

st.divider()

st.caption(
    "Made by Muhammad Saim"
)