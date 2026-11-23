"""Predict Amazon review sentiment with a trained Transformer."""


from pathlib import Path

import streamlit as st
import torch

from train import SentimentTransformer, encode


MODEL_PATH = Path(__file__).parent / "sentiment_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model(model_path):
    """Load the saved Transformer weights and matching vocabulary."""
    try:
        saved_data = torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=False,
        )
    except TypeError:  # Compatibility with older PyTorch versions.
        saved_data = torch.load(model_path, map_location=DEVICE)

    model = SentimentTransformer(
        len(saved_data["vocab"]),
        embedding_dim=saved_data["embedding_dim"],
        hidden_size=saved_data["hidden_size"],
        num_heads=saved_data.get("num_heads", 4),
        num_layers=saved_data.get("num_layers", 2),
    ).to(DEVICE)

    model.load_state_dict(saved_data["model_state_dict"])
    model.eval()

    return model, saved_data["vocab"], saved_data["max_length"]


st.set_page_config(
    page_title="Amazon Review Transformer Sentiment",
    page_icon="🛍️",
)

st.title("🛍️ Amazon Review Sentiment (Transformer)")
st.write(
    "Enter a product review to classify its sentiment with a trained Transformer."
)

if not MODEL_PATH.is_file():
    st.error("Trained Transformer model not found. Run train.py first.")
    st.stop()

try:
    model, vocab, max_length = load_model(str(MODEL_PATH))
except Exception as error:
    st.error(f"Could not load the trained Transformer model: {error}")
    st.stop()

review = st.text_area(
    "Review text",
    placeholder="Type or paste a product review here...",
    height=160,
)

if st.button("Predict sentiment", type="primary"):
    if not review.strip():
        st.warning("Enter a review first.")
    else:
        review_ids = encode(review.strip(), vocab, max_length)
        review_ids = review_ids.unsqueeze(0).to(DEVICE)
        lengths = torch.tensor([review_ids.shape[1]], dtype=torch.long)

        with torch.no_grad():
            probabilities = torch.softmax(
                model(review_ids, lengths),
                dim=1,
            )[0].cpu()

        prediction = int(probabilities.argmax())

        if prediction == 1:
            st.success("Positive review")
        else:
            st.error("Negative review")

        st.write(f"Negative: {probabilities[0].item():.1%}")
        st.write(f"Positive: {probabilities[1].item():.1%}")