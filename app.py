"""
Text-to-Image Generator
Generates images from text prompts using Hugging Face's Inference API
(FLUX.1-schnell / Stable Diffusion XL). Needs a FREE Hugging Face token.
"""
import os
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
st.set_page_config(page_title="Text to Image", page_icon="🎨", layout="centered")

st.markdown("""
<style>
/* Main background */
.stApp {
    background-color: #ffffff;
    color: #1a1a1a;
}

/* Main title */
h1 {
    color: #15803d !important;
    font-weight: 700;
}

/* Subheadings and labels */
h2, h3, label, .stMarkdown {
    color: #166534 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #f0fdf4;
    border-right: 2px solid #22c55e;
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label {
    color: #166534 !important;
}

/* Text input and text area */
.stTextInput input,
.stTextArea textarea {
    background-color: #ffffff !important;
    color: #1a1a1a !important;
    border: 2px solid #bbf7d0 !important;
    border-radius: 10px !important;
}

/* Focus effect */
.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: #22c55e !important;
    box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.15) !important;
}

/* Select boxes */
.stSelectbox > div > div {
    background-color: #ffffff !important;
    border: 2px solid #bbf7d0 !important;
    border-radius: 10px !important;
}

/* Generate button */
.stButton > button {
    background-color: #16a34a !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
}

.stButton > button:hover {
    background-color: #15803d !important;
    color: #ffffff !important;
}

/* Download button */
.stDownloadButton > button {
    background-color: #ffffff !important;
    color: #15803d !important;
    border: 2px solid #16a34a !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

.stDownloadButton > button:hover {
    background-color: #f0fdf4 !important;
    border-color: #15803d !important;
}

/* Info / warning / error boxes */
.stAlert {
    border-radius: 10px !important;
}

/* Image */
[data-testid="stImage"] {
    border-radius: 12px;
    overflow: hidden;
}

/* Spinner */
.stSpinner > div {
    color: #15803d !important;
}

/* Horizontal lines */
hr {
    border-color: #bbf7d0 !important;
}
</style>
""", unsafe_allow_html=True)

MODELS = {
    "FLUX.1 Schnell (fast, high quality)": "black-forest-labs/FLUX.1-schnell",
    "Stable Diffusion XL": "stabilityai/stable-diffusion-xl-base-1.0",
}

STYLES = {
    "None": "",
    "Photorealistic": ", photorealistic, 8k, highly detailed, sharp focus",
    "Anime": ", anime style, vibrant colors, studio ghibli inspired",
    "Digital Art": ", digital art, trending on artstation, concept art",
    "Oil Painting": ", oil painting, textured brush strokes, classical art",
    "3D Render": ", 3d render, octane render, soft lighting",
}

# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
token = st.sidebar.text_input(
    "Hugging Face Token",
    value=os.getenv("HF_TOKEN", ""),
    type="password",
    help="Get a free token at https://huggingface.co/settings/tokens",
)
model_name = st.sidebar.selectbox("Model", list(MODELS.keys()))
style = st.sidebar.selectbox("Style", list(STYLES.keys()))
size = st.sidebar.selectbox("Size", ["1024x1024", "768x768", "512x512", "1024x768", "768x1024"])
width, height = map(int, size.split("x"))

# ---------------- Main ----------------
st.title("🎨 Text to Image Generator")
st.write("Describe an image and let AI draw it.")

prompt = st.text_area(
    "Your prompt",
    placeholder="A cozy tea stall on a rainy evening in Hyderabad, warm lights, cinematic",
    height=100,
)
negative = st.text_input("Negative prompt (optional)", placeholder="blurry, low quality, distorted")

if st.button("✨ Generate", type="primary"):
    if not token:
        st.error("Please add your Hugging Face token in the sidebar (or in a .env file).")
    elif not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        client = InferenceClient(token=token)
        full_prompt = prompt.strip() + STYLES[style]
        kwargs = {"width": width, "height": height}
        if negative.strip():
            kwargs["negative_prompt"] = negative.strip()

        with st.spinner("Generating image... (10-60 seconds)"):
            try:
                image = client.text_to_image(full_prompt, model=MODELS[model_name], **kwargs)
            except Exception as e:
                st.error(f"Generation failed: {e}")
                st.info(
                    "Common fixes: check your token, wait a minute if the model is loading, "
                    "or try the other model / a smaller size."
                )
                st.stop()

        st.image(image, caption=prompt)
        buf = BytesIO()
        image.save(buf, format="PNG")
        st.download_button("⬇️ Download PNG", buf.getvalue(), "generated.png", "image/png")