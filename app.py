# app.py
import os
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

# --- Load environment variables from .env file ---
load_dotenv()

st.set_page_config(page_title="AI Blogging Assistant", layout="wide")
st.title("📝 AI Blogging Assistant — Free Edition (Groq + Stability AI)")

# ========== Helper: Check if API Keys are available ==========
groq_api_key = os.getenv("GROQ_API_KEY")
stability_api_key = os.getenv("STABILITY_API_KEY")

if not groq_api_key:
    st.info("Groq API key not found. Please add it to your secrets to enable text generation.")

if not stability_api_key:
    st.warning("Stability AI API key not found. Please add it to your secrets to enable image generation.")

# ========== Functions ==========

def generate_blog_with_groq(prompt: str, max_tokens: int = 800) -> str | None:
    """
    Uses the Groq API to generate blog text with the correct Llama model.
    Returns the blog text as a string, or None if it fails.
    """
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY not found in secrets.")

    full_prompt = (
        f"You are an expert blogger. Write a long-form, engaging, and informative blog post about: '{prompt}'. "
        "The blog post should have a clear introduction, several sections with descriptive headings, and a concluding summary. "
        "Ensure the tone is friendly and accessible to a general audience."
    )

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json"
            },
            json={
                # --- THIS IS THE CORRECT, WORKING MODEL NAME FROM YOUR LIST ---
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": full_prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7,
            }
        )
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        error_message = f"Error calling Groq API: {e}"
        if e.response is not None:
            try:
                error_details = e.response.json()
                with st.expander("Click to see API Error Details"):
                    st.json(error_details)
            except ValueError:
                 with st.expander("Click to see API Error Details"):
                    st.text(e.response.text)
        st.error(error_message)
        return None

def generate_image_with_stability(prompt: str, n: int = 1):
    if not stability_api_key:
        raise RuntimeError("STABILITY_API_KEY not found in secrets.")
    
    images = []
    for _ in range(n):
        try:
            response = requests.post(
                "https://api.stability.ai/v2beta/stable-image/generate/core",
                headers={
                    "authorization": f"Bearer {stability_api_key}",
                    "accept": "image/*"
                },
                files={"none": ''},
                data={
                    "prompt": prompt,
                    "output_format": "png",
                    "aspect_ratio": "16:9"
                },
            )
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            images.append(img)
        except requests.exceptions.RequestException as e:
            st.error(f"Error calling Stability AI API: {e}")
            if e.response is not None:
                 with st.expander("Click to see API Error Details"):
                    st.text(e.response.text)
            continue
    return images

# ========== Streamlit UI ==========
with st.sidebar:
    st.header("Settings")
    st.markdown("This app uses free-tier APIs from Groq (for text) and Stability AI (for images).")
    max_tokens = st.slider("Max tokens for blog", 200, 4000, 1000, step=100)
    num_images = st.number_input("Number of images", min_value=0, max_value=2, value=1)
    image_style = st.text_input("Image prompt style (optional)", value="digital art, cinematic lighting")

st.markdown("### Input")
topic = st.text_input("Blog topic / angle", placeholder="e.g., 'The Future of Renewable Energy'")
prompt_template = st.text_area("Extra prompt instructions (optional)", height=120, value="Target audience: general public. Tone: optimistic and informative.")

if st.button("Generate blog + images"):
    if not topic.strip():
        st.warning("Please enter a blog topic.")
    elif not groq_api_key or not stability_api_key:
        st.error("Please make sure you have added both API keys to your Streamlit secrets.")
    else:
        user_prompt = f"{topic}\n\n{prompt_template}"
        blog_text = None
        with st.spinner("Generating blog post with Groq..."):
            blog_text = generate_blog_with_groq(user_prompt, max_tokens=max_tokens)

        if blog_text:
            st.subheader("Generated Blog Post")
            st.markdown(blog_text)
            st.download_button("Download blog as .txt", blog_text, file_name="generated_blog.txt", mime="text/plain")

            if num_images > 0:
                img_prompt = f"{topic}. {image_style}"
                with st.spinner(f"Generating {num_images} image(s) with Stability AI..."):
                    images = generate_image_with_stability(img_prompt, n=num_images)
                    if images:
                        st.subheader("Generated Image(s)")
                        cols = st.columns(min(2, len(images)))
                        for i, img in enumerate(images):
                            with cols[i % len(cols)]:
                                st.image(img, use_column_width=True)
                                buf = BytesIO()
                                img.save(buf, format="PNG")
                                byte_im = buf.getvalue()
                                st.download_button(f"Download image {i+1}", data=byte_im, file_name=f"image_{i+1}.png", mime="image/png")
        else:
            st.error("Content generation failed. Check the error details above.")