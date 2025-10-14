# app.py
import os
import streamlit as st
from io import BytesIO
from PIL import Image
import requests
import base64
from dotenv import load_dotenv


load_dotenv()

st.set_page_config(page_title="AI Blog Assistant", layout="wide")
st.title("📝INKWELL:Your AI Blogging Assistant")

#  Helper: Try to import Google GenAI (Gemini) client 
gemini_available = False
gemini_client = None
try:
    import google.generativeai as genai
    # --- FIX: Use os.getenv() to get the key ---
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        genai.configure(api_key=google_api_key)
        gemini_available = True
        gemini_client = genai
    else:
        st.info("Gemini API key not found. The app will fall back to OpenAI text model.")
        
except ImportError:
    st.info("Gemini client not found (pip install google-generativeai). The app will fall back to OpenAI text model.")

# Helper: OpenAI client for images & fallback
openai_available = False
try:
    import openai
    # --- FIX: Use os.getenv() to get the key ---
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        openai.api_key = openai_api_key
        openai_available = True
    else:
        st.warning("OPENAI_API_KEY not set. Image generation will not work without it.")

except ImportError:
    st.warning("OpenAI client not installed (pip install openai). Image generation will not work without it.")


#  Functions 
def generate_blog_with_gemini(prompt: str, max_tokens: int = 800) -> str:
    if not gemini_available:
        raise RuntimeError("Gemini client not available.")
    try:
        model = genai.GenerativeModel('models/gemini-pro')

        response = model.generate_content(
            f"Write a long-form blog post about:\n\n{prompt}\n\nInclude an intro, sections with headings, and a conclusion. Make it friendly and informative.",
            generation_config=genai.types.GenerationConfig(max_output_tokens=max_tokens)
        )
        return response.text
    except Exception as ex:
        st.error(f"Error generating with Gemini: {ex}")
        raise

def generate_blog_with_openai(prompt: str, max_tokens: int = 800) -> str:
    if not openai_available:
        raise RuntimeError("OpenAI client not available.")
    completion_prompt = (
        f"Write a long-form blog post about:\n\n{prompt}\n\n"
        "Include an intro, sections with headings, bullet points where useful, and a short conclusion.\n"
        "Make it friendly, readable, and helpful."
    )
   
    resp = openai.Completion.create(
        engine="text-davinci-003", 
        prompt=completion_prompt,
        max_tokens=max_tokens,
        temperature=0.7,
        n=1,
        top_p=1
    )
    return resp.choices[0].text.strip()

def generate_image_openai(prompt: str, n: int = 1, size: str = "1024x1024"):
    if not openai_available:
        raise RuntimeError("OpenAI client not available.")
    images = []
    try:
        
        result = openai.Image.create(prompt=prompt, n=n, size=size, response_format="b64_json")
        for item in result['data']:
            img_data = base64.b64decode(item["b64_json"])
            img = Image.open(BytesIO(img_data))
            images.append(img)
        return images
    except Exception as e:
        st.error(f"Image generation error: {e}")
        return []

#  Streamlit UI 
with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox("Content model", options=["Gemini (if available)", "OpenAI (fallback)"])
    max_tokens = st.slider("Max tokens for blog", 200, 2000, 800, step=50)
    num_images = st.number_input("Number of images", min_value=0, max_value=4, value=1)
    image_style = st.text_input("Image prompt style (optional)", value="clean modern illustration")

st.markdown("### Input")
topic = st.text_input("Blog topic / angle", placeholder="e.g., 'How AI will change personal finance in 2025'")
prompt_template = st.text_area("Extra prompt instructions (optional)", height=120, value="Target audience: beginners. Tone: friendly. Include examples and 3 headings.")

if st.button("Generate blog + images"):
    if not topic.strip():
        st.warning("Please enter a blog topic.")
    else:
        user_prompt = f"{topic}\n\n{prompt_template}"
        blog_text = None
        with st.spinner("Generating blog..."):
            try:
                if model_choice.startswith("Gemini") and gemini_available:
                    blog_text = generate_blog_with_gemini(user_prompt, max_tokens=max_tokens)
                elif openai_available:
                    blog_text = generate_blog_with_openai(user_prompt, max_tokens=max_tokens)
                else:
                    st.error("No content generation models are available. Please check your API keys and installations.")
            except Exception as e:
                st.error(f"Content generation failed: {e}")

        if blog_text:
            st.subheader("Generated Blog")
            st.markdown(blog_text)
            st.download_button("Download blog as .txt", blog_text, file_name="generated_blog.txt", mime="text/plain")

            if num_images > 0:
                img_prompt = f"{topic}. {image_style} -- high quality, suitable as a blog header image."
                with st.spinner("Generating image(s)..."):
                    images = generate_image_openai(img_prompt, n=num_images)
                    if images:
                        st.subheader("Generated Image(s)")
                        cols = st.columns(min(3, len(images)))
                        for i, img in enumerate(images):
                            with cols[i % len(cols)]:
                                st.image(img, use_column_width=True)
                                buf = BytesIO()
                                img.save(buf, format="PNG")
                                byte_im = buf.getvalue()
                                st.download_button(f"Download image {i+1}", data=byte_im, file_name=f"image_{i+1}.png", mime="image/png")
