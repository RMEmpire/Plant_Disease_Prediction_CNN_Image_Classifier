import streamlit as st
from PIL import Image
import base64

# Set page config
st.set_page_config(
    page_title="Plant Disease Detection App 🌿",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Background Image Setup ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f"""
    <style>
    .stApp {{
      background-image: url("data:image/png;base64,{bin_str}");
      background-size: cover;
      background-position: center;
      background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# 🖼️ Set your background
set_background('assets/background.jpg')  # Corrected path

# --- Centered Title and Subheader in Black Color ---
st.markdown(
    """
    <h1 style='text-align: center; color: black;'>
        🌿 Plant Disease Detection System 🍃
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h3 style='text-align: center; color: black;'>
        👨‍💻 Final Year Project by Yash, Aryan, AbhinayKumar and Sohan 🚀
    </h3>
    """,
    unsafe_allow_html=True
)

# --- Body Content ---
st.markdown("""
<br>
<h5 style='text-align: center; color: black;'>
This project aims to help farmers and researchers by accurately identifying plant diseases 
and recommending fertilizers for better crop management.

It uses a powerful CNN deep learning model, trained on **PlantVillage dataset** 
(over 50,000+ images) to predict diseases from plant leaves in real-time.


🫲 Navigate using the sidebar to explore Prediction, Fertilizer Recommendation, and History!
</h5>""", unsafe_allow_html=True)


# --- Footer ---
st.markdown("""
---
<center>Powered by TensorFlow, Streamlit, CNN and Image Processing 🚀</center>
""", unsafe_allow_html=True)
