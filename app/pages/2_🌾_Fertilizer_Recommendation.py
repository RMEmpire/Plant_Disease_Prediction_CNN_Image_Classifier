import streamlit as st
import base64

st.markdown(
    """
    <h1 style='text-align: center; color: black;'>
        🌱 Fertilizer Recommendation System🧪
    </h1>
    """,
    unsafe_allow_html=True
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
set_background('assets\Fertilizer img2.jpg')  # Corrected path
fertilizer_mapping = {
    "Apple___Apple_scab": "Use Mancozeb fungicide + Urea spray",
    "Apple___Black_rot": "Use Captan or Ziram + Nitrogen fertilizers",
    "Apple___Cedar_apple_rust": "Use Fungicide (Myclobutanil) + Balanced NPK",
    "Apple___healthy": "Maintain balanced NPK fertilizer monthly",

    "Blueberry___healthy": "Use Acidic fertilizer (Ammonium sulfate)",

    "Cherry_(including_sour)___Powdery_mildew": "Apply Sulfur fungicide + Organic compost",
    "Cherry_(including_sour)___healthy": "Use Balanced NPK (10-10-10)",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Nitrogen supplement + Fungicide",
    "Corn_(maize)___Common_rust_": "Fungicide + Potassium boost",
    "Corn_(maize)___Northern_Leaf_Blight": "Use Nitrogen rich fertilizers + Fungicide",
    "Corn_(maize)___healthy": "Use Urea + Micronutrients spray",

    "Grape___Black_rot": "Use Sulfur based fungicide + Balanced NPK",
    "Grape___Esca_(Black_Measles)": "Use Fungicide + Potassium-Rich fertilizer",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Use Captan + Increase Phosphorus supply",
    "Grape___healthy": "Balanced NPK every 2 months",

    "Orange___Haunglongbing_(Citrus_greening)": "Apply Micronutrients + Zn, Mn foliar spray",

    "Peach___Bacterial_spot": "Use Copper fungicide + Potassium nitrate",
    "Peach___healthy": "Use Balanced NPK fertilizer 14-14-14",

    "Pepper,_bell___Bacterial_spot": "Use Copper spray + Potash fertilizer",
    "Pepper,_bell___healthy": "Use Compost + Balanced organic fertilizers",

    "Potato___Early_blight": "Apply Mancozeb + Nitrogen fertilizer",
    "Potato___Late_blight": "Apply Metalaxyl fungicide + DAP fertilizer",
    "Potato___healthy": "Apply DAP and Potash fertilizers",

    "Raspberry___healthy": "Use 10-10-10 fertilizer during growing season",

    "Soybean___healthy": "Use Inoculated Rhizobium + Potash fertilizers",

    "Squash___Powdery_mildew": "Use Sulfur fungicide + Compost mulch",

    "Strawberry___Leaf_scorch": "Use Phosphorus-rich fertilizer + Fungicide",
    "Strawberry___healthy": "Apply Straw mulch + NPK (10-10-10)",

    "Tomato___Bacterial_spot": "Use Copper fungicide + Potash rich fertilizer",
    "Tomato___Early_blight": "Use Mancozeb fungicide + NPK 19:19:19",
    "Tomato___Late_blight": "Use Metalaxyl fungicide + Balanced fertilizer",
    "Tomato___Leaf_Mold": "Apply Chlorothalonil + Boost Nitrogen",
    "Tomato___Septoria_leaf_spot": "Use Chlorothalonil spray + NPK fertilizers",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Use Neem Oil Spray + Organic fertilizer",
    "Tomato___Target_Spot": "Apply Fungicide + Balanced nutrition",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Use Imidacloprid + Potassium supplement",
    "Tomato___Tomato_mosaic_virus": "Remove infected plants + Apply Phosphorus fertilizers",
    "Tomato___healthy": "Apply Compost + Balanced NPK (10-10-10)"
}
# 🔥 Check if a disease was already predicted
auto_detected_disease = st.session_state.get('predicted_disease', None)

if auto_detected_disease:
    st.success(f"**Auto-detected Disease:** `{auto_detected_disease}`")
    fertilizer = fertilizer_mapping.get(auto_detected_disease, "No fertilizer recommendation found.")
    st.info(f"**Suggested Fertilizer:** {fertilizer}")
else:
    st.warning("⚠️ No auto-predicted disease found. Please manually enter.")

st.divider()

st.subheader("🔎 OR Manually Enter Disease Name:")

disease_input = st.text_input("Enter Disease Name (Example: Tomato___Late_blight)")

if st.button("Recommend Fertilizer"):
    recommendation = fertilizer_mapping.get(disease_input,
        "⚠️ No recommendation found. Please use general fungicide and NPK 20:20:20.")
    st.success(f"**🌿 Fertilizer Recommendation:** {recommendation}")