import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Set up paths
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "trained_model/plant_disease_prediction_model.h5")
class_indices_path = os.path.join(working_dir, "class_indices.json")

# Load the pre-trained model and class indices
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(class_indices_path))

# Define fertilizers for diseases
disease_fertilizers = {
    "Apple___Apple_scab": "Apply fungicides containing captan or myclobutanil.",
    "Apple___Black_rot": "Use fungicides with active ingredients like thiophanate-methyl or captan.",
    "Apple___Cedar_apple_rust": "Apply fungicides such as myclobutanil or propiconazole during the early stages of infection.",
    "Cherry_(including_sour)___Powdery_mildew": "Use fungicides containing sulfur or myclobutanil.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Apply fungicides like azoxystrobin or pyraclostrobin.",
    "Corn_(maize)___Common_rust_": "Use fungicides with active ingredients such as mancozeb or chlorothalonil.",
    "Corn_(maize)___Northern_Leaf_Blight": "Apply fungicides containing azoxystrobin or propiconazole.",
    "Grape___Black_rot": "Use fungicides like myclobutanil or captan.",
    "Grape___Esca_(Black_Measles)": "No effective chemical treatment; focus on preventive measures such as proper pruning and avoiding trunk injuries.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Apply fungicides containing mancozeb or copper-based compounds.",
    "Orange___Haunglongbing_(Citrus_greening)": "No cure available; manage by removing infected trees and controlling the Asian citrus psyllid vector.",
    "Peach___Bacterial_spot": "Use copper-based bactericides; however, effectiveness may be limited.",
    "Pepper,_bell___Bacterial_spot": "Apply copper-based bactericides; their efficacy can vary.",
    "Potato___Early_blight": "Use fungicides containing chlorothalonil or mancozeb.",
    "Potato___Late_blight": "Apply fungicides with active ingredients like mefenoxam or chlorothalonil.",
    "Squash___Powdery_mildew": "Use fungicides containing sulfur or potassium bicarbonate.",
    "Strawberry___Leaf_scorch": "Apply fungicides such as myclobutanil or captan.",
    "Tomato___Bacterial_spot": "Use copper-based bactericides; their effectiveness may be limited.",
    "Tomato___Early_blight": "Apply fungicides containing chlorothalonil or mancozeb.",
    "Tomato___Late_blight": "Use fungicides with active ingredients like mefenoxam or chlorothalonil.",
    "Tomato___Leaf_Mold": "Apply fungicides such as chlorothalonil or copper-based compounds.",
    "Tomato___Septoria_leaf_spot": "Use fungicides containing chlorothalonil or mancozeb.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Apply insecticidal soaps or horticultural oils.",
    "Tomato___Target_Spot": "Use fungicides with active ingredients like azoxystrobin or chlorothalonil.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Manage by controlling whitefly populations using appropriate insecticides.",
    "Tomato___Tomato_mosaic_virus": "No chemical treatment available; focus on preventive measures such as using virus-free seeds and practicing good sanitation.",
    "Healthy": "No disease detected. Maintain regular plant care and monitoring."
}
# Function to preprocess image
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Function to predict the class of an image
def predict_image_class(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    confidence = np.max(predictions) * 100  # Confidence percentage
    fertilizer = disease_fertilizers.get(predicted_class_name, "Consult an expert for the best treatment.")
    return predicted_class_name, confidence, fertilizer

# Streamlit UI setup
st.set_page_config(page_title="Plant Disease Classifier", layout="wide")

# Add custom background image
import base64

# def set_background(image_file):
#     with open(image_file, "rb") as image:
#         encoded_string = base64.b64encode(image.read()).decode()
#     bg_image_style = f"""
#     <style>
#     .stApp {{
#         background-image: url("data:image/jpeg;base64,{encoded_string}");
#         background-size: cover;
#         background-position: center;
#         background-repeat: no-repeat;
#     }}
#     </style>
#     """
#     st.markdown(bg_image_style, unsafe_allow_html=True)


def set_background(image_file):
    image_path = os.path.join(os.path.dirname(__file__), image_file)  # Get full path
    if not os.path.exists(image_path):  # Debugging: Check if file exists
        raise FileNotFoundError(f"Background image not found at: {image_path}")
    
    with open(image_path, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()

    bg_image_style = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(bg_image_style, unsafe_allow_html=True)

# Call this function at the beginning of your Streamlit app
set_background("background_image.jpg")  # Replace with your local image filename

st.markdown(""" <h1 style='text-align: center; color: green;'>🌿 Plant Disease Classifier 🍃</h1> """, unsafe_allow_html=True)

st.markdown(""" <p style='text-align: center;'><b>Upload an image of a plant leaf and our AI will detect any disease present and will suggest certain fertilizers</b>.</p> """, unsafe_allow_html=True)

# Styled Upload image
dropbox_style = """
<style>
    div[data-testid="stFileUploader"] {
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        font-size: 18px;
        color: #e65100;
        font-weight: bold;
    }
</style>
"""
st.markdown(dropbox_style, unsafe_allow_html=True)

st.markdown(dropbox_style, unsafe_allow_html=True)
uploaded_image = st.file_uploader("**Upload an image of a plant leaf..**", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.markdown(""" <h3 style='text-align: center;'>Processing Image...</h3> """, unsafe_allow_html=True)
        with st.spinner("Classifying the image..."):
            prediction, confidence, fertilizer = predict_image_class(model, image, class_indices)
        st.success(f"**Prediction:** {prediction}")
        st.info(f"**Confidence**: {confidence:.2f}%")
        st.warning(f"**Recommended Fertilizer**: {fertilizer}")

    st.markdown(""" <p style='text-align: center;'>🔄 Upload another image to classify again.</p> """, unsafe_allow_html=True)
