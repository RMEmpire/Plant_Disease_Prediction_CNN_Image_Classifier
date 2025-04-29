# import streamlit as st
# import tensorflow as tf
# from keras.utils import load_img, img_to_array
# import numpy as np
# import json
# import pandas as pd
# import os
# from datetime import datetime
# import base64

# st.title("🍂🦠 Predict Plant Disease🔍")
# def get_base64_of_bin_file(bin_file):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# def set_background(png_file):
#     bin_str = get_base64_of_bin_file(png_file)
#     page_bg_img = f"""
#     <style>
#     .stApp {{
#       background-image: url("data:image/png;base64,{bin_str}");
#       background-size: cover;
#       background-position: center;
#       background-attachment: fixed;
#     }}
#     </style>
#     """
#     st.markdown(page_bg_img, unsafe_allow_html=True)

# # 🖼️ Set your background
# set_background('assets\disease img.avif')  # Corrected path

# # Load model
# model = tf.keras.models.load_model("models\plant_disease_prediction_model.h5")

# # Load class indices
# with open("models/class_indices.json", "r") as f:
#     class_indices = json.load(f)

# # Reverse mapping
# class_labels = {int(k): v for k, v in class_indices.items()}

# # Upload Image
# uploaded_file = st.file_uploader("Upload Plant Leaf Image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Load and display the image
#     img = load_img(uploaded_file, target_size=(224, 224))
#     st.image(img, caption="Uploaded Leaf Image", use_column_width=False)

#     # Preprocess the image
#     img_array = img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0) / 255.0

#     # Make prediction
#     preds = model.predict(img_array)
#     pred_class = class_labels[np.argmax(preds)]
#     st.session_state['predicted_disease'] =pred_class
#     confidence = np.max(preds) * 100

#     # Display prediction
#     st.success(f"🌿 **Predicted Disease:** {pred_class}")
#     st.info(f"✅ **Confidence:** {confidence:.2f}%")


#     # Save to Prediction History
#     if not os.path.exists('history/prediction_logs.csv'):
#         df = pd.DataFrame(columns=["Date", "Disease", "Confidence"])
#         df.to_csv('history/prediction_logs.csv', index=False)

#     df = pd.read_csv('history/prediction_logs.csv')
#     new_row = {
#         "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         "Disease": pred_class,
#         "Confidence": f"{confidence:.2f}%"
#     }
#     df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
#     df.to_csv('history/prediction_logs.csv', index=False)


import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array  # Fixed import
import numpy as np
import json
import pandas as pd
import os
from datetime import datetime
import base64
import gdown  # For downloading the model

st.title("🍂🦠 Predict Plant Disease🔍")

# 🖼️ Set your background
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    if os.path.exists(png_file):
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
    else:
        st.warning("⚠️ Background image not found.")

set_background('assets/disease_img.avif')  # Use correct relative path

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Download model from Google Drive if not present
model_path = "models/plant_disease_prediction_model.h5"
if not os.path.exists(model_path):
    st.info("Downloading model, please wait...")
    model_url = "https://drive.google.com/uc?id=1AGgEJj1hF5Sjiu_fUvFn6fwmhf6qbIZl"
    gdown.download(model_url, model_path, quiet=False)

# Load the model
model = tf.keras.models.load_model(model_path)

# Load class indices
with open("models/class_indices.json", "r") as f:
    class_indices = json.load(f)

# Reverse mapping
class_labels = {int(k): v for k, v in class_indices.items()}

# Upload Image
uploaded_file = st.file_uploader("Upload Plant Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption="Uploaded Leaf Image", use_column_width=False)

    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    preds = model.predict(img_array)
    pred_class = class_labels[np.argmax(preds)]
    st.session_state['predicted_disease'] = pred_class
    confidence = np.max(preds) * 100

    st.success(f"🌿 **Predicted Disease:** {pred_class}")
    st.info(f"✅ **Confidence:** {confidence:.2f}%")

    # Save to Prediction History
    os.makedirs('history', exist_ok=True)
    if not os.path.exists('history/prediction_logs.csv'):
        df = pd.DataFrame(columns=["Date", "Disease", "Confidence"])
        df.to_csv('history/prediction_logs.csv', index=False)

    df = pd.read_csv('history/prediction_logs.csv')
    new_row = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Disease": pred_class,
        "Confidence": f"{confidence:.2f}%"
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv('history/prediction_logs.csv', index=False)
