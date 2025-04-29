import streamlit as st
import pandas as pd
import os
import base64

st.title("📜 Prediction History")
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
set_background('app/assets/Prediction_History.jpg')  # Corrected path
# Display Prediction History
if os.path.exists('history/prediction_logs.csv'):
    df = pd.read_csv('history/prediction_logs.csv')
    st.dataframe(df)

    # Download as CSV
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Prediction History as CSV",
        data=csv,
        file_name='prediction_history.csv',
        mime='text/csv',
    )
else:
    st.warning("No prediction history available yet.")
