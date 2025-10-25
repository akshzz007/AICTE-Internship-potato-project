import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

#  Function for Model Prediction (Fixed)
def model_prediction(test_image):
    # Correcting model path
    model_path = os.path.join(os.getcwd(), "trained_plant_disease_model.keras")
    
    # Debugging Step: Print Model Path
    st.write(f"🔍 Checking Model Path: `{model_path}`")

    if not os.path.exists(model_path):
        st.error("⚠️ Model file not found! Please upload the correct model.")
        return None
    
    model = tf.keras.models.load_model(model_path, compile=False)  # ✅ Load model with compile=False

    # Convert UploadedFile to Image
    image = Image.open(test_image)
    image = image.resize((128, 128))
    input_arr = np.array(image) / 255.0  # Normalize the image
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert to batch format

    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# ✅ Sidebar for Navigation
st.sidebar.title("🌿 Plant Disease Detection System")
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Disease Recognition'])

# ✅ Display Image (Fixed Warning)
try:
    img = Image.open('diseases.png')
    st.image(img, use_container_width=True)  # ✅ Fixed the deprecated parameter
except FileNotFoundError:
    st.warning("⚠️ Warning: diseases.png not found!")

# ✅ Home Page
if app_mode == 'Home':
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", 
                unsafe_allow_html=True)

# ✅ Disease Recognition Page
elif app_mode == 'Disease Recognition':
    st.header('🔍 Plant Disease Detection')

    test_image = st.file_uploader('📤 Upload an Image:', type=['jpg', 'png', 'jpeg'])

    if test_image:
        st.image(test_image, use_container_width=True, caption="Uploaded Image")  # ✅ Fixed the deprecated parameter

        if st.button('🔮 Predict'):
            st.snow()
            st.write('🔍 Analyzing the Image...')

            result_index = model_prediction(test_image)

            if result_index is not None:
                class_name = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___Healthy']
                st.success(f'🌱 Model Prediction: **{class_name[result_index]}**')




