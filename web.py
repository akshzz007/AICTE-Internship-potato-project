import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# ---------------- MODEL ----------------

@st.cache_resource
def load_model():
    model_path = "trained_plant_disease_model.keras"
    return tf.keras.models.load_model(model_path, compile=False)

model = load_model()

classes = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

disease_info = {
    "Potato___Early_blight": {
        "emoji": "🟤",
        "description":
        "Early blight is a fungal disease that creates brown circular spots on potato leaves.",
        "treatment":
        "✔ Remove infected leaves\n"
        "✔ Spray Mancozeb or Chlorothalonil fungicide\n"
        "✔ Avoid overhead watering"
    },

    "Potato___Late_blight": {
        "emoji": "⚫",
        "description":
        "Late blight is a dangerous disease that spreads rapidly in cool and humid weather.",
        "treatment":
        "✔ Remove infected plants\n"
        "✔ Spray Copper based fungicide\n"
        "✔ Improve air circulation"
    },

    "Potato___healthy": {
        "emoji": "🌱",
        "description":
        "The potato leaf appears healthy and disease free.",
        "treatment":
        "✔ No treatment required\n"
        "✔ Continue proper watering\n"
        "✔ Monitor crop regularly"
    }
}


# ---------------- SIDEBAR ----------------

st.sidebar.title("🌿 Plant Disease Detection")

page = st.sidebar.selectbox(
    "Select Page",
    ["Home", "Disease Recognition"]
)


# ---------------- HOME ----------------

if page == "Home":

    st.title("🌿 Plant Disease Detection System")

    st.markdown("""
This project uses a Deep Learning CNN model to detect potato leaf diseases.

### Features

✅ Detect Potato Diseases

✅ Upload Leaf Image

✅ Shows Confidence

✅ Gives Treatment Recommendation

### Supported Diseases

- Potato Early Blight
- Potato Late Blight
- Healthy Potato Leaf

---
Developed using **TensorFlow + Streamlit**
""")


# ---------------- DISEASE PAGE ----------------

if page == "Disease Recognition":

    st.title("🔍 Disease Recognition")

    uploaded = st.file_uploader(
        "Upload Potato Leaf Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded is not None:

        image = Image.open(uploaded)

        st.image(image,
                 caption="Uploaded Image",
                 use_container_width=True)

        if st.button("🔮 Predict"):

            img = image.resize((128,128))

            img = np.array(img)/255.0

            img = np.expand_dims(img,axis=0)

            prediction = model.predict(img)

            index = np.argmax(prediction)

            confidence = float(np.max(prediction))*100

            disease = classes[index]

            info = disease_info[disease]

            st.success(
                f"{info['emoji']} Prediction : {disease.replace('___',' ')}"
            )

            st.info(
                f"🎯 Confidence : {confidence:.2f}%"
            )

            st.subheader("📖 Description")

            st.write(info["description"])

            st.subheader("💊 Recommendation")

            st.success(info["treatment"])