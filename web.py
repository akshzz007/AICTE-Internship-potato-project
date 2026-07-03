import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# ---------------- MODEL ---------------- #

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "trained_plant_disease_model.keras",
        compile=False
    )

model = load_model()

classes = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

disease_info = {

    "Potato___Early_blight":{

        "emoji":"🟤",

        "description":
        "Early Blight is a fungal disease caused by Alternaria solani. Brown spots with concentric rings appear on leaves and reduce crop yield.",

        "treatment":
        """
✅ Remove infected leaves

✅ Spray Mancozeb / Chlorothalonil fungicide

✅ Avoid overhead irrigation

✅ Maintain crop rotation
        """
    },

    "Potato___Late_blight":{

        "emoji":"⚫",

        "description":
        "Late Blight is caused by Phytophthora infestans. It spreads rapidly during cool and humid weather and can destroy crops quickly.",

        "treatment":
        """
✅ Remove infected plants immediately

✅ Spray Copper based fungicide

✅ Improve air circulation

✅ Avoid excess moisture
        """
    },

    "Potato___healthy":{

        "emoji":"🌱",

        "description":
        "The uploaded potato leaf appears healthy with no visible disease symptoms.",

        "treatment":
        """
✅ No treatment required

✅ Continue regular watering

✅ Use balanced fertilizer

✅ Monitor plants regularly
        """
    }
}

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("🌿 Plant Disease Detection")

page = st.sidebar.selectbox(
    "Select Page",
    ["🏠 Home","🔍 Disease Recognition"]
)

# ---------------- HOME ---------------- #

if page=="🏠 Home":

    st.title("🌿 Plant Disease Detection System")

    st.image(
        "diseases.png",
        use_container_width=True
    )

    st.markdown("""
## Welcome 👋

This application uses a **Deep Learning CNN Model** to identify potato leaf diseases.

### ✨ Features

- ✅ Detect Potato Diseases
- ✅ Upload Potato Leaf Images
- ✅ Prediction Confidence
- ✅ Disease Description
- ✅ Treatment Recommendation

---

### 🦠 Detects

- 🟤 Potato Early Blight
- ⚫ Potato Late Blight
- 🌱 Healthy Potato Leaf

---

Developed using **TensorFlow + Streamlit**
""")

# ---------------- PREDICTION ---------------- #

if page=="🔍 Disease Recognition":

    st.title("🔍 Disease Recognition")

    uploaded=st.file_uploader(
        "Upload Potato Leaf Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded:

        image=Image.open(uploaded)

        col1,col2=st.columns([1,1])

        with col1:
            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        with col2:

            st.write("### Ready for Prediction")

            if st.button("🚀 Predict Disease"):

                img=image.resize((128,128))

                img=np.array(img)/255.0

                img=np.expand_dims(img,axis=0)

                prediction=model.predict(img)

                index=np.argmax(prediction)

                confidence=np.max(prediction)*100

                disease=classes[index]

                info=disease_info[disease]

                st.success(
                    f"{info['emoji']} **Prediction : {disease.replace('___',' ')}**"
                )

                st.progress(int(confidence))

                st.info(
                    f"🎯 Confidence : **{confidence:.2f}%**"
                )

                st.subheader("📖 Disease Description")

                st.write(info["description"])

                st.subheader("💊 Recommended Treatment")

                st.success(info["treatment"])