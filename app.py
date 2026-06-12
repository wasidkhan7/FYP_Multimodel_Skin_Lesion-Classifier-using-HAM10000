
# our app should:

# ✔ upload image
# ✔ input metadata
# ✔ predict lesion
# ✔ show confidence
# ✔ display image
# ✔ show medical disclaimer

# ==================================================

# ==================================================
# Streamlit Skin Lesion Detection App
# ==================================================

import streamlit as st
from PIL import Image
import os 
import requests


# imports for Grad-CAM visualization    
from gradcam_utils import generate_gradcam
from model import (load_fusion_model,device)
from utils.preprocessing import (preprocess_metadata)


# ==================================================
# Load Fusion Model
# ==================================================

API_URL = os.getenv("SKIN_LESION_API_URL", "http://127.0.0.1:8000/predict")
API_TIMEOUT_SECONDS = 30

fusion_model = load_fusion_model(
    "models/fusion_model.pth",

    device
)

# ==================================================
# Page Configuration
# ==================================================

st.set_page_config(

    page_title="Skin Lesion Detection",

    layout="centered"
)



# ==================================================
# Title
# ==================================================

st.title("Multimodal Skin Lesion Detection")

st.write(
    "Upload a dermoscopic image and provide metadata for prediction."
)


# ==================================================
# Image Upload
# ==================================================

uploaded_file = st.file_uploader(

    "Upload Skin Lesion Image",

    type=["jpg", "jpeg", "png"]
)

# ==================================================
# Metadata Inputs
# ==================================================

age = st.number_input(

    "Age",

    min_value=0,

    max_value=120,

    value=45
)

sex = st.selectbox(

    "Sex",

    ["male", "female"]
)

localization = st.selectbox(

    "Localization",

    [
        'scalp',
        'ear',
        'face',
        'back',
        'trunk',
        'chest',
        'upper extremity',
        'abdomen',
        'unknown',
        'lower extremity',
        'genital',
        'neck',
        'hand',
        'foot',
        'acral'
    ]
)



# ==================================================
# Predict Button
# ==================================================

if st.button("Predict"):
    
    if uploaded_file is not None:

        image = Image.open(uploaded_file)


        # ==================================================
        # Save Uploaded Image
        # ==================================================
    
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)

        image_path = os.path.join(  upload_dir,uploaded_file.name)

        with open(image_path, "wb") as f:

            f.write(uploaded_file.getbuffer())

        # ============
        # Prediction
      

        # ==================================================
        # Send Request To FastAPI
        # ==================================================

        form_data = {

        "age": age,

        "sex": sex,

        "localization": localization}

        with open(image_path, "rb") as img_file:

            files = {"image": img_file}

            # sends and recieves the data from the API and stores it in response variable
            try:
                response = requests.post(
                API_URL,
                files=files,
                data=form_data,
                timeout=API_TIMEOUT_SECONDS)
            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not connect to the prediction API at "
                    f"{API_URL}.\n\n"
                    "Start it in another terminal with:\n\n"
                    "`uvicorn fast_api:app --host 127.0.0.1 --port 8000`")
                st.stop()
            except requests.exceptions.Timeout:
                st.error(
                    "The prediction API took too long to respond. "
                    "Please try again after the model finishes loading.")
                st.stop()
            except requests.exceptions.RequestException as exc:
                st.error(f"Prediction API request failed: {exc}")
                st.stop()
      

        if response.status_code != 200:
            st.error(
                f"API Error: {response.text}")
            st.stop()

        # Convert JSON response
        try:
            result = response.json()
        except requests.exceptions.JSONDecodeError:
            st.error("Prediction API returned an invalid response.")
            st.stop()


       # ==================================================
        # Prepare Metadata For Grad-CAM
        # ==================================================

        metadata = preprocess_metadata(
        age,
        sex,
        localization
        )
        metadata = metadata.to(device)

        # ==================================================
        # Generate Grad-CAM
        # ==================================================

        gradcam_image = generate_gradcam(
        fusion_model=fusion_model,
        image_path=image_path,
        metadata=metadata,
        device=device)


        # ==================================================
        # Cancerous vs Non-Cancerous
        # ==================================================

        predicted_class = result["prediction"]

        # Cancerous lesion classes
        cancer_classes = [

        "Actinic Keratoses (akiec)",

        "Basal Cell Carcinoma (bcc)",

        "Melanoma (mel)"]

        # Check category
        if predicted_class in cancer_classes:

            lesion_category = "Cancerous Lesion"

        else:

            lesion_category = "Non-Cancerous Lesion"            

        # ==================================================
        # Display Results
        # ==================================================

        # ==================================================
        # Confidence Threshold
        # ==================================================

        CONFIDENCE_THRESHOLD = 55

        if result['confidence'] < CONFIDENCE_THRESHOLD:

            st.error(
            "No clear skin lesion detected.\n\n"

            "Please upload a proper dermoscopic "
            "skin lesion image.")

        else:

            st.success(

                f"Prediction: {result['prediction']}")

            st.subheader(
            f"Category: {lesion_category}")

            st.info(
            f"Confidence: {result['confidence']}%")


            # ==================================================
            # Display Grad-CAM
            # ==================================================

            # =====================================
            # Resize Images For Display
            # =====================================

            display_size = (350, 350)

            original_display = image.resize(display_size)

            gradcam_display = Image.fromarray(
            gradcam_image).resize(display_size)

            # ==================================================
            # Side-by-Side Visualization
            # ==================================================
            st.subheader("Grad-CAM Visualization")

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                original_display,
                caption="Original Image",
                use_container_width=True)

            with col2:
                st.image(
                gradcam_display,
                caption="Grad-CAM Heatmap",
                use_container_width=True)

            st.warning(
                "Predictions are for research purposes only "
                "and should not be considered medical diagnosis.")

    else:
        st.error("Please upload an image.")

