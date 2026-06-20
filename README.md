
# Multimodal Skin Lesion Classification System using HAM10000

## Overview

This project presents a complete end-to-end AI-powered Skin Lesion Classification System developed using Deep Learning, FastAPI, React.js, Explainable AI (Grad-CAM), Large Language Models (GPT-4o-mini), and Docker.

The system analyzes dermoscopic skin lesion images together with patient metadata (age, sex, and lesion localization) to classify skin lesions into seven diagnostic categories from the HAM10000 dataset. To improve robustness and reliability, an additional Skin vs Non-Skin Validator is integrated before classification to reject irrelevant images such as faces, children, vehicles, animals, and other non-medical images.

The application provides visual explanations through Grad-CAM and generates concise AI-assisted educational feedback using GPT-4o-mini. The complete solution includes a React frontend, FastAPI backend, trained deep learning models, and Docker support for deployment.

---

## Features

### Multimodal Skin Lesion Classification

The system combines:

* Dermoscopic skin lesion image
* Patient age
* Patient sex
* Lesion localization

to perform classification.

### Seven HAM10000 Classes

The model predicts one of the following classes:

* Actinic Keratoses (akiec)
* Basal Cell Carcinoma (bcc)
* Benign Keratosis-like Lesions (bkl)
* Dermatofibroma (df)
* Melanoma (mel)
* Melanocytic Nevi (nv)
* Vascular Lesions (vasc)

---

### Skin vs Non-Skin Validation

A dedicated MobileNetV2-based validator is used before classification.

Purpose:

* Accept valid dermoscopic lesion images
* Reject non-medical images
* Prevent false high-confidence predictions on unrelated images

Examples of rejected images:

* Human faces
* Children photographs
* Cars
* Animals
* Flowers
* Random natural images

If an uploaded image is not recognized as a dermoscopic skin lesion image, the system returns:

```json
{
  "prediction": "Unknown",
  "category": "Unknown"
}
```

without performing disease classification.

---

### Out-of-Distribution (OOD) Detection

Confidence-based filtering is implemented to improve reliability.

If model confidence is below the predefined threshold:

* Prediction is rejected
* Grad-CAM is disabled
* GPT explanation is disabled

Response:

```json
{
  "prediction": "Unknown",
  "is_low_confidence": true
}
```

This prevents uncertain predictions from being presented as reliable results.

---

### Explainable AI (Grad-CAM)

Grad-CAM is integrated to visualize regions of the lesion image that influenced the model prediction.

Benefits:

* Increased transparency
* Improved interpretability
* Better understanding of model attention

The frontend displays:

* Original lesion image
* Grad-CAM heatmap

side-by-side.

---

### GPT-4o-mini Integration

For confident predictions, GPT-4o-mini generates concise educational explanations.

The generated response includes:

* Prediction interpretation
* General precautions
* Skin-care recommendations
* Professional consultation advice
* Research disclaimer

Example:

> The model predicts Melanoma with high confidence. It is recommended to monitor changes in skin appearance and consult a dermatologist for proper medical evaluation. This AI-generated prediction is intended for research purposes only and should not be considered a medical diagnosis.

---

## System Architecture

```text
User Upload
     │
     ▼
Skin vs Non-Skin Validator
     │
     ├── Reject Non-Skin Images
     │
     ▼
Multimodal Fusion Model
(Image + Metadata)
     │
     ▼
Confidence Threshold Check
     │
     ├── Reject Low Confidence Predictions
     │
     ▼
Grad-CAM Generation
     │
     ▼
GPT-4o-mini Explanation
     │
     ▼
React Frontend Display
```

---

## Technology Stack

### Backend

* Python
* FastAPI
* PyTorch
* OpenAI API
* Pillow
* NumPy

### Frontend

* React.js
* Axios
* CSS

### Deep Learning

* MobileNetV2
* EfficientNetB0
* ResNet18
* Multimodal Fusion Network

### Explainable AI

* Grad-CAM

### Deployment

* Docker
* Docker Compose

---

## Dataset

### Primary Dataset

HAM10000 Dataset

Contains dermoscopic skin lesion images belonging to seven diagnostic categories.

### Validator Dataset

A custom Skin vs Non-Skin dataset was created using:

* HAM10000 lesion images (Skin)
* Natural Images Dataset (Non-Skin)

The non-skin category includes:

* Airplanes
* Cars
* Cats
* Dogs
* Flowers
* Fruits
* Motorbikes
* People

This validator improves robustness against irrelevant user uploads.

---

## API Response

Successful prediction:

<img width="771" height="869" alt="image" src="https://github.com/user-attachments/assets/d46ae9f3-f915-4ffb-8b36-9edb5b01e568" />
<img width="752" height="456" alt="image" src="https://github.com/user-attachments/assets/f08af480-a08e-4e45-b1eb-c8148edd7fb4" />



```json
{
  "prediction": "Melanoma (mel)",
  "confidence": 91.4,
  "category": "Cancerous Lesion",
  "gradcam_image": "...",
  "llm_explanation": "...",
  "is_low_confidence": false
}
```

Rejected image:
<img width="752" height="456" alt="image" src="https://github.com/user-attachments/assets/4b869ba0-907e-4d42-b609-1bb0beb84220" />


```json
{
  "prediction": "Unknown",
  "confidence": 33.0,
  "category": "Unknown",
  "message": "Uploaded image does not appear to be a dermoscopic skin lesion image."
}
```

---

## Research Contributions

This project extends traditional skin lesion classification by integrating:

* Multimodal learning (Image + Metadata)
* Skin vs Non-Skin validation
* Confidence-based OOD detection
* Explainable AI using Grad-CAM
* LLM-generated educational explanations
* Full-stack deployment with React and FastAPI
* Dockerized backend and frontend services

These additions improve reliability, interpretability, and user experience compared to conventional image-only classification systems.

---

## Disclaimer

This project is intended for educational and research purposes only.

Predictions generated by the system should not be considered medical diagnoses. Users should consult qualified healthcare professionals for medical evaluation and treatment decisions.
