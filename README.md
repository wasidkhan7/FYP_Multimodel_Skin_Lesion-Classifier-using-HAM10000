
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

## Comparative Performance Analysis

An extensive ablation study was conducted on the HAM10000 dataset to evaluate the effectiveness of different image feature extractors, metadata learning, and fusion strategies. The Metadata MLP trained solely on patient information (age, sex, and lesion localization) achieved a test accuracy of 45.64%, with a Macro F1-score of 0.2322 and a Weighted F1-score of 0.5183. These results demonstrate that metadata alone contains limited discriminatory information for accurate skin lesion classification and is heavily affected by class imbalance.

Among the image-only models, EfficientNetB0 achieved the highest performance with a test accuracy of 88.52%, a Macro F1-score of 0.8292, and a Weighted F1-score of 0.8866. ResNet18 followed closely with an accuracy of 88.02%, while MobileNetV2 achieved 87.52%. Although the difference in accuracy appears small, EfficientNetB0 consistently produced the strongest F1-scores, indicating better overall class-wise discrimination and more balanced predictions across lesion categories.

EfficientNetB0 outperformed the other architectures primarily due to its compound scaling methodology, which jointly scales network depth, width, and input resolution. This design enables the model to capture fine-grained lesion characteristics such as color variation, border irregularities, texture patterns, and structural features more effectively than conventional architectures. These characteristics are particularly important in dermoscopic image analysis where subtle visual differences separate benign and malignant lesions.

To investigate whether patient metadata could improve classification performance, multiple multimodal fusion strategies were evaluated, including Simple Concatenation Fusion, Attention Fusion, and Gated Fusion. Contrary to expectations, none of the fusion approaches surpassed the image-only EfficientNetB0 baseline. The best fusion model, EfficientNetB0 with Gated Fusion, achieved a test accuracy of 88.12%, a Macro F1-score of 0.8227, and a Weighted F1-score of 0.8836, which remained slightly below the image-only EfficientNetB0 model. Similar trends were observed for ResNet18 and MobileNetV2 fusion variants, suggesting that the available metadata contributed limited additional information beyond what was already captured from the dermoscopic images.

Among the fusion techniques, Gated Fusion consistently outperformed Attention Fusion and Simple Concatenation Fusion, indicating that adaptive feature weighting is more effective than direct feature merging. However, the performance gains were insufficient to exceed the strongest image-only baseline. Furthermore, the Hair Removal preprocessing experiment was not adopted in the final system because it reduced prediction confidence and introduced additional misclassifications, negatively affecting overall model reliability.

Based on these findings, EfficientNetB0 was selected as the primary image backbone for the final system due to its superior classification performance, strong class-wise generalization, and computational efficiency. The study also highlights an important research observation: in the HAM10000 dataset, high-quality dermoscopic image features contribute significantly more to classification performance than the available demographic metadata, making image representation the dominant factor in model success.



<img width="1235" height="565" alt="ablationtable" src="https://github.com/user-attachments/assets/95a75e6f-ff22-452e-80df-0971ce98d9d7" />



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
