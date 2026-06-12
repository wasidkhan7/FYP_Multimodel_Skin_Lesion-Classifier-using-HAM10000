
#Predict function

import torch

import torch.nn.functional as F

from pathlib import Path


try:
    from utils.preprocessing import (
        preprocess_image,
        preprocess_metadata)

    from model import (
        load_fusion_model,
        device)

    from utils.class_names import class_names
except ModuleNotFoundError:
    from utils.preprocessing import (
        preprocess_image,
        preprocess_metadata)

    from model import (
        load_fusion_model,
        device)

    from utils.class_names import class_names


BASE_DIR = Path(__file__).resolve().parent



# Load model globally

model = load_fusion_model(
    BASE_DIR / "models" / "fusion_model.pth",
    device
)


# ==================================================
# Prediction Function
# ==================================================

def predict_fusion(

    image_path,
    age,
    sex,
    localization
):

    # ------------------------------------------
    # Preprocess image
    # ------------------------------------------

    image = preprocess_image(image_path)

    image = image.to(device)

    # ------------------------------------------
    # Preprocess metadata
    # ------------------------------------------

    metadata = preprocess_metadata(
        age,
        sex,
        localization
    )

    metadata = metadata.to(device)

    # ------------------------------------------
    # Inference
    # ------------------------------------------

    with torch.no_grad():

        outputs = model(
            image,
            metadata
        )

        probs = F.softmax(
            outputs,
            dim=1
        )

        pred_idx = torch.argmax(
            probs,
            dim=1
        ).item()

        confidence = probs[
            0,
            pred_idx
        ].item()

    # ------------------------------------------
    # Return prediction
    # ------------------------------------------

    return {

        "prediction": class_names[pred_idx],

        "confidence": round(
            confidence * 100,
            2
        )
    }
