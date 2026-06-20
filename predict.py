import torch
import torch.nn.functional as F
from pathlib import Path

try:
    from utils.preprocessing import preprocess_image, preprocess_metadata
    from utils.skin_validator import load_skin_validator, validate_skin_from_path
    from model import load_fusion_model, device
    from utils.class_names import class_names
except ModuleNotFoundError:
    from utils.preprocessing import preprocess_image, preprocess_metadata
    from utils.skin_validator import load_skin_validator, validate_skin_from_path
    from model import load_fusion_model, device
    from utils.class_names import class_names

BASE_DIR = Path(__file__).resolve().parent

# ==================================================
# Load models globally
# ==================================================
model = load_fusion_model(
    BASE_DIR / "models" / "fusion_model.pth",
    device
)

validator_model = load_skin_validator(
    BASE_DIR / "models" / "skin_validator.pth",
    device
)

# ==================================================
# Prediction Function
# ==================================================
def predict_fusion(image_path, age, sex, localization):

    # ------------------------------------------
    # Step 1 — Skin Validation Gate
    # ------------------------------------------
    is_skin, skin_confidence = validate_skin_from_path(
        validator_model,
        image_path,
        device
    )

    if not is_skin:
        return {
            "prediction"      : "Rejected",
            "confidence"      : 0.0,
            "skin_confidence" : skin_confidence,
            "error"           : "Image does not appear to be a dermoscopic skin lesion image.",
            "suggestion"      : "Please upload a valid dermoscopic skin lesion image."
        }

    # ------------------------------------------
    # Step 2 — Preprocess image
    # ------------------------------------------
    image    = preprocess_image(image_path)
    image    = image.to(device)

    # ------------------------------------------
    # Step 3 — Preprocess metadata
    # ------------------------------------------
    metadata = preprocess_metadata(age, sex, localization)
    metadata = metadata.to(device)

    # ------------------------------------------
    # Step 4 — Inference
    # ------------------------------------------
    with torch.no_grad():
        outputs  = model(image, metadata)
        probs    = F.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()

    confidence = probs[0, pred_idx].item()

    # ------------------------------------------
    # Step 5 — Return prediction
    # ------------------------------------------
    return {
        "prediction"      : class_names[pred_idx],
        "confidence"      : round(confidence * 100, 2),
        "skin_confidence" : skin_confidence
    }