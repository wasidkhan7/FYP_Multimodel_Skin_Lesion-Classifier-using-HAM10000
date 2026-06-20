# ==================================================
# Skin vs Non-Skin Validator
# ==================================================
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image
from pathlib import Path
from io import BytesIO

# ==================================================
# Constants
# ==================================================
SKIN_IDX       = 1       # from class_to_idx: {'nonskin': 0, 'skin': 1}
NONSKIN_IDX    = 0
SKIN_THRESHOLD = 0.70    # 70% confidence required to pass as skin

# ==================================================
# Transform — must match training transform exactly
# ==================================================
validator_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==================================================
# Load Validator Model
# ==================================================
def load_skin_validator(model_path, device):
    """
    Loads the MobileNetV2 skin validator from a .pth file.
    Call this once at startup.

    Args:
        model_path : path to skin_validator_best.pth
        device     : torch.device (cuda or cpu)

    Returns:
        model (nn.Module) in eval mode
    """
    model = mobilenet_v2(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
             weights_only=True
        )
    )
    model = model.to(device)
    model.eval()
    print(f"✔ Skin validator loaded from: {model_path}")
    return model


# ==================================================
# Validate from image bytes (used in FastAPI)
# ==================================================
def validate_skin_from_bytes(
    validator_model,
    image_bytes: bytes,
    device,
    threshold: float = SKIN_THRESHOLD
) -> tuple[bool, float]:
    """
    Validates whether image bytes represent a skin lesion.

    Args:
        validator_model : loaded MobileNetV2 validator
        image_bytes     : raw bytes from uploaded image
        device          : torch.device
        threshold       : minimum skin confidence to pass

    Returns:
        (is_skin, skin_confidence)
        is_skin          : True if image passes validation
        skin_confidence  : float 0-100 (percentage)
    """
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return False, 0.0

    image_tensor = validator_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = validator_model(image_tensor)
        probs   = F.softmax(outputs, dim=1)[0]

    skin_confidence = probs[SKIN_IDX].item()
    is_skin         = skin_confidence >= threshold

    return is_skin, round(skin_confidence * 100, 2)


# ==================================================
# Validate from image path (used in predict.py)
# ==================================================
def validate_skin_from_path(
    validator_model,
    image_path,
    device,
    threshold: float = SKIN_THRESHOLD
) -> tuple[bool, float]:
    """
    Validates whether an image file is a skin lesion.

    Args:
        validator_model : loaded MobileNetV2 validator
        image_path      : path to image file
        device          : torch.device
        threshold       : minimum skin confidence to pass

    Returns:
        (is_skin, skin_confidence)
        is_skin          : True if image passes validation
        skin_confidence  : float 0-100 (percentage)
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return False, 0.0

    image_tensor = validator_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = validator_model(image_tensor)
        probs   = F.softmax(outputs, dim=1)[0]

    skin_confidence = probs[SKIN_IDX].item()
    is_skin         = skin_confidence >= threshold

    return is_skin, round(skin_confidence * 100, 2)