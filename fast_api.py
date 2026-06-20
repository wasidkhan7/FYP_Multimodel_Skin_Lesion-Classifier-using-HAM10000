from contextlib import asynccontextmanager
from io import BytesIO
from typing import Annotated, Dict, List, Literal
import torch
import torch.nn.functional as F
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, ValidationError, field_validator
from utils.skin_validator import load_skin_validator, validate_skin_from_bytes
import base64
from gradcam_utils import generate_gradcam_from_bytes
from model import device, load_fusion_model
from utils.class_names import class_names
from utils.preprocessing import preprocess_metadata, val_transform

from llm_utils import generate_llm_explanation


MODEL_PATH = "models/fusion_model.pth"
VALIDATOR_PATH  = "models/skin_validator.pth"

CONFIDENCE_THRESHOLD = 55.0

VALID_SEXES = {"male", "female"}
VALID_LOCALIZATIONS = {
    "scalp",
    "ear",
    "face",
    "back",
    "trunk",
    "chest",
    "upper extremity",
    "abdomen",
    "unknown",
    "lower extremity",
    "genital",
    "neck",
    "hand",
    "foot",
    "acral",
}
CANCER_CLASSES = {
    "Actinic Keratoses (akiec)",
    "Basal Cell Carcinoma (bcc)",
    "Melanoma (mel)",}


SexValue = Literal["male", "female"]
LocalizationValue = Literal[
    "scalp",
    "ear",
    "face",
    "back",
    "trunk",
    "chest",
    "upper extremity",
    "abdomen",
    "unknown",
    "lower extremity",
    "genital",
    "neck",
    "hand",
    "foot",
    "acral",
]


class PredictionMetadata(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sex: SexValue
    localization: LocalizationValue

    @field_validator("sex", "localization", mode="before")
    @classmethod
    def normalize_text_fields(cls, value: str) -> str:
        if isinstance(value, str):
            return value.strip().lower()
        return value


class TopPrediction(BaseModel):
    class_name: str
    confidence: float


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    category: str
    is_low_confidence: bool
    message: str
    gradcam_image: str |None = None
    probabilities: Dict[str, float]
    top_predictions: List[TopPrediction]
    metadata: PredictionMetadata
    disclaimer: str
    llm_explanation: str | None = None


model = None
validator_model = None


# Loads the model once when the API starts instead of loading it on every request.
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, validator_model
    model = load_fusion_model(MODEL_PATH, device)
    validator_model = load_skin_validator(VALIDATOR_PATH, device)
    yield


app = FastAPI(
    title="Skin Lesion Fusion Model API",
    description="Predicts skin lesion class from a dermoscopic image and metadata.",
    version="1.0.0",
    lifespan=lifespan,)
# Allow CORS for local development with React frontend on localhost:3000.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_uploaded_image(image_bytes: bytes) -> torch.Tensor:
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image.",
        ) from exc

    image_tensor = val_transform(image).unsqueeze(0)
    return image_tensor

def encode_gradcam_as_data_url(gradcam_image) -> str:

    image_buffer = BytesIO()

    Image.fromarray(
        gradcam_image
    ).save(
        image_buffer,
        format="PNG"
    )

    encoded_image = base64.b64encode(
        image_buffer.getvalue()
    ).decode(
        "utf-8"
    )

    return (
        f"data:image/png;base64,{encoded_image}"
    )


def get_prediction_metadata(
    age: Annotated[int, Form(...)],
    sex: Annotated[str, Form(...)],
    localization: Annotated[str, Form(...)],
) -> PredictionMetadata:
    try:
        return PredictionMetadata(
            age=age,
            sex=sex,
            localization=localization,
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=exc.errors(),
        )from exc


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "message": "Skin Lesion Fusion Model API is running.",
        "predict_endpoint": "/predict",
    }



# =========================================================
# Health Check Endpoint
# =========================================================

@app.get("/health")
def health_check():

    return {

        "status": "healthy",

        "model_loaded": model is not None,
        "validator_loaded": validator_model is not None,

        "device": str(device)
    }

@app.get("/metadata-options")
def metadata_options() -> Dict[str, List[str]]:
    return {
        "sex": sorted(VALID_SEXES),
        "localization": sorted(VALID_LOCALIZATIONS),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    # Keep form fields flat: image, age, sex, localization.
    # A nested Pydantic form model can cause 422 errors in multipart requests.
    metadata: Annotated[PredictionMetadata, Depends(get_prediction_metadata)],
    image: UploadFile = File(...),
) -> PredictionResponse:
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Upload must be an image file.",
        )

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty.",
        )
    
    # ── Skin Validator Gate ──────────────────────────────────
    is_skin, skin_confidence = validate_skin_from_bytes(
        validator_model,
        image_bytes,
        device
    )   

    if not is_skin:
        raise HTTPException(
            status_code=400,
            detail={
                "error"           : "Image rejected by skin validator.",
                "reason"          : "Uploaded image does not appear to be a dermoscopic skin lesion image.",
                "skin_confidence" : skin_confidence,
                "suggestion"      : "Please upload a valid dermoscopic skin lesion image."
            }
        )   

    image_tensor = preprocess_uploaded_image(image_bytes).to(device)

    metadata_tensor = preprocess_metadata(
        metadata.age,
        metadata.sex,
        metadata.localization,
    ).to(device)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet. Please try again shortly.",
        )

    with torch.no_grad():
        outputs = model(image_tensor, metadata_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        pred_idx = int(torch.argmax(probabilities).item())

    prediction = class_names[pred_idx]

    # =========================================================
    # Top-K Predictions
    # =========================================================

    top_k = 3

    top_probs, top_indices = torch.topk(
        probabilities,
        top_k
        )

    top_predictions = [
        TopPrediction(
            class_name=class_names[int(idx.item())],
            confidence=round(float(prob.item()) * 100, 2),
        )

        for prob, idx in zip(top_probs, top_indices)]


    confidence = round(float(probabilities[pred_idx].item()) * 100, 2)
    lesion_category = (
        "Cancerous Lesion"
        if prediction in CANCER_CLASSES
        else "Non-Cancerous Lesion"
    )
    
    # =========================================================
    # OOD Handling
    # =========================================================

    if confidence < CONFIDENCE_THRESHOLD:
    
        prediction = "Unknown"
    
        lesion_category = "Unknown"
    
        top_predictions = []
    
        probabilities_dict = {}
    
    else:
    
        lesion_category = (
            "Cancerous Lesion"
            if prediction in CANCER_CLASSES
            else "Non-Cancerous Lesion"
        )
    
        probabilities_dict = {
            class_name: round(
                float(probability.item()) * 100,
                2
            )
            for class_name, probability in zip(
                class_names,
                probabilities 
            )
        }

    # =========================================================
    # Generate LLM Explanation
    # Only for confident predictions
    # =========================================================
    # =========================================================
    # OOD Handling
    # =========================================================

    probabilities_dict = {
    class_name: round(float(probability.item()) * 100, 2)
    for class_name, probability in zip(class_names, probabilities) }

    llm_explanation = None
    gradcam_data_url = None

    if confidence >= CONFIDENCE_THRESHOLD:

        # Generate GPT explanation
        llm_explanation = generate_llm_explanation(
            prediction=prediction,
            confidence=confidence
        )

        # Generate GradCAM
        gradcam_image = generate_gradcam_from_bytes(
            fusion_model=model,
            image_bytes=image_bytes,
            metadata=metadata_tensor,
            device=device,
        )

        gradcam_data_url = encode_gradcam_as_data_url(
            gradcam_image
        )

    else:

        # OOD / Low confidence
        prediction = "Unknown"    
        lesion_category = "Unknown"
        top_predictions = []
        probabilities_dict = {}




    return PredictionResponse(
        prediction=prediction,
        confidence=confidence,
        category=lesion_category,
        gradcam_image=gradcam_data_url,
        llm_explanation=llm_explanation,
        is_low_confidence=confidence < CONFIDENCE_THRESHOLD,
        message=(
            "No clear skin lesion detected. Please upload a proper dermoscopic "
            "skin lesion image."
            if confidence < CONFIDENCE_THRESHOLD
            else "Prediction completed successfully."
        ),
        probabilities=probabilities_dict,

        top_predictions=top_predictions,
        
        metadata=metadata,
        disclaimer=(
            "Predictions are for research purposes only and should not be "
            "considered medical diagnosis."
        ),
    )
