# transforms for validation and testing

from torchvision import transforms
from PIL import Image
import torch

# ==================================================
# Validation / Inference Transform
# ==================================================

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ==================================================
# Image Preprocessing Function
# ==================================================

def preprocess_image(image_path,transform=val_transform):

    img=Image.open(image_path).convert("RGB")

    img=transform(img)

    # Add batch dimension
    img=img.unsqueeze(0)

    return img

# ==================================================
# Metadata Preprocessing Function
# ==================================================

def preprocess_metadata(age,sex,localization):

    # ------------------------------------------
    # Normalize age
    # IMPORTANT:
    # Use SAME normalization as training
    # ------------------------------------------

    age_normalized=(age-51.85)/16.9

    # ------------------------------------------
    # Encode sex
    # ------------------------------------------

    sex_mapping={
        "male":1,
        "female":0
    }

    sex_encoded=sex_mapping[sex.lower()]

    # ------------------------------------------
    # One-Hot Encode localization
    # ------------------------------------------

    localization_categories=[
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
        "acral"
    ]

    localization_vector=[0]*len(localization_categories)

    idx=localization_categories.index(
        localization.lower()
    )

    localization_vector[idx]=1

    # ------------------------------------------
    # Convert to tensor
    # ------------------------------------------

    metadata=torch.tensor(
        [[
            age_normalized,
            sex_encoded,
            *localization_vector
        ]],
        dtype=torch.float32
    )

    return metadata


metadata=preprocess_metadata(
    age=45,
    sex="male",
    localization="back"
)

print(metadata.shape)
print(metadata)