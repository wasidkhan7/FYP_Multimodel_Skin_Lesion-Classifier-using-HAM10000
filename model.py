# fusion model

# model.py should contain ONLY:
# ✔ imports
# ✔ FusionModel class
# ✔ load_model() function

# ✔ imports
import torch
import torch.nn as nn

# Define ✔ FusionModel class with EfficientNetB0 backbone for image features and MLP for metadata features

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)

class FusionModel(nn.Module):

    def __init__(self,num_classes=7,
                 meta_input_dim=17, # 15  for localization +1  age + 1 sex
                 meta_hidden_dim=32):
        super().__init__()

        # ==================================================
        # Image Branch (EfficientNetB0 Backbone)
        # ==================================================
        self.image_model = efficientnet_b0(weights=None)

        # Remove EfficientNet classifier
        self.image_feature_dim = (self.image_model.classifier[1].in_features)

        self.image_model.classifier = nn.Identity()

        # Output shape:
        # (batch_size, 1280)

        # ==================================================
        # Metadata Branch (MLP)
        # ==================================================

        self.meta_model = nn.Sequential(
            nn.Linear(meta_input_dim, meta_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(meta_hidden_dim, meta_hidden_dim),
            nn.ReLU()
        )

        # ==================================================
        # Gating Module
        # ==================================================

        self.gate = nn.Sequential(
            nn.Linear(
                meta_hidden_dim,
                self.image_feature_dim
            ),
            nn.Sigmoid()
        )


        # ==================================================
        # Fusion Classifier
        # ==================================================

        self.classifier = nn.Sequential(
            nn.Linear(self.image_feature_dim + meta_hidden_dim,128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, images, metadata):

        # ------------------------------------------
        # Image features
        # ------------------------------------------

        img_features = self.image_model(images)
        
        # Shape of img data:
        # (B, 1280)
        # ------------------------------------------
        # Metadata features
        # ------------------------------------------

        meta_features = self.meta_model(metadata)

        # Shape of metadata:
        # (B, meta_hidden_dim)

        # ------------------------------------------
        # Gating fusion
        # ------------------------------------------

        gate_weights = self.gate(
            meta_features
        )

        gated_img_features = (
            img_features *
            gate_weights
        )

        # ------------------------------------------
        # Fusion
        # ------------------------------------------

        fused = torch.cat(
            [
                gated_img_features,
                meta_features
            ],
            dim=1
        )

        # ------------------------------------------
        # Classification
        # ------------------------------------------

        out = self.classifier(fused)

        return out
    
# ==================================================
# Device Configuration
# ==================================================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
# ===================================================
# ✔ load_model() function i,e EFFICIENTNET WEIGHTS ( We are transfering efficientnetb0 models wight to fusion model)
# ===================================================
def load_fusion_model(model_path, device):

    model = FusionModel(
        num_classes=7,
        meta_input_dim=17,
        meta_hidden_dim=32
    )

    model = model.to(device)

    # Load trained weights
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    return model


if __name__ == "__main__":

    model = load_fusion_model(
        "models/fusion_model.pth",
        device
    )

    print("Fusion model loaded successfully.")