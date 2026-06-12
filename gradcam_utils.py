
import numpy as np
import torch
from io import BytesIO
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import (
    show_cam_on_image)
from utils.preprocessing import preprocess_image, val_transform

# ==================================================
# Grad-CAM Visualization Utility                   #
# ==================================================




# ==================================================
# Wrapper Model For Grad-CAM
# ==================================================

class GradCAMWrapper(torch.nn.Module):

    def __init__(self,fusion_model,metadata):

        super().__init__()

        self.fusion_model = fusion_model

        self.metadata = metadata

    def forward(self, image):   # Grad-CAM now thinks model input is ONLY the image

        return self.fusion_model(image,self.metadata) 
    

    
# ==================================================
# Generate Grad-CAM
# ==================================================

def generate_gradcam(fusion_model,image_path,metadata,device):

    #PREPROCESS IMAGE
    input_tensor = preprocess_image(
        image_path
    ).to(device)

    #WRAP MODEL
    wrapped_model = GradCAMWrapper(fusion_model,metadata)

    #Select target layer for Grad-CAM (last conv layer of EfficientNetB0)
    target_layer = wrapped_model.fusion_model.image_model.features[-1]

    #cteate Grad-CAM object
    cam = GradCAM(model=wrapped_model,target_layers=target_layer)

    #create heatmap
    grayscale_cam = cam(input_tensor=input_tensor)[0]

    # Load original image
    rgb_img = np.array(
        Image.open(image_path).convert("RGB").resize((224, 224))) / 255.0


    # Overlay heatmap on original image
    visualization = show_cam_on_image(

        rgb_img,

        grayscale_cam,

        use_rgb=True
    )

    return visualization


# ==================================================
# Generate Grad-CAM From Uploaded Image Bytes
# ==================================================

def generate_gradcam_from_bytes(fusion_model, image_bytes, metadata, device):

    # Convert the uploaded file bytes into a PIL RGB image.
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Apply the same validation/inference transform used by prediction.
    input_tensor = val_transform(image).unsqueeze(0).to(device)

    # Wrap the fusion model so Grad-CAM only needs the image tensor as input.
    wrapped_model = GradCAMWrapper(fusion_model, metadata)

    # Select the final convolution feature block from EfficientNetB0.
    target_layer = wrapped_model.fusion_model.image_model.features[-1]

    # Create the Grad-CAM object for the selected layer.
    cam = GradCAM(model=wrapped_model, target_layers=target_layer)

    # Generate the grayscale class activation map.
    grayscale_cam = cam(input_tensor=input_tensor)[0]

    # Resize the original image to match the model input size for overlay.
    rgb_img = np.array(image.resize((224, 224))) / 255.0

    # Overlay the heatmap on the original image and return a RGB numpy array.
    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    return visualization
