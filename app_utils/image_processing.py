import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transform():
    """Get image transformation pipeline"""
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def load_and_preprocess_image(image_path, device='cuda'):
    """Load and preprocess MRI image"""
    img = np.array(Image.open(image_path).convert("RGB"))
    transform = get_transform()
    transformed = transform(image=img)
    img_tensor = transformed["image"].unsqueeze(0).to(device)
    return img_tensor, img

def overlay_cam_on_image(img, cam, alpha=0.5):
    """Overlay Grad-CAM heatmap on original image (robust version)."""
    import cv2
    import numpy as np
    import torch

    # Convert PyTorch tensor to NumPy
    if isinstance(cam, torch.Tensor):
        cam = cam.detach().cpu().numpy()

    # Remove extra dimensions
    cam = np.squeeze(cam)

    # Replace NaNs/Infs with 0
    cam = np.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize to 0-1
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min != 0:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    # Convert to uint8 single channel
    cam_uint8 = np.uint8(255 * cam)

    # Resize heatmap to match image size
    cam_resized = cv2.resize(cam_uint8, (img.shape[1], img.shape[0]))

    # Make sure it's 2D (CV_8UC1) for applyColorMap
    if cam_resized.ndim != 2:
        cam_resized = cam_resized[:, :, 0]

    # Apply JET colormap
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)

    # Convert BGR to RGB
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Ensure original image is float32
    img_float = img.astype(np.float32)

    # Overlay
    overlay = alpha * heatmap.astype(np.float32) + (1 - alpha) * img_float
    return np.uint8(overlay)



def calculate_tumor_metrics(pred_mask):
    """Calculate tumor area and location"""
    pred_binary = (pred_mask > 0.5).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(pred_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return {
            "area_pixels": 0,
            "area_percentage": 0.0,
            "centroid": {"x": 0, "y": 0},
            "bounding_box": {"x": 0, "y": 0, "width": 0, "height": 0},
            "contour_count": 0
        }
    
    # Get largest contour (main tumor)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate area
    area_pixels = cv2.contourArea(largest_contour)
    total_pixels = pred_binary.shape[0] * pred_binary.shape[1]
    area_percentage = (area_pixels / total_pixels) * 100
    
    # Calculate centroid
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return {
        "area_pixels": int(area_pixels),
        "area_percentage": round(float(area_percentage), 2),
        "centroid": {"x": int(cx), "y": int(cy)},
        "bounding_box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
        "contour_count": len(contours)
    }