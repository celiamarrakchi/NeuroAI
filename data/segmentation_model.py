import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import numpy as np
import cv2
from typing import Any

class DiceLoss(nn.Module):
    """Standard Dice loss for binary segmentation."""
    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = torch.sigmoid(preds)
        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (preds * targets).sum(1)
        union = preds.sum(1) + targets.sum(1)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class BrainTumorSegmentationModel:
    """U-Net segmentation wrapper with Grad-CAM explainability."""
    def __init__(self, model_path: str, device: str = "cuda") -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.model.eval()
        self.activations = None
        self.gradients = None

    def _load_model(self, model_path: str) -> nn.Module:
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )
        state = torch.load(model_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            model.load_state_dict(state)
        model.to(self.device)
        return model

    def predict(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Segmentation probability map."""
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = torch.sigmoid(logits)
        return probs

    def get_target_layer(self) -> nn.Module:
        """Layer for Grad-CAM."""
        return self.model.encoder.layer4[-1].conv2

    # --- Grad-CAM ---
    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def get_gradcam(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Compute Grad-CAM heatmap for first image in batch."""
        self.model.zero_grad()
        image_tensor = image_tensor.to(self.device)
        image_tensor.requires_grad = True

        target_layer = self.get_target_layer()
        fhook = target_layer.register_forward_hook(self._forward_hook)
        bhook = target_layer.register_full_backward_hook(self._backward_hook)

        # Forward pass
        logits = self.model(image_tensor)
        if logits.shape[1] > 1:
            score = logits[:, 1, :, :].sum()
        else:
            score = logits.sum()

        # Backward pass
        score.backward(retain_graph=True)

        fhook.remove()
        bhook.remove()

        # Convert to numpy
        gradients = self.gradients[0].cpu().data.numpy()       # [C, H, W]
        feature_maps = self.activations[0].cpu().data.numpy()  # [C, H, W]

        # Compute Grad-CAM
        weights = np.mean(gradients, axis=(1, 2))  # global average pooling
        cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * feature_maps[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (image_tensor.shape[3], image_tensor.shape[2]))
        cam -= cam.min()
        cam /= cam.max() + 1e-6
        return cam
