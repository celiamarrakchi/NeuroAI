import torch
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
    def _forward_hook(self, module, input, output):
        self.activations = output
        
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def generate_cam(self, input_tensor):
        """Generate Grad-CAM heatmap"""
        self.model.eval()
        
        # Register hooks
        handle_fw = self.target_layer.register_forward_hook(self._forward_hook)
        handle_bw = self.target_layer.register_backward_hook(self._backward_hook)
        
        # Forward pass
        pred = self.model(input_tensor)
        pred_sigmoid = torch.sigmoid(pred)
        
        # Backward pass
        loss = pred_sigmoid.sum()
        self.model.zero_grad()
        loss.backward()
        
        # Compute Grad-CAM
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        
        # Ensure numeric array
        cam = cam.detach().cpu().numpy().astype(np.float32)
        
        # Normalize heatmap to [0, 1]
        cam = np.maximum(cam, 0)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Remove hooks
        handle_fw.remove()
        handle_bw.remove()
        
        return cam

# Example usage:
# gradcam = GradCAM(model, model.layer4[2].conv3)
# heatmap = gradcam.generate_cam(input_tensor)
# xai_results['gradcam_heatmap'] = heatmap  # store as numeric array, not list
