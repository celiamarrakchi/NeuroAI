import os
import numpy as np
import cv2
from app_utils.llama3_llm import Llama3LLM


class XAIAgent:
    def __init__(self, seg_model, image_tensor, prediction_mask):
        self.seg_model = seg_model
        self.image_tensor = image_tensor
        self.prediction_mask = prediction_mask.astype(float)
        self.gradcam_heatmap = None

        self.llm = Llama3LLM(
            model="llama3:latest",
            temperature=0.3,
            max_tokens=1200
        )

    # -------------------------------------------------------
    # 1. GRAD-CAM GENERATION
    # -------------------------------------------------------
    def generate_gradcam(self):
        cam = self.seg_model.get_gradcam(self.image_tensor)
        self.gradcam_heatmap = cam

        hot_regions = (cam > 0.7).sum()
        total_pixels = cam.size
        hot_percentage = (hot_regions / total_pixels) * 100

        return {
            "hot_regions": int(hot_regions),
            "hot_percentage": float(round(hot_percentage, 2))
        }

    # -------------------------------------------------------
    # 2. ATTENTION ANALYSIS
    # -------------------------------------------------------
    def analyze_attention(self):
        if self.gradcam_heatmap is None:
            return {"error": "Grad-CAM not generated."}

        cam = self.gradcam_heatmap
        mask = self.prediction_mask

        # Resize if segmentation mask size differs from CAM size
        if cam.shape != mask.shape:
            cam = cv2.resize(cam, (mask.shape[1], mask.shape[0]))

        mask_bin = (mask > 0.5).astype(float)

        alignment_numer = (cam * mask_bin).sum()
        mask_area = mask_bin.sum()

        alignment_score = (alignment_numer / mask_area * 100) if mask_area > 0 else 0

        peak_value = cam.max()
        peak_locations = np.where(cam > 0.8 * peak_value)
        num_peaks = len(peak_locations[0])

        mean_attention = alignment_numer / max(mask_area, 1)

        return {
            "alignment_score": float(round(alignment_score, 2)),
            "peak_value": float(round(peak_value, 3)),
            "num_peaks": int(num_peaks),
            "mean_attention": float(round(mean_attention, 3))
        }

    # -------------------------------------------------------
    # 3. TEXTUAL LLaMA-3 EXPLANATION
    # -------------------------------------------------------
    def generate_explanation(self, attention_summary=None):
        if attention_summary is None:
            attention_summary = self.analyze_attention()

        if "error" in attention_summary:
            return "Explanation cannot be generated: " + attention_summary["error"]

        prompt = (
            "You are an expert medical imaging explainability agent.\n"
            "Based on the following attention analysis from a brain tumor segmentation model:\n"
            f"- Alignment Score: {attention_summary['alignment_score']}%\n"
            f"- Peak Attention Value: {attention_summary['peak_value']}\n"
            f"- Number of High-Attention Peaks: {attention_summary['num_peaks']}\n"
            f"- Mean Attention in Tumor Region: {attention_summary['mean_attention']}\n"
            "Generate a concise medical-style textual justification describing:\n"
            "- tumor location\n"
            "- what the model focused on\n"
            "- the reliability of this focus\n"
            "- the clinical relevance of the highlighted regions."
        )

        return self.llm._call(prompt)

    # -------------------------------------------------------
    # 4. FULL AGENT WORKFLOW
    # -------------------------------------------------------
    def run(self):
        gradcam_summary = self.generate_gradcam()
        attention_summary = self.analyze_attention()
        explanation_text = self.generate_explanation(attention_summary)

        return {
            "gradcam_summary": gradcam_summary,
            "attention_summary": attention_summary,
            "gradcam_heatmap": self.gradcam_heatmap,
            "explanation": explanation_text
        }
