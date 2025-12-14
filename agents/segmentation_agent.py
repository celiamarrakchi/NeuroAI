from __future__ import annotations
import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from data.segmentation_model import BrainTumorSegmentationModel
from app_utils.image_processing import calculate_tumor_metrics

# Optional LLaMA-3 LLM integration
try:
    from app_utils.llama3_llm import Llama3LLM  # type: ignore
    _HAS_LLAMA = True
except Exception:
    Llama3LLM = None
    _HAS_LLAMA = False

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s - %(message)s")
_handler.setFormatter(_formatter)
if not logger.handlers:
    logger.addHandler(_handler)


# -------------------------------------------------------------------
# Segmentation Agent
# -------------------------------------------------------------------
class SegmentationAgent:
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        llama_api_key: Optional[str] = None,
        llama_model: str = "llama3:latest",
    ) -> None:

        # -------------------------------------------------------------
        # Device
        # -------------------------------------------------------------
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing SegmentationAgent on device={self.device}")

        # -------------------------------------------------------------
        # Load Segmentation Model
        # -------------------------------------------------------------
        self.seg_model = BrainTumorSegmentationModel(model_path, device=self.device)
        self.prediction_mask: Optional[np.ndarray] = None
        self.metrics: Dict[str, Any] = {}

        # -------------------------------------------------------------
        # LLaMA-3 Initialization (Optional)
        # -------------------------------------------------------------
        self.llama: Optional[Llama3LLM] = None
        api_key = llama_api_key or os.getenv("LLAMA_API_KEY")

        if _HAS_LLAMA and api_key:
            try:
                self.llama = Llama3LLM(
                    api_key=api_key,
                    model=llama_model,
                    temperature=0.2,
                    max_tokens=300,
                )
                logger.info("Llama3LLM initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize LLaMA-3. Continuing without LLM. Error: {e}")
                self.llama = None
        else:
            if not api_key:
                logger.info("No LLAMA_API_KEY found. LLM features disabled.")
            else:
                logger.warning("Llama3LLM module missing. Install llama3 dependencies to enable LLM features.")

    # -------------------------------------------------------------------
    # Core: Segmentation
    # -------------------------------------------------------------------
    def segment_tumor(self, image_tensor: torch.Tensor) -> Tuple[np.ndarray, str]:
        logger.info("Starting segmentation")

        if image_tensor is None:
            raise ValueError("image_tensor is None")

        pred_prob = self.seg_model.predict(image_tensor)
        pred_prob = pred_prob.detach().cpu().numpy()

        pred_prob = np.squeeze(pred_prob) if pred_prob.ndim > 2 else pred_prob

        mask = (np.clip(pred_prob, 0, 1) >= 0.5).astype(np.uint8)
        self.prediction_mask = mask

        msg = f"Segmentation completed. Mask shape: {mask.shape}"
        logger.info(msg)
        return mask, msg

    # -------------------------------------------------------------------
    # Core: Metrics
    # -------------------------------------------------------------------
    def calculate_metrics(self, mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        logger.info("Calculating tumor metrics")

        mask = mask if mask is not None else self.prediction_mask
        if mask is None:
            logger.warning("No mask available; returning empty metrics.")
            self.metrics = {}
            return {}

        try:
            metrics = calculate_tumor_metrics(mask)
        except Exception:
            metrics = {
                "area_pixels": 0,
                "area_percentage": 0.0,
                "centroid": {"x": None, "y": None},
                "bounding_box": {"x": None, "y": None, "width": None, "height": None},
                "contour_count": 0,
            }

        metrics["area_pixels"] = int(metrics.get("area_pixels", 0))
        metrics["area_percentage"] = float(metrics.get("area_percentage", 0.0))

        self.metrics = metrics
        logger.info(f"Metrics calculated: {metrics}")
        return metrics

    # -------------------------------------------------------------------
    # Optional LLM Orchestration (now uses LLaMA-3)
    # -------------------------------------------------------------------
    def create_simple_agent_prompt(self, question: str) -> str:
        return (
            "You are an orchestration agent for a medical imaging system. "
            "Answer with one token from [SEGMENT, METRICS, BOTH, NONE].\n"
            f"Task: {question}\n"
        )

    def _ask_llama_for_plan(self, question: str) -> Optional[str]:
        if not self.llama:
            return None
        try:
            prompt = self.create_simple_agent_prompt(question)
            out = self.llama._call(prompt)
            token = str(out).strip().split()[0].upper()
            return token if token in {"SEGMENT", "METRICS", "BOTH", "NONE"} else None
        except Exception:
            return None

    # -------------------------------------------------------------------
    # Public Run
    # -------------------------------------------------------------------
    def run(self, image_tensor: torch.Tensor, use_agent: bool = False) -> Dict[str, Any]:
        logger.info(f"Run called with use_agent={use_agent}")

        plan = (
            self._ask_llama_for_plan("Perform brain tumor segmentation and calculate metrics.")
            if use_agent else None
        )

        run_segment = run_metrics = True

        if plan == "SEGMENT":
            run_metrics = False
        elif plan == "METRICS":
            run_segment = False
        elif plan == "NONE":
            run_segment = run_metrics = False

        messages = []
        mask = None

        # Segmentation
        if run_segment:
            try:
                mask, msg = self.segment_tumor(image_tensor)
                messages.append(msg)
            except Exception as e:
                logger.warning(f"Segmentation failed: {e}")
                messages.append(f"Segmentation failed: {e}")

        # Metrics
        if run_metrics:
            try:
                metrics = self.calculate_metrics(mask)
                messages.append("Metrics calculation completed.")
            except Exception as e:
                logger.warning(f"Metrics failed: {e}")
                metrics = {}
                messages.append(f"Metrics failed: {e}")
        else:
            metrics = self.metrics or {}

        return {
            "agent_output": "\n".join(messages) or "No action taken.",
            "metrics": metrics,
            "prediction_mask": self.prediction_mask,
        }


if __name__ == "__main__":
    logger.info("SegmentationAgent imported.")
