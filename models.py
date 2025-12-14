# models.py
from __future__ import annotations
from typing import Dict, Any, List, Literal
from pydantic import BaseModel, Field
from datetime import datetime

class ParsedHealthData(BaseModel):
    patient_id: int
    name: str = Field(default="unknown")
    age: int
    sex: Literal["male", "female", "other", "unknown"]
    symptoms: Dict[str, bool]
    raw_data: Dict[str, Any]

class SymptomScore(BaseModel):
    patient_id: int
    risk_score: int = Field(..., ge=0, le=100)
    severity_level: Literal["low", "moderate", "high", "critical"]
    total_symptoms: int
    red_flag_alerts: List[str] = Field(default_factory=list)

class CancerRiskAssessment(BaseModel):
    patient_id: int
    cancer_probability: int = Field(..., ge=0, le=100)
    risk_category: Literal["very low", "low", "moderate", "high", "very high"]
    primary_concerns: List[str] = Field(default_factory=list)
    recommended_tests: List[str] = Field(default_factory=list)
    evidence_sources: List[str] = Field(default_factory=list)

class SafetyReport(BaseModel):
    validation_passed: bool
    issues: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

class FinalClinicalReport(BaseModel):
    report_metadata: Dict[str, Any]
    patient_info: Dict[str, Any]
    symptoms_analysis: Dict[str, Any]
    cancer_risk_assessment: Dict[str, Any]
    explanation: str
    quality_control: Dict[str, Any]
    provenance: List[str]
    next_steps: Dict[str, Any]
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")