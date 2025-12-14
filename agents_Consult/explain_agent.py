# agents/explain_agent.py
from models import ParsedHealthData, SymptomScore, CancerRiskAssessment
from typing import List  # ← Add this line
from langchain_ollama import ChatOllama

class ExplainabilityAgent:
    def __init__(self):
        self.llm = ChatOllama(model="llama3.1", temperature=0.3)

    def explain(self, parsed: ParsedHealthData, score: SymptomScore, assessment: CancerRiskAssessment) -> str:
        active = [k.replace("_", " ") for k, v in parsed.symptoms.items() if v]
        print(f"   [llama3.1] Generating warm explanation for Patient {parsed.patient_id}...")

        base_text = f"""# Brain Tumor Risk Assessment Report

**Patient**: {parsed.name} (ID: {parsed.patient_id}) | **Age**: {parsed.age} | **Sex**: {parsed.sex.title()}
**Reported Symptoms**: {", ".join(active) or "None"}
**Symptom Risk Score**: {score.risk_score}/100 → **{score.severity_level.upper()} Severity**
**Estimated Brain Tumor Probability**: **{assessment.cancer_probability}%** → **{assessment.risk_category.upper()} Risk**

**Primary Clinical Concern**: {", ".join(assessment.primary_concerns)}

**Recommended Next Steps**:
{chr(10).join(f"• {t}" for t in assessment.recommended_tests)}
"""

        try:
            polished = self.llm.invoke(
                f"Rewrite this medical report to be warm, empathetic, clear and professional. "
                f"Keep all numbers and facts exactly the same:\n\n{base_text}"
            ).content.strip()
            return polished
        except Exception as e:
            print(f"   [ExplainAgent] Using fallback text")
            return base_text.strip()