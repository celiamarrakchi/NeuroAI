# agents/scorer_agent.py
from models import ParsedHealthData, SymptomScore

class SymptomScorerAgent:
    WEIGHTS = {"headache": 8, "nausea": 6, "vomiting": 7, "seizures": 35,
               "vision_problems": 23, "balance_issues": 20, "memory_problems": 25,
               "speech_difficulties": 26, "weakness": 22}

    def score(self, p: ParsedHealthData) -> SymptomScore:
        print(f"   [ScorerAgent] Scoring Patient {p.patient_id}...")
        active = [s for s, v in p.symptoms.items() if v]
        n = len(active)
        base = sum(self.WEIGHTS.get(s, 5) for s in active) * 1.8
        if n <= 1: base = min(base * 0.3, 30)
        bonus = 40 if "seizures" in active else 0
        bonus += 25 if n >= 5 else 0
        score_val = min(int(base + bonus), 100)
        if n <= 1: score_val = min(score_val, 38)
        severity = "low" if score_val < 26 else "moderate" if score_val < 51 else "high" if score_val < 76 else "critical"
        alerts = ["New-onset seizure"] if "seizures" in active else []
        if n >= 4: alerts.append("Progressive neurological deficits")
        return SymptomScore(patient_id=p.patient_id, risk_score=score_val, severity_level=severity,
                             total_symptoms=n, red_flag_alerts=alerts)