# agents/parser_agent.py
from models import ParsedHealthData

class ParserAgent:
    SYMPTOMS = {"headache", "nausea", "vomiting", "seizures", "vision_problems",
                "balance_issues", "memory_problems", "speech_difficulties", "weakness"}

    def parse(self, raw: dict) -> ParsedHealthData:
        print(f"   [ParserAgent] Parsing Patient {raw.get('id', '?')}...")
        clean = {k.lower().replace(" ", "_").replace("-", "_"): str(v).strip().lower()
                 for k, v in raw.items()}
        symptoms = {s: clean.get(s, "no") in {"yes", "true", "1", "y", "present"} for s in self.SYMPTOMS}
        age = next((int(v) for k, v in clean.items() if k == "age" and v.isdigit()), 40)
        age = max(0, min(120, age))
        sex = "male" if any(x in clean.get("sex", "") for x in ["m", "male"]) else "female" if any(x in clean.get("sex", "") for x in ["f", "female"]) else "unknown"
        name = str(raw.get("name", "unknown")).strip() or "unknown"
        return ParsedHealthData(
            patient_id=int(raw.get("id", 0)),
            name=name,
            age=age,
            sex=sex,
            symptoms=symptoms,
            raw_data=raw
        )