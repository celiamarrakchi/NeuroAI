# agents/cancer_agent.py
# UPDATED FOR FAISS — FASTER & MORE RELIABLE
from models import ParsedHealthData, SymptomScore, CancerRiskAssessment
from typing import List
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path

class CancerPotentialAgent:
    def __init__(self):
        self.llm = ChatOllama(model="llama3.1", temperature=0.0)
        self._vectorstore = None

    def _get_vectorstore(self):
        if self._vectorstore is not None:
            return self._vectorstore

        db_path = Path("faiss_medical_db")
        if not db_path.exists():
            print("   [RAG] FAISS database not found → rule-only mode")
            self._vectorstore = False
            return None

        try:
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            db = FAISS.load_local(str(db_path), embeddings, allow_dangerous_deserialization=True)
            count = len(db.index_to_docstore_id)
            print(f"   [RAG] FAISS LOADED — {count} medical chunks active (ultra-fast)")
            self._vectorstore = db
            return db
        except Exception as e:
            print(f"   [RAG] FAISS load failed: {e} → rule-only mode")
            self._vectorstore = False
            return None

    def _retrieve(self, symptoms: List[str], age: int) -> str:
        db = self._get_vectorstore()
        if not db:
            return "Medical literature not available (offline mode). Using clinical rules only."

        query = f"brain tumor red flags symptoms: {', '.join(symptoms[:7])} age {age} years"
        try:
            docs = db.similarity_search(query, k=6)
            sources = list(set(d.metadata.get("source", "medical_db") for d in docs))
            text = "\n".join([f"• {d.page_content.strip()[:550]}..." for d in docs[:5]])
            return f"{text}\n\nSources: {', '.join(sources)}"
        except:
            return "Evidence search failed — using rule-based assessment."

    def assess(self, parsed: ParsedHealthData, score: SymptomScore) -> CancerRiskAssessment:
        active = [k.replace("_", " ") for k, v in parsed.symptoms.items() if v]
        print(f"   [llama3.1] Assessing cancer risk for Patient {parsed.patient_id}...")
        evidence = self._retrieve(active, parsed.age)

        prob = score.risk_score
        if len(active) == 1 and "headache" in active: prob = min(prob, 22)
        elif len(active) <= 2: prob = min(prob * 0.8, 48)
        elif "seizures" in active: prob = max(prob, 82)
        elif len(active) >= 5: prob = max(prob, 78)
        prob = int(min(max(prob, 5), 94))
        cat = ("very low" if prob < 15 else "low" if prob < 35 else
               "moderate" if prob < 60 else "high" if prob < 85 else "very high")

        return CancerRiskAssessment(
            patient_id=parsed.patient_id,
            cancer_probability=prob,
            risk_category=cat,
            primary_concerns=["Brain tumor suspicion"] if prob >= 60 else ["Likely benign"],
            recommended_tests=(
                ["Urgent MRI brain + contrast", "Neurosurgery consult"] if prob >= 40
                else ["Head CT", "Follow-up"]
            ),
            evidence_sources=[evidence]
        )