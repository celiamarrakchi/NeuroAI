import torch
import torch.nn as nn
import numpy as np
import faiss
from PIL import Image
import torchvision.transforms as transforms
import sys
from pymongo import MongoClient
import certifi
import time
import json
from datetime import datetime
import google.generativeai as genai
import warnings
from typing import Optional, List, Dict, Any
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import StructuredTool
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

warnings.filterwarnings('ignore')

#sys.path.append(r"C:\Users\USER\MedViT")
sys.path.append(r"C:\projet_4IA2_sem1\ApplicationECM\ApplicationECM\ApplicationECM\MedViT")
from MedViT import MedViT_small

# =====================================================================
# CONTEXTE GLOBAL PARTAGÉ
# =====================================================================

class MedicalContext:
    """Contexte partagé entre les agents"""
    def __init__(self):
        self.query_image_path: Optional[str] = None
        self.query_vector: Optional[np.ndarray] = None
        self.faiss_indices: Optional[List[int]] = None
        self.reference_documents: List[Dict] = []
        self.reference_descriptions: List[Dict] = []
        self.query_description: Optional[str] = None
        self.final_report: Optional[Dict] = None
    
    def reset(self):
        self.__init__()

# Instance globale
CTX = MedicalContext()

# =====================================================================
# PYDANTIC MODELS POUR FORMAT JSON FINAL
# =====================================================================

class ROIFinding(BaseModel):
    """Région d'intérêt trouvée dans l'IRM"""
    region: str = Field(description="Localisation anatomique (ex: left temporal lobe)")
    type: str = Field(description="Type de finding (mass/edema/hemorrhage/tumor/normal)")
    size_mm: float = Field(description="Taille en millimètres")
    intensity_pattern: str = Field(description="Pattern IRM (ex: T1 hypointense, T2 hyperintense)")
    confidence: float = Field(description="Confiance (0-1)", ge=0, le=1)

class MedicalReport(BaseModel):
    """Rapport médical final (FORMAT SANS VECTEUR)"""
    roi_findings: List[ROIFinding] = Field(description="Régions d'intérêt identifiées")
    radiologist_report: str = Field(description="Rapport textuel détaillé du radiologue")
    visual_summary: str = Field(description="Résumé clinique structuré")
    diagnosis: str = Field(description="Diagnostic principal")
    clinical_significance: str = Field(description="Signification clinique")

# =====================================================================
# OUTILS MÉDICAUX POUR LES AGENTS
# =====================================================================

class MedicalTools:
    """Outils utilisés par les agents LangChain"""
    
    def __init__(self, model, device, collection, index_path, gemini_api_key):
        self.model = model
        self.device = device
        self.collection = collection
        self.index_path = index_path
        genai.configure(api_key=gemini_api_key)
        self.gemini = genai.GenerativeModel("gemini-2.5-flash-preview-09-2025")
        self.safety = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
    
    def extract_and_search(self, image_path: str) -> str:
        """
        OUTIL 1: Extraction vecteur + Recherche FAISS combinée
        Input: chemin de l'image
        Output: message de succès avec les 5 indices trouvés
        """
        try:
            print(f"\n🔧 [TOOL 1] Extraction vecteur + Recherche FAISS")
            print(f"   Image: {image_path}")
            
            # Extraction vecteur MedViT
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            
            image = Image.open(image_path).convert('RGB')
            img_tensor = transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                x = self.model.stem(img_tensor)
                x = self.model.features(x)
                x = self.model.norm(x)
                x = self.model.avgpool(x)
                x = torch.flatten(x, 1)
                features = x.cpu().numpy()
            
            # Stocker dans contexte
            CTX.query_image_path = image_path
            CTX.query_vector = features
            
            print(f"   ✅ Vecteur extrait: {features.shape}")
            
            # Recherche FAISS
            index = faiss.read_index(self.index_path)
            distances, indices = index.search(features, 5)
            
            CTX.faiss_indices = indices[0].tolist()
            
            print(f"   ✅ FAISS: 5 cas similaires trouvés")
            print(f"      Indices: {CTX.faiss_indices}")
            
            return f"SUCCESS: Extracted vector (768D) and found 5 similar cases with FAISS IDs: {CTX.faiss_indices}"
            
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    def retrieve_and_describe(self, dummy: str = "none") -> str:
        """
        OUTIL 2: Récupération MongoDB + Génération descriptions
        Input: dummy (pas utilisé)
        Output: message avec nombre de descriptions récupérées
        """
        try:
            print(f"\n🔧 [TOOL 2] Récupération MongoDB + Descriptions")
            
            if not CTX.faiss_indices:
                return "ERROR: No FAISS indices available"
            
            # Récupération MongoDB
            matched_docs = []
            for idx in CTX.faiss_indices:
                doc = self.collection.find_one({"faiss_id": int(idx)})
                if doc:
                    matched_docs.append({
                        'faiss_id': doc['faiss_id'],
                        'image_name': doc['image_name'],
                        'label': doc['label'],
                        'image_path': doc['image_path'],
                        'image_data': doc.get('image_data'),
                        'description': doc.get('description')
                    })
            
            CTX.reference_documents = matched_docs
            print(f"   ✅ {len(matched_docs)} documents MongoDB récupérés")
            
            # Génération descriptions manquantes
            generated = 0
            for doc in matched_docs:
                if not doc.get('description') or not doc['description'].strip():
                    print(f"   🤖 Génération pour {doc['image_name']}...")
                    
                    #desc = self._generate_description(doc['image_path'], doc['label'])
                    image_source = doc.get('image_data') or doc.get('image_path')  # ← LIGNE MODIFIÉE
                    if not image_source:  # ← LIGNE AJOUTÉE
                        print(f"      ❌ Aucune source d'image pour {doc['image_name']}")
                        continue
                    desc = self._generate_description(image_source, doc['label'])  # ← Reste identique
                    
                    if desc:
                        # Mise à jour MongoDB
                        self.collection.update_one(
                            {"faiss_id": doc['faiss_id']},
                            {"$set": {
                                "description": desc,
                                "description_generated_at": datetime.now()
                            }}
                        )
                        doc['description'] = desc
                        generated += 1
                        print(f"      ✅ Description générée et sauvegardée")
                    
                    time.sleep(5)
            
            # Stocker descriptions finales
            CTX.reference_descriptions = [
                {
                    'image_name': d['image_name'],
                    'label': d['label'],
                    'description': d['description']
                }
                for d in matched_docs
                if d.get('description')
            ]
            
            print(f"   ✅ Descriptions disponibles: {len(CTX.reference_descriptions)}")
            
            return f"SUCCESS: Retrieved {len(matched_docs)} docs, generated {generated} new descriptions. Total ready: {len(CTX.reference_descriptions)}"
            
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    def analyze_query_and_generate_report(self, dummy: str = "none") -> str:
        """
        OUTIL 3: Analyse image d'entrée + Génération rapport final
        Input: dummy (pas utilisé)
        Output: message de succès avec nom du fichier
        """
        try:
            print(f"\n🔧 [TOOL 3] Analyse image d'entrée + Rapport final")
            
            if not CTX.query_image_path or not CTX.reference_descriptions:
                return "ERROR: Missing query image or reference descriptions"
            
            # Déterminer label majoritaire
            from collections import Counter
            labels = [d['label'] for d in CTX.reference_descriptions]
            majority_label = Counter(labels).most_common(1)[0][0] if labels else "notumor"
            
            # 1. Générer description de l'image d'entrée
            print(f"   🤖 Génération description image d'entrée...")
            query_desc = self._generate_description(CTX.query_image_path, majority_label)
            
            if not query_desc:
                return "ERROR: Failed to generate query description"
            
            CTX.query_description = query_desc
            print(f"   ✅ Description générée ({len(query_desc)} chars)")
            
            # 2. Générer rapport médical structuré (FORMAT SANS VECTEUR)
            print(f"   🤖 Génération rapport médical...")
            
            ref_context = "\n\n".join([
                f"REFERENCE {i+1} ({d['label']}):\n{d['description'][:400]}..."
                for i, d in enumerate(CTX.reference_descriptions)
            ])
            
            prompt = f"""
You are a senior neuroradiologist generating a structured medical report.

CRITICAL: This report is ONLY about the PRIMARY/QUERY scan. Reference cases are context ONLY.

═══════════════════════════════════════════════════════════════
PRIMARY SCAN (PATIENT'S IMAGE):
═══════════════════════════════════════════════════════════════
{query_desc}

REFERENCE DATABASE (Context only - DO NOT report on these):
═══════════════════════════════════════════════════════════════
{ref_context}

Output EXACTLY this JSON structure (NO VECTORS, NO EMBEDDINGS):

```json
{{
  "roi_findings": [
    {{
      "region": "specific anatomical location",
      "type": "PRIMARY DIAGNOSIS (e.g., Glioblastoma, Meningioma, Pituitary Adenoma)"",
      "size_mm": numeric_value,
      "intensity_pattern": "MRI characteristics",
      "confidence": float_0_to_1
    }}
  ],
  "radiologist_report": "Comprehensive professional report (250-400 words) with EXAMINATION, TECHNIQUE, FINDINGS, COMPARISON, IMPRESSION sections",
  "visual_summary": "Concise 3-5 sentence clinical summary",
  "diagnosis": "Primary diagnosis",
  "clinical_significance": "Clinical impact and urgency"
}}
```

REQUIREMENTS:
- Include 2-5 ROIs from PRIMARY scan
- Sizes in mm (5-80mm realistic range)
- All findings describe PRIMARY scan ONLY
- Reference cases = medical knowledge context
- Output ONLY valid JSON

Generate now:
"""
            
            for attempt in range(3):
                try:
                    response = self.gemini.generate_content(prompt, safety_settings=self.safety)
                    text = response.text.strip()
                    
                    # Extraire JSON
                    if "```json" in text:
                        text = text.split("```json")[1].split("```")[0].strip()
                    elif "```" in text:
                        text = text.split("```")[1].split("```")[0].strip()
                    
                    report = json.loads(text)
                    
                    # Valider structure
                    required = ['roi_findings', 'radiologist_report', 'visual_summary']
                    if all(k in report for k in required):
                        CTX.final_report = report
                        
                        # Sauvegarder
                        #filename = f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        #with open(filename, 'w', encoding='utf-8') as f:
                            #json.dump(report, f, indent=2, ensure_ascii=False)
                        
                        #print(f"   ✅ Rapport généré et sauvegardé: {filename}")
                        #return f"SUCCESS: Report generated and saved as {filename}"
                    else:
                        return f"ERROR: Missing required fields in report"
                        
                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower():
                        wait = 60 * (attempt + 1)
                        print(f"   ⚠️ Rate limit (attempt {attempt+1}/3) - waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        if attempt < 2:
                            time.sleep(30)
                        else:
                            return f"ERROR: {str(e)}"
            
            return "ERROR: Failed after 3 attempts"
            
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    def _generate_description(self, image_source: str, label: str) -> Optional[str]:
        import base64
        from io import BytesIO
        """Helper pour générer une description avec Gemini"""
        time.sleep(4)
        
        context = {
            "notumor": "Normal Brain MRI - describe normal structures",
            "glioma": "Glioma - describe infiltrative tumor, irregular borders, edema",
            "meningioma": "Meningioma - describe extra-axial tumor, well-defined borders",
            "pituitary": "Pituitary Tumor - describe sellar mass, optic chiasm relation"
        }.get(label, "Normal Brain MRI")
        
        prompt = f"""
You are an expert neuroradiologist.

DIAGNOSIS: {context}

Provide a comprehensive medical description (300-500 words) covering:
1. Imaging characteristics (4-6 sentences)
2. Visual findings (6-10 sentences)
3. Pathological analysis (5-8 sentences)
4. Clinical correlation (4-5 sentences)
5. Radiological impression (3-4 sentences)

Output format:
```json
{{
  "description": "A flowing narrative combining all sections into one professional report."
}}
```
"""
        
        try:
            # Auto-détection du type d'input
            if isinstance(image_source, str):
                if len(image_source) > 500:  # C'est probablement du base64
                    image_data = base64.b64decode(image_source)
                    image = Image.open(BytesIO(image_data))
                else:  # C'est un chemin de fichier
                    image = Image.open(image_source)
            else:  # C'est déjà une image PIL
                image = image_source
            
            response = self.gemini.generate_content([prompt, image], safety_settings=self.safety)
            text = response.text.strip()
            
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(text)
            return result.get("description")
        except Exception as e:
            print(f"      ❌ Error: {e}")
            return None

# =====================================================================
# AGENT PRINCIPAL
# =====================================================================

def create_medical_agent(tools: MedicalTools, llm) -> AgentExecutor:
    """
    Agent unique qui orchestre tout le pipeline
    """
    
    agent_tools = [
        StructuredTool.from_function(
            func=tools.extract_and_search,
            name="extract_and_search",
            description="Extract MedViT vector from image and search 5 similar cases in FAISS. Input: image_path (string)"
        ),
        StructuredTool.from_function(
            func=tools.retrieve_and_describe,
            name="retrieve_and_describe",
            description="Retrieve MongoDB documents and generate missing descriptions. Input: none (use 'none')"
        ),
        StructuredTool.from_function(
            func=tools.analyze_query_and_generate_report,
            name="analyze_and_report",
            description="Analyze query image and generate final JSON report. Input: none (use 'none')"
        )
    ]
    
    template = """You are a MEDICAL RAG AGENT that processes brain MRI images.Your task is to execute **EXACTLY three steps in this strict order (1 -> 2 -> 3)** and never repeat a step that has been successfully completed.

Your mission:
1. Extract vector from query image and search 5 similar cases (use extract_and_search)
2. Retrieve MongoDB documents and generate descriptions for references (use retrieve_and_describe)
3. Analyze query image and generate final JSON report (use analyze_and_report)

Available tools:
{tools}

Tool names: {tool_names}

STRICT EXECUTION FORMAT:
Question: the input question
Thought: I must execute Step 1: extract vector and search 5 similar cases.
Action: extract_and_search
Action Input: path/to/image.jpg
Observation: result of Step 1
Thought: Step 1 is complete. I must now execute Step 2: retrieve MongoDB documents and generate descriptions.
Action: retrieve_and_describe
Action Input: none
Observation: result of Step 2
Thought: Step 2 is complete. I I must now execute the final Step 3: analyze query image and generate final JSON report.
Action: analyze_and_report
Action Input: none
Observation: SUCCESS: Report generated and saved as medical_report_....json 
# --- KEY MODIFICATION HERE ---
Thought: Step 3 is completed successfully. The report has been generated and saved. I MUST STOP the chain and provide the Final Answer now.
Final Answer: The medical RAG process is complete. The final JSON report has been successfully generated and saved.
Question: {input}

{agent_scratchpad}"""
    
    prompt = PromptTemplate.from_template(template)
    agent = create_react_agent(llm, agent_tools, prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=agent_tools,
        verbose=True,
        max_iterations=15,
        handle_parsing_errors=True
    )

# =====================================================================
# ORCHESTRATEUR
# =====================================================================

class MedicalOrchestrator:
    """Orchestrateur simplifié avec 1 agent"""
    
    def __init__(self, tools: MedicalTools, gemini_api_key: str):
        self.tools = tools
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-09-2025",
            google_api_key=gemini_api_key,
            temperature=0.1,
            convert_system_message_to_human=True
        )
        self.agent = create_medical_agent(tools, self.llm)
    
    def process(self, image_path: str) -> Dict[str, Any]:
        """
        Pipeline complet en une seule exécution
        """
        print(f"\n{'='*70}")
        print("🚀 MEDICAL RAG PIPELINE - LANGCHAIN AGENT")
        print(f"{'='*70}")
        print(f"Image: {image_path}\n")
        
        CTX.reset()
        
        agent_input = f"""
Process this brain MRI image and generate a medical report:
Image path: {image_path}

Execute these steps in order:
1. Extract vector and search 5 similar cases in FAISS
2. Retrieve MongoDB documents and generate descriptions
3. Analyze the query image and create final JSON report (WITHOUT vectors)

Provide confirmation when complete.
"""
        
        try:
            result = self.agent.invoke({"input": agent_input})
            
            print(f"\n{'='*70}")
            print("✅ AGENT TERMINÉ")
            print(f"{'='*70}")
            print(f"Output: {result.get('output', 'No output')}\n")
            
            if CTX.final_report:
                self._display_report()
                return {
                    "success": True,
                    "query_image": image_path,
                    "reference_cases": len(CTX.reference_descriptions),
                    "report": CTX.final_report
                }
            else:
                return {"error": "Report not generated"}
                
        except Exception as e:
            return {"error": f"Agent failed: {str(e)}"}
    
    def _display_report(self):
        """Affiche le rapport de manière structurée"""
        report = CTX.final_report
        
        print(f"\n{'='*70}")
        print("📋 RAPPORT MÉDICAL FINAL (FORMAT SANS VECTEUR)")
        print(f"{'='*70}\n")
        
        print("🔍 RÉGIONS D'INTÉRÊT (ROI):")
        for i, roi in enumerate(report.get('roi_findings', []), 1):
            print(f"\n  ═══ ROI {i} ═══")
            print(f"  📍 Région: {roi.get('region')}")
            print(f"  🏷️  Type: {roi.get('type')}")
            print(f"  📏 Taille: {roi.get('size_mm')} mm")
            print(f"  🎨 Pattern: {roi.get('intensity_pattern')}")
            print(f"  ✅ Confiance: {roi.get('confidence', 0):.1%}")
        
        print(f"\n\n📄 RAPPORT DU RADIOLOGUE:")
        print(f"{'─'*70}")
        print(report.get('radiologist_report', 'N/A'))
        print(f"{'─'*70}")
        
        print(f"\n💡 RÉSUMÉ VISUEL:")
        print(f"{'─'*70}")
        print(report.get('visual_summary', 'N/A'))
        print(f"{'─'*70}")
        
        print(f"\n🩺 DIAGNOSTIC: {report.get('diagnosis', 'N/A')}")
        print(f"\n⚕️ SIGNIFICATION CLINIQUE:")
        print(f"   {report.get('clinical_significance', 'N/A')}")

# =====================================================================
# INITIALISATION
# =====================================================================

def load_medvit_model():
    """Charge le modèle MedViT"""
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    model = MedViT_small(num_classes=len(classes))
    
    if hasattr(model, "proj_head") and isinstance(model.proj_head, nn.Linear):
        in_features = model.proj_head.in_features
        model.proj_head = nn.Linear(in_features, len(classes))
    
    device = torch.device("cpu")
    checkpoint = torch.load(
        'C:\\projet_4IA2_sem1\\ApplicationECM\\ApplicationECM\\ApplicationECM\\models\\best_medvit_model.pth',
        map_location=device
    )
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    print("✅ Modèle MedViT chargé")
    return model, device

def connect_mongodb():
    """Connexion à MongoDB"""
    uri = "mongodb+srv://unipath913_db_user:Ab7LiKL4e35DuGcC@cluster0.dsmntgc.mongodb.net/?retryWrites=true&w=majority"
    client = MongoClient(uri, tlsCAFile=certifi.where())
    
    try:
        client.admin.command('ping')
        print("✅ Connexion MongoDB réussie")
    except Exception as e:
        print(f"❌ Erreur MongoDB: {e}")
        exit()
    
    db = client['medvit_mri_db']
    collection = db['brain_tumor_vectors_v3']
    
    return client, collection

# =====================================================================
# MAIN
# =====================================================================
"""
if __name__ == "__main__":
    
    INDEX_PATH = "brain_tumor_faiss_v3.index"
    GEMINI_API_KEY = "AIzaSyBbwSneXNfIMszOH-lR5QXeBTcFelyIDoU"
    
    print("🔧 Initialisation...\n")
    
    # Charger composants
    model, device = load_medvit_model()
    mongo_client, collection = connect_mongodb()
    
    # Créer tools
    print("🛠️  Création des tools médicaux...")
    medical_tools = MedicalTools(
        model=model,
        device=device,
        collection=collection,
        index_path=INDEX_PATH,
        gemini_api_key=GEMINI_API_KEY
    )
    print("✅ Tools créés\n")
    
    # Créer orchestrateur
    print("🤖 Création de l'agent LangChaiC:n...")
    orchestrator = MedicalOrchestrator(
        tools=medical_tools,
        gemini_api_key=GEMINI_API_KEY
    )
    print("✅ Agent créé\n")
    
    # Exécuter pipeline
    try:
        result = orchestrator.process(IMAGE_PATH)
        
        if result.get("success"):
            print(f"\n{'='*70}")
            print("🎉 PIPELINE TERMINÉ AVEC SUCCÈS!")
            print(f"{'='*70}")
            print(f"\n📊 STATISTIQUES:")
            print(f"   • Image analysée: {result['query_image']}")
            print(f"   • Cas de référence: {result['reference_cases']}")
            print(f"   • ROI identifiées: {len(result['report'].get('roi_findings', []))}")
            print(f"   • Format: JSON sans vecteurs")
            
            # Afficher JSON final
            print(f"\n{'='*70}")
            print("📄 RAPPORT JSON FINAL")
            print(f"{'='*70}\n")
            print(json.dumps(result['report'], indent=2, ensure_ascii=False))
        else:
            print(f"\n❌ ERREUR: {result.get('error')}")
            
    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        mongo_client.close()
        print(f"\n{'='*70}")
        print("🔒 Connexion MongoDB fermée")
        print("="*70)
        print("\n🏁 PROGRAMME TERMINÉ\n")
 """