from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
import os
import json
import base64
from io import BytesIO
from datetime import datetime
import sys
import torch
import numpy as np
import cv2
import secrets
import tempfile
from pathlib import Path

# Import your existing code
sys.path.append(r"MedViT")
from agents.medical_agent import (
    load_medvit_model,
    connect_mongodb,
    MedicalTools,
    MedicalOrchestrator,
    CTX
)

# Import the segmentation agent
from agents.segmentation_agent import SegmentationAgent

# Import the XAI agent
from agents.xai_agent import XAIAgent
from agent import HealthDataAgent
from database import Database
from whisper_service import transcribe_audio_file

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}
app.secret_key = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(16))

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for the medical system
MODEL = None
DEVICE = None
COLLECTION = None
MONGO_CLIENT = None
ORCHESTRATOR = None

# Global variables for additional agents
SEGMENTATION_AGENT = None
XAI_AGENT = None

# Store last segmentation results for XAI
LAST_SEGMENTATION = {
    'image_tensor': None,
    'prediction_mask': None,
    'seg_model': None
}

# Global resources for the consultation assistant
CONSULT_SESSION_KEY = 'consult_session_id'
CONSULT_DATA_KEY = 'consult_collected_data'
HEALTH_DB_PATH = os.path.join(os.path.dirname(__file__), 'health_data.json')
HEALTH_DB = Database(db_path=HEALTH_DB_PATH)
HEALTH_AGENT_API_KEY = os.getenv('HEALTH_AGENT_GROQ_API_KEY')
HEALTH_AGENT = (
    HealthDataAgent(api_key=HEALTH_AGENT_API_KEY)
    if HEALTH_AGENT_API_KEY
    else HealthDataAgent()
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def initialize_medical_system():
    """Initialize the complete medical system at startup"""
    global MODEL, DEVICE, COLLECTION, MONGO_CLIENT, ORCHESTRATOR
    global SEGMENTATION_AGENT
    
    print("\n" + "="*70)
    print("🏥 INITIALIZING MULTI-AGENT MEDICAL SYSTEM")
    print("="*70 + "\n")
    
    # Configuration
    INDEX_PATH = "models/brain_tumor_faiss_v3.index"
    GEMINI_API_KEY = ""
    SEGMENTATION_MODEL_PATH = "models/best_model.pth"
    LLAMA_API_KEY = None

    #remove
    
    try:
        # 1. Load MedViT
        print("🔧 Loading MedViT model...")
        MODEL, DEVICE = load_medvit_model()
        
        # 2. Connect to MongoDB
        print("🔧 Connecting to MongoDB...")
        MONGO_CLIENT, COLLECTION = connect_mongodb()
        
        # 3. Create medical tools
        print("🛠️  Creating medical tools...")
        medical_tools = MedicalTools(
            model=MODEL,
            device=DEVICE,
            collection=COLLECTION,
            index_path=INDEX_PATH,
            gemini_api_key=GEMINI_API_KEY
        )
        
        # 4. Create main orchestrator
        print("🤖 Creating main orchestrator...")
        ORCHESTRATOR = MedicalOrchestrator(
            tools=medical_tools,
            gemini_api_key=GEMINI_API_KEY
        )
        
        # 5. Initialize Segmentation Agent
        print("🧠 Initializing Segmentation Agent...")
        SEGMENTATION_AGENT = SegmentationAgent(
            model_path=SEGMENTATION_MODEL_PATH,
            device=str(DEVICE),
            llama_api_key=LLAMA_API_KEY
        )
        
        # Note: XAI Agent is initialized on-demand after segmentation
        print("🔍 XAI Agent will be initialized on-demand after segmentation")
        
        print("\n✅ ALL AGENTS INITIALIZED SUCCESSFULLY!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ INITIALIZATION FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def numpy_to_base64(array):
    """Convert numpy array to base64 string for JSON transmission"""
    if array is None:
        return None
    
    # Normalize to 0-255 range
    if array.max() <= 1.0:
        array = (array * 255).astype(np.uint8)
    else:
        array = array.astype(np.uint8)
    
    # Encode as PNG
    _, buffer = cv2.imencode('.png', array)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

# =====================================================================
# ROUTES HTML
# =====================================================================

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/virtual-consultation')
def consultation():
    """Virtual consultation page"""
    return render_template('virtual-consultation.html')

@app.route('/mri-analysis')
def mri_analysis():
    """MRI analysis page"""
    return render_template('mri-analysis.html')

@app.route('/segmentation')
def segmentation_page():
    """Tumor segmentation page"""
    return render_template('segmentation.html')

@app.route('/explainability')
def explainability_page():
    """XAI explainability page"""
    return render_template('explainability.html')

@app.route('/consultation-report')
def consultation_report_page():
    """Consultation report page"""
    return render_template('consultation-report.html')

# =====================================================================
# API ENDPOINTS - REAL-TIME CONSULTATION ASSISTANT
# =====================================================================


@app.route('/api/consultation/start_session', methods=['POST'])
def start_consultation_session():
    """Initialize a new consultation chat session."""
    session_id = secrets.token_hex(8)
    session[CONSULT_SESSION_KEY] = session_id
    session[CONSULT_DATA_KEY] = {}

    welcome_message = HEALTH_AGENT.start_conversation()

    return jsonify({
        'session_id': session_id,
        'message': welcome_message,
        'collected_data': session[CONSULT_DATA_KEY]
    })


@app.route('/api/consultation/chat', methods=['POST'])
def consultation_chat():
    """Handle chat messages for the consultation assistant."""
    if CONSULT_SESSION_KEY not in session:
        return jsonify({'error': 'No active session'}), 400

    data = request.get_json(silent=True) or {}
    user_message = data.get('message', '').strip()

    if not user_message:
        return jsonify({'error': 'Message cannot be empty'}), 400

    collected_data = session.get(CONSULT_DATA_KEY, {})

    response, extracted_data, is_complete = HEALTH_AGENT.process_message(
        user_message,
        collected_data
    )

    if extracted_data:
        collected_data.update(extracted_data)
        session[CONSULT_DATA_KEY] = collected_data

    if is_complete:
        user_id = HEALTH_DB.save_user_data(collected_data)
        saved_user = HEALTH_DB.get_user_by_id(user_id)
        return jsonify({
            'message': response,
            'complete': True,
            'user_id': user_id,
            'collected_data': collected_data,
            'saved_user': saved_user
        })

    return jsonify({
        'message': response,
        'complete': False,
        'collected_data': collected_data
    })


@app.route('/api/consultation/users', methods=['GET'])
def consultation_users():
    """Return all stored consultation users."""
    users = HEALTH_DB.get_all_users()
    return jsonify({'users': users})


@app.route('/api/consultation/users/<int:user_id>', methods=['GET'])
def consultation_user(user_id):
    """Return a single user record."""
    user = HEALTH_DB.get_user_by_id(user_id)
    if user:
        return jsonify({'user': user})
    return jsonify({'error': 'User not found'}), 404


@app.route('/api/consultation/transcribe', methods=['POST'])
def consultation_transcribe():
    """Transcribe an audio/video recording using Groq Whisper."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    content_type = getattr(audio_file, 'content_type', None) or 'audio/webm'
    suffix = '.webm'
    if 'wav' in content_type:
        suffix = '.wav'
    elif 'mp3' in content_type:
        suffix = '.mp3'
    elif 'm4a' in content_type:
        suffix = '.m4a'

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
            audio_file.save(temp_audio.name)
            temp_path = temp_audio.name

        text = transcribe_audio_file(temp_path, language='en')

        return jsonify({
            'transcription': text,
            'success': True
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


@app.route('/api/consultation/report/<int:user_id>', methods=['GET'])
def get_consultation_report(user_id):
    """
    Récupère le rapport Markdown d'un patient
    """
    try:
        reports_dir = Path("reports")
        md_path = reports_dir / f"patient_{user_id}_REPORT.md"
        json_path = reports_dir / f"rapport_patient_{user_id}.json"
        
        if not md_path.exists():
            return jsonify({
                'error': 'Report not found. Please run analysis first.',
                'success': False
            }), 404
        
        # Lire le rapport Markdown
        markdown_content = md_path.read_text(encoding='utf-8')
        
        # Lire aussi le JSON pour les métadonnées
        metadata = {}
        if json_path.exists():
            metadata = json.loads(json_path.read_text(encoding='utf-8'))

        # Compléter les métadonnées manquantes avec la base consultation
        patient_record = HEALTH_DB.get_user_by_id(user_id)
        if patient_record:
            patient_info = metadata.get('patient_info', {}) or {}
            # Priorité à ce qui est déjà dans le rapport, sinon on complète avec la base
            patient_info.setdefault('name', patient_record.get('name') or patient_record.get('Name') or "unknown")
            patient_info.setdefault('age', patient_record.get('age'))
            patient_info.setdefault('sex', patient_record.get('sex'))
            metadata['patient_info'] = patient_info
        
        return jsonify({
            'success': True,
            'user_id': user_id,
            'markdown': markdown_content,
            'metadata': metadata,
            'patient': patient_record,
            'files': {
                'markdown': str(md_path),
                'json': str(json_path)
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': f'Error loading report: {str(e)}',
            'success': False
        }), 500


@app.route('/api/consultation/analyze', methods=['POST'])
def analyze_consultation_report():
    """
    Endpoint pour analyser les données de consultation avec le pipeline médical.
    Exécute le script run_consultation_analysis.py dans venv_consult (environnement séparé)
    """
    import subprocess
    
    try:
        data = request.get_json(silent=True) or {}
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({'error': 'user_id is required', 'success': False}), 400
        
        # Récupérer les données utilisateur depuis la base de données
        user_data = HEALTH_DB.get_user_by_id(user_id)
        
        if not user_data:
            return jsonify({'error': 'User not found', 'success': False}), 404
        
        print(f"\n{'='*70}")
        print(f"🔬 ANALYZING CONSULTATION FOR USER {user_id}")
        print(f"{'='*70}\n")
        
        # Chemin vers l'interpréteur Python du venv_consult
        venv_consult_python = Path("venv_consult/Scripts/python.exe")
        
        if not venv_consult_python.exists():
            return jsonify({
                'error': 'venv_consult not found. Please create it first.',
                'success': False
            }), 500
        
        # Appeler le script d'analyse dans venv_consult
        print("🔄 Calling analysis script in venv_consult...")
        
        result = subprocess.run(
            [str(venv_consult_python), 'run_consultation_analysis.py', json.dumps(user_data, ensure_ascii=False)],
            capture_output=True,
            text=False,  # Utiliser bytes au lieu de text
            cwd=os.getcwd()
        )
        
        # Parser le résultat JSON
        if result.returncode != 0:
            try:
                error_msg = result.stderr.decode('utf-8', errors='replace') if result.stderr else \
                           result.stdout.decode('utf-8', errors='replace') if result.stdout else "Unknown error"
            except:
                error_msg = "Encoding error in subprocess output"
            print(f"❌ Subprocess error: {error_msg}")
            return jsonify({
                'error': f'Analysis subprocess failed: {error_msg}',
                'success': False
            }), 500
        
        try:
            # Décoder stdout avec gestion d'erreurs
            stdout_str = result.stdout.decode('utf-8', errors='replace') if result.stdout else ""
            if not stdout_str:
                raise ValueError("Empty output from subprocess")
            
            # Le script affiche des logs, puis le JSON sur la dernière ligne
            # On prend seulement la dernière ligne non-vide
            lines = [line.strip() for line in stdout_str.split('\n') if line.strip()]
            if not lines:
                raise ValueError("No output lines from subprocess")
            
            json_line = lines[-1]  # Dernière ligne = JSON
            print(f"📄 JSON output: {json_line[:200]}...")  # Debug
            
            analysis_result = json.loads(json_line)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"❌ JSON decode error: {e}")
            print(f"Stdout (raw): {result.stdout[:500] if result.stdout else 'None'}")
            return jsonify({
                'error': f'Invalid JSON response from analysis: {str(e)}',
                'success': False
            }), 500
        
        if not analysis_result.get('success'):
            return jsonify(analysis_result), 500
        
        print(f"\n✅ ANALYSIS COMPLETED")
        print(f"Risk: {analysis_result['analysis']['cancer_probability']}% - {analysis_result['analysis']['risk_category'].upper()}")
        print(f"Reports saved in /reports folder\n")
        
        return jsonify(analysis_result), 200
        
    except Exception as e:
        print(f"\n❌ ANALYSIS ERROR: {str(e)}\n")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'error': f'Analysis error: {str(e)}',
            'success': False
        }), 500

# =====================================================================
# API ENDPOINTS - MAIN ORCHESTRATOR
# =====================================================================

@app.route('/api/analyze', methods=['POST'])
def analyze_mri():
    """
    Endpoint for comprehensive MRI analysis using main orchestrator
    """
    try:
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image file provided',
                'success': False
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'success': False
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type. Only JPG/JPEG/PNG allowed',
                'success': False
            }), 400
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        print(f"\n{'='*70}")
        print(f"📥 NEW MRI ANALYSIS REQUEST")
        print(f"{'='*70}")
        print(f"File: {unique_filename}")
        print(f"Path: {filepath}\n")
        
        if ORCHESTRATOR is None:
            return jsonify({
                'error': 'Medical system not initialized',
                'success': False
            }), 503
        
        # Reset context
        CTX.reset()
        
        # Process with main orchestrator
        result = ORCHESTRATOR.process(filepath)
        
        if result.get("success"):
            print(f"\n{'='*70}")
            print("✅ ANALYSIS COMPLETED SUCCESSFULLY")
            print(f"{'='*70}\n")
            
            return jsonify({
                'success': True,
                'report': result['report'],
                'query_image': unique_filename,
                'reference_cases': result['reference_cases'],
                'timestamp': datetime.now().isoformat()
            }), 200
        else:
            print(f"\n❌ ANALYSIS FAILED: {result.get('error')}\n")
            return jsonify({
                'error': result.get('error', 'Analysis failed'),
                'success': False
            }), 500
            
    except Exception as e:
        print(f"\n❌ ERROR IN ANALYSIS: {str(e)}\n")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'error': f'Server error: {str(e)}',
            'success': False
        }), 500

# =====================================================================
# API ENDPOINTS - SEGMENTATION AGENT
# =====================================================================

@app.route('/api/segment', methods=['POST'])
def segment_tumor():
    """
    Endpoint for tumor segmentation using SegmentationAgent
    """
    try:
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image file provided',
                'success': False
            }), 400
        
        file = request.files['image']
        use_agent = request.form.get('use_agent', 'false').lower() == 'true'
        
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'success': False
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type',
                'success': False
            }), 400
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        print(f"\n{'='*70}")
        print(f"🧠 NEW SEGMENTATION REQUEST")
        print(f"{'='*70}")
        print(f"File: {unique_filename}")
        print(f"Use Agent: {use_agent}\n")
        
        if SEGMENTATION_AGENT is None:
            return jsonify({
                'error': 'Segmentation agent not initialized',
                'success': False
            }), 503
        
        # Load and preprocess image
        from PIL import Image
        from torchvision import transforms
        
        img = Image.open(filepath).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        # Run segmentation
        result = SEGMENTATION_AGENT.run(image_tensor, use_agent=use_agent)
        
        # Store results for XAI agent
        LAST_SEGMENTATION['image_tensor'] = image_tensor
        LAST_SEGMENTATION['prediction_mask'] = result.get('prediction_mask')
        LAST_SEGMENTATION['seg_model'] = SEGMENTATION_AGENT.seg_model
        
        # Convert mask to base64 for visualization
        mask_base64 = None
        if result.get('prediction_mask') is not None:
            mask_base64 = numpy_to_base64(result['prediction_mask'])
        
        print(f"\n✅ SEGMENTATION COMPLETED")
        print(f"Metrics: {result['metrics']}\n")
        
        return jsonify({
            'success': True,
            'agent_output': result['agent_output'],
            'metrics': result['metrics'],
            'mask_shape': list(result['prediction_mask'].shape) if result.get('prediction_mask') is not None else None,
            'mask_base64': mask_base64,
            'query_image': unique_filename,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        print(f"\n❌ SEGMENTATION ERROR: {str(e)}\n")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'error': f'Segmentation error: {str(e)}',
            'success': False
        }), 500

# =====================================================================
# API ENDPOINTS - XAI AGENT
# =====================================================================

@app.route('/api/explain', methods=['POST'])
def explain_segmentation():
    """
    Endpoint for XAI explanation of segmentation results
    Requires that segmentation was run first
    """
    try:
        print(f"\n{'='*70}")
        print(f"🔍 NEW XAI EXPLANATION REQUEST")
        print(f"{'='*70}\n")
        
        # Check if we have segmentation results
        if LAST_SEGMENTATION['prediction_mask'] is None:
            return jsonify({
                'error': 'No segmentation results available. Please run segmentation first.',
                'success': False
            }), 400
        
        # Initialize XAI Agent with last segmentation results
        print("🔍 Initializing XAI Agent...")
        xai_agent = XAIAgent(
            seg_model=LAST_SEGMENTATION['seg_model'],
            image_tensor=LAST_SEGMENTATION['image_tensor'],
            prediction_mask=LAST_SEGMENTATION['prediction_mask']
        )
        
        # Run XAI analysis
        print("🔍 Generating explanations...")
        xai_results = xai_agent.run()
        
        # Convert heatmap to base64
        gradcam_base64 = None
        if xai_results.get('gradcam_heatmap') is not None:
            gradcam_base64 = numpy_to_base64(xai_results['gradcam_heatmap'])
        
        print(f"\n✅ XAI EXPLANATION COMPLETED")
        print(f"Alignment Score: {xai_results['attention_summary']['alignment_score']}%\n")
        
        return jsonify({
            'success': True,
            'gradcam_summary': xai_results['gradcam_summary'],
            'attention_summary': xai_results['attention_summary'],
            'explanation': xai_results['explanation'],
            'gradcam_base64': gradcam_base64,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        print(f"\n❌ XAI EXPLANATION ERROR: {str(e)}\n")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'error': f'XAI explanation error: {str(e)}',
            'success': False
        }), 500

# =====================================================================
# API ENDPOINTS - COMBINED SEGMENTATION + XAI
# =====================================================================

@app.route('/api/segment-and-explain', methods=['POST'])
def segment_and_explain():
    """
    Combined endpoint: Run segmentation and XAI explanation in one request
    """
    try:
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image file provided',
                'success': False
            }), 400
        
        file = request.files['image']
        use_agent = request.form.get('use_agent', 'false').lower() == 'true'
        
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'success': False
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type',
                'success': False
            }), 400
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        print(f"\n{'='*70}")
        print(f"🧠🔍 SEGMENTATION + XAI REQUEST")
        print(f"{'='*70}")
        print(f"File: {unique_filename}\n")
        
        if SEGMENTATION_AGENT is None:
            return jsonify({
                'error': 'Segmentation agent not initialized',
                'success': False
            }), 503
        
        # Load and preprocess image
        from PIL import Image
        from torchvision import transforms
        
        img = Image.open(filepath).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        # 1. Run segmentation
        print("1️⃣  Running segmentation...")
        seg_result = SEGMENTATION_AGENT.run(image_tensor, use_agent=use_agent)
        
        # 2. Run XAI explanation
        print("2️⃣  Generating XAI explanation...")
        xai_agent = XAIAgent(
            seg_model=SEGMENTATION_AGENT.seg_model,
            image_tensor=image_tensor,
            prediction_mask=seg_result.get('prediction_mask')
        )
        xai_result = xai_agent.run()
        
        # Convert visualizations to base64
        mask_base64 = numpy_to_base64(seg_result.get('prediction_mask'))
        gradcam_base64 = numpy_to_base64(xai_result.get('gradcam_heatmap'))
        
        print(f"\n✅ COMBINED ANALYSIS COMPLETED\n")
        
        return jsonify({
            'success': True,
            'segmentation': {
                'agent_output': seg_result['agent_output'],
                'metrics': seg_result['metrics'],
                'mask_base64': mask_base64
            },
            'xai': {
                'gradcam_summary': xai_result['gradcam_summary'],
                'attention_summary': xai_result['attention_summary'],
                'explanation': xai_result['explanation'],
                'gradcam_base64': gradcam_base64
            },
            'query_image': unique_filename,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        print(f"\n❌ COMBINED ANALYSIS ERROR: {str(e)}\n")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'error': f'Combined analysis error: {str(e)}',
            'success': False
        }), 500

# =====================================================================
# API ENDPOINTS - COMPREHENSIVE MULTI-AGENT ANALYSIS
# =====================================================================

@app.route('/api/comprehensive-analysis', methods=['POST'])
def comprehensive_analysis():
    """
    Endpoint that coordinates all agents for comprehensive analysis
    """
    try:
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image file provided',
                'success': False
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'success': False
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type',
                'success': False
            }), 400
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        print(f"\n{'='*70}")
        print(f"🔬 COMPREHENSIVE MULTI-AGENT ANALYSIS")
        print(f"{'='*70}")
        print(f"File: {unique_filename}\n")
        
        results = {}
        
        # 1. Main Orchestrator Analysis
        print("1️⃣  Running main orchestrator...")
        if ORCHESTRATOR:
            CTX.reset()
            results['main_analysis'] = ORCHESTRATOR.process(filepath)
        
        # 2. Segmentation Analysis
        print("2️⃣  Running segmentation agent...")
        if SEGMENTATION_AGENT:
            from PIL import Image
            from torchvision import transforms
            
            img = Image.open(filepath).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            image_tensor = transform(img).unsqueeze(0).to(DEVICE)
            seg_result = SEGMENTATION_AGENT.run(image_tensor, use_agent=True)
            
            # 3. XAI Analysis
            print("3️⃣  Running XAI agent...")
            xai_agent = XAIAgent(
                seg_model=SEGMENTATION_AGENT.seg_model,
                image_tensor=image_tensor,
                prediction_mask=seg_result.get('prediction_mask')
            )
            xai_result = xai_agent.run()
            
            # Store results (without large arrays)
            results['segmentation'] = {
                'agent_output': seg_result['agent_output'],
                'metrics': seg_result['metrics'],
                'mask_available': True
            }
            
            results['xai'] = {
                'gradcam_summary': xai_result['gradcam_summary'],
                'attention_summary': xai_result['attention_summary'],
                'explanation': xai_result['explanation'],
                'heatmap_available': True
            }
        
        print(f"\n✅ COMPREHENSIVE ANALYSIS COMPLETED\n")
        
        return jsonify({
            'success': True,
            'results': results,
            'query_image': unique_filename,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        print(f"\n❌ COMPREHENSIVE ANALYSIS ERROR: {str(e)}\n")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'error': f'Comprehensive analysis error: {str(e)}',
            'success': False
        }), 500

# =====================================================================
# HEALTH CHECK
# =====================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Check system status"""
    return jsonify({
        'status': 'healthy' if ORCHESTRATOR is not None else 'initializing',
        'agents': {
            'model_loaded': MODEL is not None,
            'database_connected': COLLECTION is not None,
            'main_orchestrator': ORCHESTRATOR is not None,
            'segmentation_agent': SEGMENTATION_AGENT is not None,
            'xai_agent': 'initialized_on_demand'
        },
        'device': str(DEVICE) if DEVICE else None,
        'last_segmentation_available': LAST_SEGMENTATION['prediction_mask'] is not None
    }), 200

# =====================================================================
# STARTUP
# =====================================================================

if __name__ == '__main__':
    if initialize_medical_system():
        print("\n" + "="*70)
        print("🚀 STARTING MULTI-AGENT FLASK SERVER")
        print("="*70)
        print("\n📍 Web Pages:")
        print("   - Home: http://localhost:5000")
        #print("   - Consultation: http://localhost:5000/virtual-consultation")
        print("   - MRI Analysis: http://localhost:5000/mri-analysis")
        print("   - Segmentation: http://localhost:5000/segmentation")
        print("   - Explainability: http://localhost:5000/explainability")
        print("\n📡 API Endpoints:")
        print("   - /api/analyze (main orchestrator)")
        print("   - /api/segment (segmentation only)")
        print("   - /api/explain (XAI explanation - requires prior segmentation)")
        print("   - /api/segment-and-explain (combined)")
        print("   - /api/comprehensive-analysis (all agents)")
        print("   - /api/health (system status)")
        print("\n🤖 Agent Pipeline:")
        print("   Medical Orchestrator → Segmentation Agent → XAI Agent")
        print("\n" + "="*70 + "\n")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Cannot start server - initialization failed\n")
        sys.exit(1)