#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script d'analyse de consultation - Exécuté dans venv_consult
Lit les données utilisateur depuis stdin, exécute le pipeline médical, retourne JSON
"""
import sys
import json
from pathlib import Path
from datetime import datetime

def main():
    try:
        # Lire les arguments
        if len(sys.argv) < 2:
            print(json.dumps({'error': 'No user data provided', 'success': False}))
            sys.exit(1)
        
        user_data = json.loads(sys.argv[1])
        user_id = user_data.get('id', 0)
        
        # Importer les agents (dans venv_consult avec pydantic v2)
        from agents_Consult.parser_agent import ParserAgent
        from agents_Consult.scorer_agent import SymptomScorerAgent
        from agents_Consult.cancer_agent import CancerPotentialAgent
        from agents_Consult.explain_agent import ExplainabilityAgent
        from agents_Consult.report_agent import ReportAgent
        
        # Initialiser les agents
        parser = ParserAgent()
        scorer = SymptomScorerAgent()
        cancer_agent = CancerPotentialAgent()
        explainer = ExplainabilityAgent()
        reporter = ReportAgent()
        
        # Pipeline d'analyse
        parsed = parser.parse(user_data)
        score = scorer.score(parsed)
        assessment = cancer_agent.assess(parsed, score)
        explanation = explainer.explain(parsed, score, assessment)
        final_report = reporter.generate(parsed, score, assessment, explanation)
        
        # Sauvegarder les rapports
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        json_path = reports_dir / f"rapport_patient_{user_id}.json"
        md_path = reports_dir / f"patient_{user_id}_REPORT.md"
        
        json_path.write_text(final_report.model_dump_json(indent=2), encoding="utf-8")
        md_path.write_text(
            explanation + f"\n\n**URGENCY LEVEL: {final_report.next_steps['urgency_level']}**",
            encoding="utf-8"
        )
        
        # Retourner le résultat en JSON
        result = {
            'success': True,
            'user_id': user_id,
            'analysis': {
                'cancer_probability': assessment.cancer_probability,
                'risk_category': assessment.risk_category,
                'risk_score': score.risk_score,
                'severity_level': score.severity_level,
                'primary_concerns': assessment.primary_concerns,
                'recommended_tests': assessment.recommended_tests,
                'urgency_level': final_report.next_steps['urgency_level']
            },
            'explanation': explanation,
            'report_files': {
                'json': str(json_path),
                'markdown': str(md_path)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        print(json.dumps(result, ensure_ascii=False))
        
    except Exception as e:
        import traceback
        error_result = {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        print(json.dumps(error_result, ensure_ascii=False))
        sys.exit(1)

if __name__ == '__main__':
    main()

