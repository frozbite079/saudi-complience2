import json
from app.config import VIDEO_OUTPUT_DIR
from Voilation_Template.report_generator import generate_report_html, save_report

data = {
    "classification": "Electrical",
    "classifications": ["Electrical"],
    "db_category": "Electricity",
    "db_categories": ["Electricity"],
    "verdicts": [
        {
            "sbc_reference": "52-7.2",
            "category": "Electricity > Wiring Systems",
            "rule_text": "Sealing of wiring system...",
            "cv_target": "Fire-rated seal at cable entry through walls/floors",
            "detection_type": "PRESENCE",
            "priority": "CRITICAL",
            "verdict": "VIOLATION",
            "evidence": "Wires exiting ceiling...",
            "confidence": 0.9,
            "source_item_id": "item_4",
            "bbox": [450, 100, 550, 250],
            "timestamp_sec": 3.5
        }
    ],
    "annotated_image": None,
    "annotated_media": {
        "type": "video",
        "download_url": "/api/v1/download/violation_3bc7c925e0d2.mp4",
        "filename": "violation_3bc7c925e0d2.mp4"
    },
    "summary": {
        "total_rules_evaluated": 30,
        "violations": 9,
        "compliant": 0,
        "uncertain": 21,
        "high_priority_violations": 3,
        "compliance_rate": 0.0
    },
    "project_name": "Al-Nassim Tower",
    "contractor_name": "Saudi Oger Ltd.",
    "token_usage": {}
}

output = save_report(data, VIDEO_OUTPUT_DIR / "test.html")
print("Saved to", output)
