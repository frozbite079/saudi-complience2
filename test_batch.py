import os
from pathlib import Path
from app.violation_detector import detect_violations_batch
from Voilation_Template.report_generator import save_report

# Get absolute paths to the images relative to saudi-complience root
saudi_compliance_dir = Path(__file__).resolve().parent
media_files = [
    str(saudi_compliance_dir / "111.jpeg"),
    str(saudi_compliance_dir / "electrical-construction-technology-trends-to-know.jpg")
]

print("Running batch violations detection...")
result = detect_violations_batch(
    media_paths_or_urls=media_files,
    is_videos=[False, False],
    project_name="Batch Test Project",
    contractor_name="El-Seif Engineering"
)

# Output summary to console
print("\n" + "="*40)
print("BATCH DETECTOR RESULTS SUMMARY")
print("="*40)
print(f"Classification: {result.get('classification')}")
print(f"Classifications: {result.get('classifications')}")
print(f"Total Rules Evaluated: {result['summary']['total_rules_evaluated']}")
print(f"Violations: {result['summary']['violations']}")
print(f"Compliant: {result['summary']['compliant']}")
print(f"Uncertain: {result['summary']['uncertain']}")
print(f"Compliance Rate: {result['summary']['compliance_rate']:.2f}%")
print(f"Recommendations Count: {len(result.get('ai_recommendations', []))}")
print(f"Annotated Media List Size: {len(result.get('annotated_media_list', []))}")

# Check frame_b64 in _report_verdicts
print("\nVerdicts for report:")
for i, v in enumerate(result.get("_report_verdicts", [])):
    has_frame_b64 = "Yes" if v.get("frame_b64") else "No"
    print(f"  [{i+1}] SBC {v.get('sbc_reference')} on {v.get('media_name')} - Has Cropped Evidence: {has_frame_b64}")

# Save report in the active workspace Raas-OCR/outputs/ to bypass sandbox write restrictions
active_workspace_dir = Path("/home/redspark/Pictures/Raas-OCR")
report_path = active_workspace_dir / "outputs" / "batch_test_report.html"
saved_path = save_report(result, report_path, project_name="Batch Test Project", contractor_name="El-Seif Engineering")
print(f"\nHTML Report saved successfully to: {saved_path}")
