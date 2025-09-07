import json
from jsonschema import validate
import jsonschema

def test_results_schema():
    from src.vdt_scoring.schemas import results_schema
    sample = {
        "video_path": "x.mp4",
        "summary": {"global_score": 0.5, "frames_evaluated": 1, "flagged_segments": []},
        "frames": [{"index": 0, "score": 0.3, "explanations": [], "heatmap_path": None}],
    }
    with open(results_schema.__file__.replace("__init__.py", "results_schema.json"), "r", encoding="utf-8") as f:
        schema = json.load(f)
    validate(instance=sample, schema=schema)
