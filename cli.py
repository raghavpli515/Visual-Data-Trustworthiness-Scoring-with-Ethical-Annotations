import argparse, os, json
from src.vdt_scoring.config import load_config
from src.vdt_scoring.pipeline.infer import infer_video
from jsonschema import validate
import jsonschema

def main():
    parser = argparse.ArgumentParser(description="Video Trustworthiness Scoring (baseline)")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    args = parser.parse_args()

    cfg = load_config(args.config if os.path.exists(args.config) else None)
    results = infer_video(args.video, args.out, cfg)

    # Validate output against schema
    with open("src/vdt_scoring/schemas/results_schema.json", "r", encoding="utf-8") as f:
        schema = json.load(f)
    try:
        validate(instance=results, schema=schema)
    except jsonschema.ValidationError as e:
        print("WARNING: Results schema validation failed:", e)

    print(f"Done. Results saved to {os.path.join(args.out, 'results.json')}")

if __name__ == "__main__":
    main()
