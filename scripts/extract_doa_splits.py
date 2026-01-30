import argparse
import json
from pathlib import Path


def is_doa_sample(sample) -> bool:
    qtype = str(sample.get("question_type", "")).lower()
    return "doa" in qtype


def load_json(path: Path):
    data = json.load(open(path))
    if isinstance(data, dict) and "data" in data:
        return data["data"], True
    if isinstance(data, list):
        return data, False
    raise ValueError(f"Unsupported json format: {path}")


def save_json(path: Path, samples, wrap: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    if wrap:
        payload = {"data": samples}
    else:
        payload = samples
    with open(path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True, help="QA root directory")
    parser.add_argument("--stage", required=True, help="Stage subfolder name")
    parser.add_argument("--splits", default="val,test", help="Comma-separated split names")
    parser.add_argument("--prefix", default="DoA_", help="Output prefix for filtered files")
    args = parser.parse_args()

    input_root = Path(args.input_root) / args.stage
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    for split in splits:
        in_path = input_root / f"{split}.json"
        if not in_path.exists():
            raise FileNotFoundError(in_path)
        samples, wrap = load_json(in_path)
        doa_samples = [s for s in samples if is_doa_sample(s)]
        out_path = input_root / f"{args.prefix}{split}.json"
        save_json(out_path, doa_samples, wrap)
        print(f"{split}: {len(doa_samples)} / {len(samples)} -> {out_path}")


if __name__ == "__main__":
    main()
