import argparse
import json
from pathlib import Path

DEFAULT_QUESTION = "What is the azimuth, elevation, and distance of the sound source?"


def format_answer(sample):
    az = sample.get("azimuth")
    el = sample.get("elevation")
    dist = sample.get("distance")
    return f"azimuth: {az} deg, elevation: {el} deg, distance: {dist} m"


def convert_split(input_path: Path, output_path: Path, question: str):
    data = json.load(open(input_path))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {input_path}, got {type(data)}")

    out = []
    for idx, sample in enumerate(data):
        entry = {
            "id": sample.get("id", f"sample-{idx}"),
            "binaural_path": sample.get("binaural_path"),
            "question_id": idx,
            "question_type": "doa_azel_distance",
            "question": question,
            "answer": format_answer(sample),
        }
        out.append(entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"data": out}, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True, help="Folder containing bat_{train,valid,test}.json")
    parser.add_argument("--output_root", required=True, help="Output root for QA jsons")
    parser.add_argument("--stage", default="bat", help="Stage subfolder name")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root) / args.stage

    split_map = {
        "train": input_root / "bat_train.json",
        "valid": input_root / "bat_valid.json",
        "test": input_root / "bat_test.json",
    }

    for split, path in split_map.items():
        if not path.exists():
            raise FileNotFoundError(path)
        output_path = output_root / f"{split}.json"
        convert_split(path, output_path, args.question)


if __name__ == "__main__":
    main()
