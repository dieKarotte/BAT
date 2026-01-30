import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List


def load_label_map(csv_path: Path) -> Dict[str, str]:
    label_map = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_map[row["mid"]] = row["display_name"]
    return label_map


def load_prompts(prompt_path: Path) -> Dict[str, List[str]]:
    with open(prompt_path) as f:
        return json.load(f)


def format_event_answer(labels: List[str], label_map: Dict[str, str]) -> str:
    if not labels:
        return "unknown"
    names = [label_map.get(mid, mid) for mid in labels]
    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for name in names:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ", ".join(ordered)


def normalize_azimuth_deg(az: float) -> float:
    # Convert [0, 360) -> [-180, 180]
    if az is None:
        return az
    return az - 360 if az > 180 else az


def normalize_elevation_deg(el: float) -> float:
    # Convert BAT elevation (0-180, 90=horizontal) -> [-90, 90] (positive up)
    if el is None:
        return el
    return 90 - el


def format_doa_answer(sample) -> str:
    az_raw = sample.get("azimuth")
    el_raw = sample.get("elevation")
    az = normalize_azimuth_deg(az_raw)
    el = normalize_elevation_deg(el_raw)
    return f"azimuth: {az} deg, elevation: {el} deg"


def format_distance_answer(sample) -> str:
    dist = sample.get("distance")
    return f"distance: {dist} m"


def choose_prompt(prompt_dict: Dict[str, List[str]], task: str) -> str:
    candidates = prompt_dict.get(task, [])
    if not candidates:
        return "Describe the audio."
    return random.choice(candidates)


def convert_split(
    input_path: Path,
    output_path: Path,
    prompt_dict: Dict[str, List[str]],
    label_map: Dict[str, str],
    distance_mode: str,
):
    data = json.load(open(input_path))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {input_path}, got {type(data)}")

    out = []
    qid = 0
    for sample in data:
        binaural_path = sample.get("binaural_path")
        if binaural_path is None:
            raise KeyError("Missing 'binaural_path' in sample")

        # 1) Audio event detection
        event_q = choose_prompt(prompt_dict, "audio_event_detection")
        event_a = format_event_answer(sample.get("label", []), label_map)
        out.append(
            {
                "id": sample.get("id", f"sample-{qid}"),
                "binaural_path": binaural_path,
                "question_id": qid,
                "question_type": "audio_event_detection",
                "question": event_q,
                "answer": event_a,
            }
        )
        qid += 1

        # 2) Spatial DoA (azimuth/elevation)
        doa_q = choose_prompt(prompt_dict, "spatial_doa")
        doa_a = format_doa_answer(sample)
        if distance_mode == "append":
            doa_a = doa_a + ", " + format_distance_answer(sample)
        out.append(
            {
                "id": sample.get("id", f"sample-{qid}"),
                "binaural_path": binaural_path,
                "question_id": qid,
                "question_type": "spatial_doa",
                "question": doa_q,
                "answer": doa_a,
            }
        )
        qid += 1

        # 3) Distance estimation (optional separate)
        if distance_mode == "separate":
            dist_q = choose_prompt(prompt_dict, "distance_estimation")
            dist_a = format_distance_answer(sample)
            out.append(
                {
                    "id": sample.get("id", f"sample-{qid}"),
                    "binaural_path": binaural_path,
                    "question_id": qid,
                    "question_type": "distance_estimation",
                    "question": dist_q,
                    "answer": dist_a,
                }
            )
            qid += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"data": out}, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True, help="Folder containing bat_{train,valid,test}.json")
    parser.add_argument("--output_root", required=True, help="Output root for QA jsons")
    parser.add_argument("--stage", default="bat", help="Stage subfolder name")
    parser.add_argument("--prompt_file", required=True, help="SALMONN prompt json file")
    parser.add_argument("--label_map_csv", required=True, help="AudioSet class_labels_indices.csv")
    parser.add_argument(
        "--distance_mode",
        choices=["none", "append", "separate"],
        default="none",
        help="How to use distance: none | append to DoA answer | separate QA",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    prompt_dict = load_prompts(Path(args.prompt_file))
    label_map = load_label_map(Path(args.label_map_csv))

    input_root = Path(args.input_root)
    output_root = Path(args.output_root) / args.stage

    split_map = {
        "train": input_root / "bat_train.json",
        "val": input_root / "bat_valid.json",
        "test": input_root / "bat_test.json",
    }

    for split, path in split_map.items():
        if not path.exists():
            raise FileNotFoundError(path)
        output_path = output_root / f"{split}.json"
        convert_split(path, output_path, prompt_dict, label_map, args.distance_mode)


if __name__ == "__main__":
    main()
