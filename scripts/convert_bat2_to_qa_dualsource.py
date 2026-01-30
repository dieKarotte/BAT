import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm


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


def normalize_azimuth_deg(az: float) -> float:
    if az is None:
        return az
    return az - 360 if az > 180 else az


def normalize_elevation_deg(el: float) -> float:
    if el is None:
        return el
    return 90 - el


def format_event_answer(labels: List[str], label_map: Dict[str, str]) -> str:
    if not labels:
        return "unknown"
    names = [label_map.get(mid, mid) for mid in labels]
    seen = set()
    ordered = []
    for name in names:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ", ".join(ordered)


def format_doa_answer(sample) -> str:
    az_raw = sample.get("azimuth")
    el_raw = sample.get("elevation")
    az = normalize_azimuth_deg(az_raw)
    el = normalize_elevation_deg(el_raw)
    return f"azimuth: {az} deg, elevation: {el} deg"


def choose_prompt(prompt_dict: Dict[str, List[str]], task: str) -> str:
    candidates = prompt_dict.get(task, [])
    if not candidates:
        return "Describe the audio."
    return random.choice(candidates)


def gender_to_string(gender_raw: str) -> str:
    gender_map = {"M": "male", "F": "female"}
    return gender_map.get(gender_raw, gender_raw or "unknown")

def normalize_binaural_path(sample, binaural_root: str = "") -> str:
    raw_path = sample.get("binaural_path")
    if not raw_path:
        return raw_path

    # Already in flat format: .../<split>/<scene>_task2_XXXX_binaural.wav
    if raw_path.endswith("_binaural.wav") and "/task2_" not in raw_path and not binaural_root:
        return raw_path

    # Convert .../<split>/<scene>/task2_XXXX/binaural.wav -> .../<split>/<scene>_task2_XXXX_binaural.wav
    match = re.search(r"/(train|val|test)/([^/]+)/task2_(\d+)/binaural\.wav$", raw_path)
    if match:
        split = match.group(1)
        scene = match.group(2)
        task_id = match.group(3)
        root = binaural_root if binaural_root else raw_path.split(f"/{split}/")[0]
        candidate = f"{root}/{split}/{scene}_task2_{task_id}_binaural.wav"
        if Path(candidate).is_file():
            return candidate
        return candidate

    return raw_path


def convert_split(
    input_path: Path,
    output_path: Path,
    prompt_dict: Dict[str, List[str]],
    label_map: Dict[str, str],
    doa_suffix: str,
    binaural_root: str,
):
    data = json.load(open(input_path))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {input_path}, got {type(data)}")

    out = []
    qid = 0
    for sample in tqdm(data, desc=f"convert:{input_path.name}", unit="sample"):
        binaural_path = normalize_binaural_path(sample, binaural_root=binaural_root)
        if binaural_path is None:
            raise KeyError("Missing 'binaural_path' in sample")

        # 1) Audio event detection (two sources)
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

        # 2) Dual source DoA: ask for target gender location
        gender_raw = sample.get("gender", "")
        gender_str = gender_to_string(gender_raw)
        dual_template = choose_prompt(prompt_dict, "dual_source")
        dual_q = dual_template.format(gender=gender_str)
        if doa_suffix:
            dual_q = f"{dual_q} {doa_suffix}".strip()
        dual_a = format_doa_answer(sample)
        out.append(
            {
                "id": sample.get("id", f"sample-{qid}"),
                "binaural_path": binaural_path,
                "question_id": qid,
                "question_type": "dual_source_doa",
                "question": dual_q,
                "answer": dual_a,
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
    parser.add_argument("--stage", default="bat2", help="Stage subfolder name")
    parser.add_argument("--prompt_file", required=True, help="SALMONN prompt json file")
    parser.add_argument("--label_map_csv", required=True, help="AudioSet class_labels_indices.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--doa_suffix",
        default="Provide azimuth and elevation. Note: Azimuth 0° is front, positive left, range [-180°, 180°]; Elevation 0° is horizontal, positive up, range [-90°, 90°].",
    )
    parser.add_argument("--binaural_root", default="", help="Override binaural root path for remapping")
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

    for split in tqdm(list(split_map.keys()), desc="splits", unit="split"):
        path = split_map[split]
        if not path.exists():
            raise FileNotFoundError(path)
        output_path = output_root / f"{split}.json"
        convert_split(path, output_path, prompt_dict, label_map, args.doa_suffix, args.binaural_root)


if __name__ == "__main__":
    main()
