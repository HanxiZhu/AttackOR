"""
Merge NL4OPT adversarial test datasets (arithmetic, attribute, phantom, unit)
into a single JSON array, deduplicating by record id (first occurrence kept).
"""
import argparse
import json
import os

# Default data directory (can be overridden by --data-dir)
DEFAULT_DATA_DIR = "./datasets_local"

# Input files in merge order (first occurrence of each id is kept)
INPUT_FILES = [
    "IndustryOR_test_adversarial_arithmetic.jsonl",
    "IndustryOR_test_adversarial_attribute.jsonl",
    "IndustryOR_test_adversarial_phantom.jsonl",
    "IndustryOR_test_adversarial_unit.jsonl",
]

OUTPUT_FILE = "IndustryOR_test_adversarial_merged.jsonl"


def main():
    parser = argparse.ArgumentParser(description="Merge adversarial datasets and deduplicate by id.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the adversarial JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=f"Output path (default: <data-dir>/{OUTPUT_FILE})",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    output_path = args.output or os.path.join(data_dir, OUTPUT_FILE)

    seen_ids = set()
    merged = []

    for filename in INPUT_FILES:
        path = os.path.join(data_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Input file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array in {filename}")
        for record in data:
            rid = record.get("id")
            if rid is None:
                continue
            if rid not in seen_ids:
                seen_ids.add(rid)
                merged.append(record)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Merged {len(merged)} unique records -> {output_path}")


if __name__ == "__main__":
    main()
