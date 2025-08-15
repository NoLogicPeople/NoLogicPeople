#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


def build_key(category: str, service_name: str, service_app_common_name: str, service_description: str) -> str:
    # Normalize whitespace lightly; keep content intact
    def norm(s: str) -> str:
        return (s or "").strip()

    return f"{norm(category)}-{norm(service_name)}-{norm(service_app_common_name)}: {norm(service_description)}"


def read_examples(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Ensure values are lists of strings
    normalized = {}
    numbering_pattern = re.compile(r"^\s*\d+\.\s*")
    for k, v in data.items():
        if isinstance(v, list):
            cleaned_list = []
            for x in v:
                s = str(x).strip()
                s = numbering_pattern.sub("", s)
                cleaned_list.append(s)
            normalized[k] = cleaned_list
        else:
            s = str(v).strip()
            s = numbering_pattern.sub("", s)
            normalized[k] = [s]
    return normalized


def triple_rows(csv_in: Path, examples_map: dict, csv_out: Path) -> None:
    # Use utf-8-sig to gracefully handle potential BOM in header
    with csv_in.open("r", encoding="utf-8-sig", newline="") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames or []
        # Append example column at the end
        out_fieldnames = fieldnames + ["example"]

        with csv_out.open("w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=out_fieldnames)
            writer.writeheader()

            for row in reader:
                key = build_key(
                    row.get("category", ""),
                    row.get("service_name", ""),
                    row.get("service_app_common_name", ""),
                    row.get("service_description", ""),
                )
                examples = examples_map.get(key, [])
                # Ensure exactly 3 rows per input row
                for i in range(3):
                    example_text = examples[i] if i < len(examples) else ""
                    out_row = dict(row)
                    out_row["example"] = example_text
                    writer.writerow(out_row)


def main():
    parser = argparse.ArgumentParser(description="Triple CSV rows and append example utterances from JSON map.")
    parser.add_argument("--input", required=True, help="Input CSV file (e.g., rag/scraped_items_v2.csv)")
    parser.add_argument("--json", required=True, help="JSON file with key -> [examples]")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    args = parser.parse_args()

    csv_in = Path(args.input)
    json_in = Path(args.json)
    csv_out = Path(args.output)

    if not csv_in.exists():
        raise SystemExit(f"Input CSV not found: {csv_in}")
    if not json_in.exists():
        raise SystemExit(f"JSON file not found: {json_in}")
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    examples_map = read_examples(json_in)
    triple_rows(csv_in, examples_map, csv_out)


if __name__ == "__main__":
    main()
