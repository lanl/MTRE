#!/usr/bin/env python3
import os
import json
import csv
import argparse

def load_result_json(path):
    """
    Load a Ray result.json. Supports:
      - a single JSON object file
      - a file with multiple JSON objects (take the last valid line)
    """
    with open(path, "r") as f:
        text = f.read().strip()
    # Try full-file JSON first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fall back to JSON-per-line; take the last non-empty valid line
        last = None
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                last = json.loads(s)
            except json.JSONDecodeError:
                continue
        return last

def main(base_dir, out_csv):
    base_dir = os.path.expanduser(base_dir)
    rows = []
    n_dirs = 0
    n_found = 0
    n_errors = 0

    for root, dirs, files in os.walk(base_dir):
        n_dirs += 1
        if "result.json" not in files:
            continue
        n_found += 1
        fp = os.path.join(root, "result.json")
        try:
            rec = load_result_json(fp)
            if not isinstance(rec, dict):
                raise ValueError("result.json did not parse to a JSON object")

            # Standard fields you asked for
            out = {
                "trial_dir": root,
                "trial_id": rec.get("trial_id"),
                "date": rec.get("date"),
                "timestamp": rec.get("timestamp"),
                "training_iteration": rec.get("training_iteration"),
                "accuracy": rec.get("accuracy"),
                "f1": rec.get("f1"),
                "auroc": rec.get("auroc"),
                "time_total_s": rec.get("time_total_s"),
            }

            # Flatten config.* keys as columns
            cfg = rec.get("config", {})
            if isinstance(cfg, dict):
                for k, v in cfg.items():
                    out[f"config.{k}"] = v

            rows.append(out)

        except Exception as e:
            n_errors += 1
            print(f"[WARN] Skipping {fp}: {e}")

    if not rows:
        raise SystemExit("No result.json files found or none parsed successfully.")

    # Build CSV header: fixed columns + all config.* keys we saw
    fixed = [
        "trial_dir", "trial_id", "date", "timestamp",
        "training_iteration", "accuracy", "f1", "auroc", "time_total_s"
    ]
    cfg_keys = sorted({k for r in rows for k in r.keys() if k.startswith("config.")})
    fieldnames = fixed + cfg_keys

    # Write CSV
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[OK] Wrote {len(rows)} rows to: {out_csv}")
    print(f"Scanned {n_dirs} directories; found {n_found} result.json; errors: {n_errors}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Ray Tune result.json files into one CSV.")
    parser.add_argument(
        "--base_dir",
        default="~/ray_results/trainable_tau_2025-11-26_13-57-57",
        help="Top-level directory to scan"
    )
    parser.add_argument(
        "--out_csv",
        default="ray_results_summary.csv",
        help="Path to output CSV"
    )
    args = parser.parse_args()
    main(args.base_dir, args.out_csv)
