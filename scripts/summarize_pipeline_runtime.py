#!/usr/bin/env python3
"""
Summarize pipeline runtimes for 3DGS compression.

Steps:
1) Voxelize + merge attributes (from python/test_voxelize_3dgs.py).
2) RAHT encode/decode pipeline (from python/encode_3dgs.py).
"""

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List


def _read_last_row(csv_path: str) -> Dict[str, str]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Runtime log not found: {csv_path}")

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        last_row = None
        for row in reader:
            last_row = row

    if not last_row:
        raise ValueError(f"Runtime log is empty: {csv_path}")
    return last_row


def _read_raht_rows(csv_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"RAHT runtime log not found: {csv_path}")

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]

    if not rows:
        raise ValueError(f"RAHT runtime log is empty: {csv_path}")
    return rows


def _aggregate_raht(rows: List[Dict[str, str]]) -> Dict[int, Dict[str, float]]:
    """
    Mean RAHT pipeline times per quantization step (ms) with component detail.
    Returns {quant_step: {field: mean_ms}}.
    """
    sums = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)

    fields = [
        "RAHT_prelude_time",
        "Total_enc_time",
        "Total_dec_time",
    ]

    for row in rows:
        qstep = int(float(row["Quantization_Step"]))
        counts[qstep] += 1
        for f in fields:
            sums[qstep][f] += float(row[f]) * 1000.0  # to ms

    aggregated = {}
    for qstep, field_sums in sums.items():
        aggregated[qstep] = {f: field_sums[f] / counts[qstep] for f in fields}
        # Total_ms excludes RAHT_transform_time because it is already inside Total_enc_time
        total_ms = aggregated[qstep]["RAHT_prelude_time"] + aggregated[qstep]["Total_enc_time"] + aggregated[qstep]["Total_dec_time"]
        aggregated[qstep]["Total_ms"] = total_ms

    return aggregated


def _write_markdown(out_path: str, summary: dict, detail: dict, voxel_log: str, raht_log: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Pipeline runtime summary (ms)\n\n")

        f.write("## Aggregated\n\n")
        f.write("| " + " | ".join(summary["header"]) + " |\n")
        f.write("|" + "|".join([" --- " for _ in summary["header"]]) + "|\n")
        for row in summary["rows"]:
            f.write("| " + " | ".join(row) + " |\n")
        f.write("\n")

        f.write("## RAHT breakdown (mean per quant step)\n\n")
        f.write("| " + " | ".join(detail["header"]) + " |\n")
        f.write("|" + "|".join([" --- " for _ in detail["header"]]) + "|\n")
        for row in detail["rows"]:
            f.write("| " + " | ".join(row) + " |\n")
        f.write("\n")

        f.write("Sources:\n")
        f.write(f"- Voxelization log: {os.path.abspath(voxel_log)}\n")
        f.write(f"- RAHT log: {os.path.abspath(raht_log)}\n")


def summarize(voxel_log: str, raht_log: str, out_path: str | None = None) -> None:
    voxel_row = _read_last_row(voxel_log)
    raht_rows = _read_raht_rows(raht_log)

    voxel_ms = float(voxel_row["Total_time_ms"])
    raht_by_q = _aggregate_raht(raht_rows)
    if not raht_by_q:
        raise ValueError("No RAHT runtimes found to summarize.")

    # Sorted by quantization step for consistent output
    header_summary = [
        "Quant_step",
        "Voxelize+Merge (ms)",
        "RAHT+Entropy (ms)",
        "End-to-end (ms)",
    ]
    col_widths = [len(h) for h in header_summary]
    rows_out = []
    for qstep in sorted(raht_by_q.keys()):
        raht_ms = raht_by_q[qstep]["Total_ms"]
        total_ms = voxel_ms + raht_ms
        row = [str(qstep), f"{voxel_ms:.2f}", f"{raht_ms:.2f}", f"{total_ms:.2f}"]
        rows_out.append(row)
        col_widths = [max(col_widths[i], len(row[i])) for i in range(len(header_summary))]

    def _fmt_row(items, widths):
        return "  ".join(item.rjust(widths[idx]) for idx, item in enumerate(items))

    print("Pipeline runtime summary (ms)")
    print(_fmt_row(header_summary, col_widths))
    print("-" * (sum(col_widths) + 2 * (len(header_summary) - 1)))
    for row in rows_out:
        print(_fmt_row(row, col_widths))
    print("  Quant_step: quantization step size.")
    print("  Voxelize+Merge: voxelization + attribute merge (fixed across steps).")
    print("  RAHT+Entropy: RAHT + quantize/reorder + entropy encode/decode (varies by step).")
    print("  End-to-end: Voxelize+Merge + RAHT+Entropy.")

    summary_md = {"header": header_summary, "rows": rows_out}

    # Detailed RAHT breakdown per quantization step
    detail_fields = [
        ("RAHT_prelude_time", "RAHT_prelude"),
        ("Total_enc_time", "Encode_total"),
        ("Total_dec_time", "Decode_total"),
        ("Total_ms", "Total"),
    ]
    detail_header = ["Quant_step"] + [label for _, label in detail_fields]
    detail_widths = [len(h) for h in detail_header]
    detail_rows = []
    for qstep in sorted(raht_by_q.keys()):
        vals = [f"{raht_by_q[qstep][field]:.2f}" for field, _ in detail_fields]
        row = [str(qstep)] + vals
        detail_rows.append(row)
        detail_widths = [max(detail_widths[i], len(row[i])) for i in range(len(detail_header))]

    print("\nRAHT+Entropy runtime details (ms, mean over frames per quant step)")
    print(_fmt_row(detail_header, detail_widths))
    print("-" * (sum(detail_widths) + 2 * (len(detail_header) - 1)))
    for row in detail_rows:
        print(_fmt_row(row, detail_widths))
    print("  RAHT_prelude: list/flags/weights construction.")
    print("  Encode_total: RAHT transform + quantize + reorder + entropy encode.")
    print("  Decode_total: entropy decode + dequant + reorder + iRAHT.")
    print("  Total: RAHT_prelude + encode_total + decode_total (per step).")

    detail_md = {"header": detail_header, "rows": detail_rows}

    print("\nSources:")
    print(f"- Voxelization log: {os.path.abspath(voxel_log)}")
    print(f"- RAHT log: {os.path.abspath(raht_log)}")
    print("RAHT times include RAHT prelude/transform, encoding, and decoding.")

    if out_path:
        _write_markdown(out_path, summary_md, detail_md, voxel_log, raht_log)
        print(f"\nMarkdown summary written to: {os.path.abspath(out_path)}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_voxel_log = os.path.abspath(os.path.join(script_dir, "..", "results", "runtime_voxelize_3dgs.csv"))
    default_raht_log = os.path.abspath(os.path.join(script_dir, "..", "results", "runtime_3dgs.csv"))
    default_out = os.path.abspath(os.path.join(script_dir, "..", "results", "pipeline_runtime_summary.md"))

    parser = argparse.ArgumentParser(description="Summarize 3DGS pipeline runtimes.")
    parser.add_argument("--voxel-log", default=default_voxel_log, help="Path to voxelization runtime CSV.")
    parser.add_argument("--raht-log", default=default_raht_log, help="Path to RAHT runtime CSV.")
    parser.add_argument("--out", default=default_out, help="Path to write markdown summary (set empty to skip).")
    args = parser.parse_args()

    out_path = args.out if args.out else None
    summarize(args.voxel_log, args.raht_log, out_path=out_path)


if __name__ == "__main__":
    main()
