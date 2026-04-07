"""
Compute exact-match at a given attempt number for all result JSONs in a folder.

For each example, we use the output produced at exactly attempt K.
If the example has fewer than K attempts, we use the last available attempt.

Usage:
    python agentic_clean/eval_at_attempt.py --results_dir agentic_clean/results --attempt 3
    python agentic_clean/eval_at_attempt.py --results_dir agentic_clean/results-max_attempts10 --attempt 1 2 3 5 10
"""
import os, sys, json, argparse, csv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from isomorphism_pred_KBQA.utils.evaluation import compute_exact_match


def em_at_attempt(entry: dict, k: int) -> bool:
    """Return exact match using the output from exactly attempt k.
    Falls back to the last attempt if the example finished before k."""
    log = entry.get("attempts_log", [])
    gold = entry.get("gold_sexp", "")
    if not log or not gold:
        return False

    # Find attempt k; fall back to last if not reached
    chosen = None
    for step in log:
        if step.get("attempt", 0) == k:
            chosen = step
            break
    if chosen is None:
        chosen = log[-1]

    output = chosen.get("output", "")
    if not output:
        return False
    return bool(compute_exact_match(output, gold))


def evaluate_file(path: str, attempts: list) -> dict:
    with open(path) as f:
        data = json.load(f)
    results = {}
    for k in attempts:
        ems = [em_at_attempt(v, k) for v in data.values()]
        n = len(ems)
        correct = sum(ems)
        results[k] = {"n": n, "correct": correct, "em": correct / n if n else 0.0}
    return results


def parse_filename(fname: str) -> tuple[str, str, str]:
    """Parse '{dataset}_{model}_{shotmode}_{mode}[_pred_iso].json' into
    (dataset, model, mode) where mode is exe_plain / double_iso / double_iso_pred_iso."""
    stem = fname.removesuffix(".json")
    # Split on first _fs_ or _zs_ to separate dataset+model from the rest
    for sep in ("_fs_", "_zs_"):
        if sep in stem:
            left, right = stem.split(sep, 1)
            # left  = "{dataset}_{model}"
            # right = "{verifier}_{feedback}[_pred_iso]"
            parts = left.split("_", 1)
            dataset = parts[0]
            model   = parts[1] if len(parts) > 1 else left
            mode    = right    # e.g. double_iso / exe_plain / double_iso_pred_iso
            return dataset, model, mode
    # Fallback: can't parse
    return stem, "", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True, help="Folder containing result JSONs")
    ap.add_argument("--attempt", type=int, nargs="+", required=True,
                    help="Attempt number(s) to evaluate at (1-indexed)")
    ap.add_argument("--pattern", default=".json",
                    help="Only include files whose name contains this string")
    ap.add_argument("--out_csv", default=None,
                    help="Optional path to save results as CSV")
    args = ap.parse_args()

    files = sorted(
        f for f in os.listdir(args.results_dir)
        if f.endswith(".json") and args.pattern in f
    )
    if not files:
        print(f"No JSON files found in {args.results_dir}")
        return

    attempts = sorted(args.attempt)
    header = f"{'Dataset':<12}  {'Model':<40}  {'Mode':<24}" + "".join(f"  @{k:>2}" for k in attempts)
    print(header)
    print("-" * len(header))

    # Group by model then dataset (preserve insertion order within groups)
    from collections import defaultdict
    grouped = defaultdict(lambda: defaultdict(list))
    file_meta = {}
    for fname in files:
        dataset, model, mode = parse_filename(fname)
        grouped[model][dataset].append(fname)
        file_meta[fname] = (dataset, model, mode)

    ordered_files = []
    for model in grouped:
        for dataset in grouped[model]:
            ordered_files.extend(grouped[model][dataset])

    csv_rows = []
    for fname in ordered_files:
        path = os.path.join(args.results_dir, fname)
        dataset, model, mode = file_meta[fname]
        try:
            res = evaluate_file(path, attempts)
        except Exception as e:
            print(f"{dataset:<12}  {model:<40}  {mode:<24}  ERROR: {e}")
            continue
        row = f"{dataset:<12}  {model:<40}  {mode:<24}" + "".join(f"  {res[k]['em']:.4f}" for k in attempts)
        print(row)
        csv_rows.append({"dataset": dataset, "model": model, "mode": mode,
                         **{f"@{k}": round(res[k]["em"], 4) for k in attempts}})

    if args.out_csv:
        fieldnames = ["dataset", "model", "mode"] + [f"@{k}" for k in attempts]
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nSaved to {args.out_csv}")


if __name__ == "__main__":
    main()
