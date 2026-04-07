#!/usr/bin/env python3
"""
agentic_clean/sexp_prompting.py

Single-pass S-expression generation (no feedback loop) with optional
isomorphism-type hint injection.

Datasets   : webqsp | grailqa | grailqapp
Providers  : openai | vllm_server | openrouter | huggingface | gemini | vllm
ISO mode   : none (no hint) | gold (computed from gold s-expression) |
             predicted (loaded from --iso_pred_path)

Output (per example saved in result dict):
  id, question, pred_sexp, gold_sexp, correct, iso_type, iso_mode

Usage examples:
  # Baseline (no ISO hint)
  python -m agentic_clean.sexp_prompting \\
      --dataset grailqapp \\
      --data_path /path/to/data.dill \\
      --provider openrouter --model moonshotai/kimi-k2.5 \\
      --out_dir agentic_clean/results_sexp

  # With gold ISO hint
  python -m agentic_clean.sexp_prompting \\
      --dataset webqsp \\
      --data_path /path/to/data.dill \\
      --provider openai --model gpt-4o \\
      --iso_mode gold \\
      --out_dir agentic_clean/results_sexp

  # With predicted ISO hint
  python -m agentic_clean.sexp_prompting \\
      --dataset grailqa \\
      --data_path /path/to/data.dill \\
      --provider openrouter --model moonshotai/kimi-k2.5 \\
      --iso_mode predicted \\
      --iso_pred_path agentic_clean/pred_iso/grailqa_dev1000_majority_vote.json \\
      --out_dir agentic_clean/results_sexp
"""

import os
import sys
import argparse
import asyncio
import json

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------------
# Import shared helpers from verify_loop
# ---------------------------------------------------------------------------
from agentic_clean.verify_loop import (
    create_llm,
    seed_everything,
    extract_sexpr,
    read_file,
    clean_value,
    format_kg_tuples,
    render_user_prompt,
    load_dataset,
    ISO_DEFINITIONS,
)
from compute_isomorphisms.code.get_isomorphism import obtain_isomorphism_id
from isomorphism_pred_KBQA.utils.evaluation import compute_exact_match


# ---------------------------------------------------------------------------
# Prompt-file selection
# ---------------------------------------------------------------------------
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

# System prompts: (dataset, iso_mode) → filename
_SYS_PROMPTS = {
    ("webqsp",    "none"):      "fs_sys_prompt_webqsp.txt",
    ("webqsp",    "gold"):      "fs_sys_prompt_webqsp_w_gold_iso.txt",
    ("webqsp",    "predicted"): "fs_sys_prompt_webqsp_w_gold_iso.txt",
    ("grailqa",   "none"):      "fs_sys_prompt_grailqa.txt",
    ("grailqa",   "gold"):      "fs_sys_prompt_grailqa_w_gold_iso.txt",
    ("grailqa",   "predicted"): "fs_sys_prompt_grailqa_w_gold_iso.txt",
    ("grailqapp", "none"):      "fs_sys_prompt_grailqa.txt",
    ("grailqapp", "gold"):      "fs_sys_prompt_grailqa_w_gold_iso.txt",
    ("grailqapp", "predicted"): "fs_sys_prompt_grailqa_w_gold_iso.txt",
}

# User prompt templates: dataset → filename
_TEMPLATES = {
    "webqsp":    "user_prompt.txt",
    "grailqa":   "user_prompt_grailqa.txt",
    "grailqapp": "user_prompt_grailqa.txt",
}


def get_sys_prompt(dataset: str, iso_mode: str) -> str:
    fname = _SYS_PROMPTS[(dataset, iso_mode)]
    return read_file(os.path.join(PROMPTS_DIR, fname))


def get_user_template(dataset: str) -> str:
    fname = _TEMPLATES[dataset]
    return read_file(os.path.join(PROMPTS_DIR, fname))


# ---------------------------------------------------------------------------
# ISO hint injection
# ---------------------------------------------------------------------------
def inject_iso_hint(user_prompt: str, iso_type: str) -> str:
    """Insert a reasoning-pattern hint before '**S-expression:**' in the prompt."""
    definition = ISO_DEFINITIONS.get(iso_type, "")
    hint = (
        f"\nHint — The reasoning pattern for this question is **{iso_type}**: "
        f"{definition}\n"
        if definition else
        f"\nHint — The reasoning pattern for this question is **{iso_type}**.\n"
    )
    marker = "**S-expression:**"
    if marker in user_prompt:
        return user_prompt.replace(marker, hint + marker)
    return user_prompt + hint


# ---------------------------------------------------------------------------
# Predicted ISO loader  (mirrors verify_loop.py logic)
# ---------------------------------------------------------------------------
def _iso_from_item(item: dict):
    """Extract ISO type string from a pred-iso dict entry.

    Handles:
      - 'predicted_iso' or 'pred_rp'  → already a T-x string
      - 'predicted' (integer)          → convert to "T-{n}" (webqsp unified-vote format)
    """
    iso = item.get("predicted_iso") or item.get("pred_rp")
    if iso:
        return iso
    # webqsp unified-vote format: {"predicted": 1, "true": 1, ...}
    if "predicted" in item and item["predicted"] is not None:
        return f"T-{item['predicted']}"
    return None


def load_predicted_iso(json_path: str) -> dict:
    """Load predicted-ISO file → {qid: iso_type} for INCORRECT examples only.

    Mirrors verify_loop.py: only keeps entries where correct==False so that
    correct examples can be copied from the gold-iso run instead of re-run.

    Supports:
      - list of dicts  {id, predicted_iso|pred_rp|predicted, correct, ...}
      - dict of dicts  {qid: {predicted_iso|pred_rp, correct, ...}}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    mapping = {}
    if isinstance(raw, list):
        for item in raw:
            if not item.get("correct", True):
                qid = str(item["id"])
                iso = _iso_from_item(item)
                if iso:
                    mapping[qid] = iso
    elif isinstance(raw, dict):
        for qid, item in raw.items():
            if isinstance(item, dict) and not item.get("correct", True):
                iso = _iso_from_item(item)
                if iso:
                    mapping[str(qid)] = iso
    else:
        raise ValueError(f"Unrecognised format in {json_path}")
    return mapping


# ---------------------------------------------------------------------------
# Async single-call — OpenAI-compatible (openai / vllm_server / openrouter)
# ---------------------------------------------------------------------------
async def _call_async(client, semaphore, model, sys_prompt, user_prompt,
                      temperature, max_tokens, retries=3, backoff=2):
    from agentic_clean.verify_loop import get_model_extra_body
    async with semaphore:
        for attempt in range(retries):
            try:
                t_kw = {"temperature": temperature} if temperature is not None else {}
                extra_body = get_model_extra_body(model)
                req_kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    "max_tokens": max_tokens,
                    **t_kw,
                }
                if extra_body is not None:
                    req_kwargs["extra_body"] = extra_body
                resp = await client.chat.completions.create(**req_kwargs)
                content = resp.choices[0].message.content
                return (content or "").strip()
            except Exception as e:
                if attempt < retries - 1:
                    await asyncio.sleep(backoff * (2 ** attempt))
                else:
                    return f"[ERROR] {e}"



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    ap = argparse.ArgumentParser(
        description="Single-pass S-expression generation with optional ISO hint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    ap.add_argument("--dataset",   required=True,
                    choices=["webqsp", "grailqa", "grailqapp"])
    ap.add_argument("--data_path", required=True,
                    help="Path to dill dataset file.")
    ap.add_argument("--filter_ids", default=None,
                    help="Optional JSON file of IDs to restrict evaluation to.")
    # ISO
    ap.add_argument("--iso_mode",  default="none",
                    choices=["none", "gold", "predicted"],
                    help="ISO hint mode: none | gold (computed) | predicted (from file).")
    ap.add_argument("--iso_pred_path", default=None,
                    help="JSON file of predicted ISO types (required when --iso_mode=predicted).")
    # Provider
    ap.add_argument("--provider",  default="openai",
                    choices=["openai", "openrouter", "vllm_server",
                             "huggingface", "gemini", "vllm", "claude"])
    ap.add_argument("--model",     default="gpt-4o")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    # async (openai / openrouter / vllm_server)
    ap.add_argument("--batch_size",     type=int, default=200)
    ap.add_argument("--max_concurrent", type=int, default=100)
    ap.add_argument("--api_base", default="http://localhost:8000/v1")
    ap.add_argument("--api_key",  default="EMPTY")
    ap.add_argument("--openrouter_api_key",
                    default=os.environ.get("OPENROUTER_API_KEY", ""))
    ap.add_argument("--anthropic_api_key",
                    default=os.environ.get("ANTHROPIC_API_KEY", ""))
    # vllm local
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--max_length", type=int, default=None)
    # Output
    ap.add_argument("--out_dir", required=True,
                    help="Directory for result JSON files.")
    ap.add_argument("--run_name", default=None,
                    help="Output file stem (auto-generated if omitted).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.iso_mode == "predicted" and not args.iso_pred_path:
        ap.error("--iso_pred_path is required when --iso_mode=predicted")

    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Output filename ─────────────────────────────────────────────────────
    model_safe = args.model.replace("/", "_")
    run_name = args.run_name or f"{args.dataset}_{model_safe}_fs_sexp_{args.iso_mode}"
    out_file = os.path.join(args.out_dir, f"{run_name}.json")

    # ── Load assets ─────────────────────────────────────────────────────────
    sys_prompt    = get_sys_prompt(args.dataset, args.iso_mode)
    user_template = get_user_template(args.dataset)
    data          = load_dataset(args.data_path, args.dataset)

    # ── Predicted ISO map ───────────────────────────────────────────────────
    # Only incorrect examples are kept; correct ones will be copied from the
    # corresponding gold-iso sexp run (auto-derived path).
    pred_iso_map = {}
    gold_sexp_results = {}   # results to copy correct examples from
    if args.iso_mode == "predicted":
        pred_iso_map = load_predicted_iso(args.iso_pred_path)
        print(f"[PredISO] {len(pred_iso_map)} incorrect examples loaded from {args.iso_pred_path}")

        # Auto-load gold-iso sexp results: {out_dir}/{dataset}_{model_safe}_fs_sexp_gold.json
        gold_sexp_path = os.path.join(args.out_dir, f"{args.dataset}_{model_safe}_fs_sexp_gold.json")
        if os.path.exists(gold_sexp_path):
            with open(gold_sexp_path, "r", encoding="utf-8") as f:
                gold_sexp_results = json.load(f)
            print(f"[PredISO] Loaded {len(gold_sexp_results)} gold-iso results from {gold_sexp_path}")
        else:
            print(f"[PredISO] No gold-iso results found at {gold_sexp_path}; correct examples will be skipped.")

    # ── Filter IDs ──────────────────────────────────────────────────────────
    allowed_ids = None
    if args.filter_ids:
        with open(args.filter_ids, "r", encoding="utf-8") as f:
            fd = json.load(f)
        if isinstance(fd, list):
            allowed_ids = {str(e.get("id", e.get("ID", ""))) for e in fd}
        elif isinstance(fd, dict):
            allowed_ids = {str(k) for k in fd}
        print(f"[Filter] Restricting to {len(allowed_ids)} IDs")

    # ── Resume ──────────────────────────────────────────────────────────────
    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"[Resume] {len(results)} existing results in {out_file}")
    else:
        results = {}

    # ── Build pending list ───────────────────────────────────────────────────
    entity_key = "entities" if args.dataset == "webqsp" else "head_entities"
    pending = []  # (qid, question, user_prompt_text, gold_sexp, iso_type)
    n_already = len(results)
    n_copied  = 0

    for idx, ex in enumerate(data, 1):
        qid = str(ex.get("ID", ex.get("id", f"ex_{idx}")))
        if qid in results:
            continue
        if allowed_ids is not None and qid not in allowed_ids:
            continue

        gold_sexp = ex.get("s_expression", "")
        if not isinstance(gold_sexp, str) or gold_sexp.strip().lower() == "none":
            continue

        entities = ex.get(entity_key, [])
        if not entities:
            print(f"[Skip] {qid}: no entities.")
            continue

        # Determine ISO type
        iso_type = None
        if args.iso_mode == "gold":
            try:
                iso_type = str(obtain_isomorphism_id(gold_sexp))
            except Exception as e:
                print(f"[Skip] {qid}: cannot compute gold ISO → {e}")
                continue
        elif args.iso_mode == "predicted":
            # Only process incorrectly-classified examples; copy correct ones
            # from the gold-iso run (same pattern as verify_loop.py pred_iso).
            if qid not in pred_iso_map:
                if qid in gold_sexp_results:
                    results[qid] = gold_sexp_results[qid]
                    n_copied += 1
                continue
            iso_type = pred_iso_map[qid]

        # Build user prompt
        try:
            user_prompt = render_user_prompt(
                user_template, ex["question"], entities, ex["kg_tuples"],
                args.dataset)
        except Exception as e:
            print(f"[Skip] {qid}: prompt build failed → {e}")
            continue

        if iso_type is not None:
            user_prompt = inject_iso_hint(user_prompt, iso_type)

        pending.append((qid, ex["question"], user_prompt, gold_sexp, iso_type))

    info_parts = [f"{len(pending)} to process", f"{n_already} already done"]
    if n_copied:
        info_parts.append(f"{n_copied} copied from gold-iso (correct pred_iso)")
    print(f"[Info] {', '.join(info_parts)}.")

    # ── Inference ────────────────────────────────────────────────────────────
    provider = args.provider.lower()

    if provider == "claude":
        # ── Claude Batch API path (avoids proxy interception) ─────────────────
        import anthropic, time, re as _re

        client = anthropic.Anthropic(api_key=args.anthropic_api_key)
        temp_kw = {"temperature": args.temperature} if args.temperature is not None else {}

        def _safe_id(qid: str) -> str:
            return _re.sub(r"[^a-zA-Z0-9_-]", "_", qid)[:64]

        # Build safe_id <-> qid mapping (custom_id must match ^[a-zA-Z0-9_-]{1,64}$)
        safe_to_qid = {}
        for i, (qid, _, _, _, _) in enumerate(pending):
            sid = _safe_id(qid)
            if sid in safe_to_qid and safe_to_qid[sid] != qid:
                sid = f"{sid[:60]}_{i}"
            safe_to_qid[sid] = qid
        qid_to_safe = {v: k for k, v in safe_to_qid.items()}

        for batch_start in range(0, len(pending), args.batch_size):
            batch = pending[batch_start: batch_start + args.batch_size]

            requests = [
                {
                    "custom_id": qid_to_safe[qid],
                    "params": {
                        "model":      args.model,
                        "max_tokens": args.max_new_tokens,
                        "system":     sys_prompt,
                        "messages":   [{"role": "user", "content": user_prompt_text}],
                        **temp_kw,
                    },
                }
                for qid, _, user_prompt_text, _, _ in batch
            ]

            api_batch = client.messages.batches.create(requests=requests)
            print(f"[Batch submitted] {api_batch.id} ({len(requests)} requests)")

            while api_batch.processing_status == "in_progress":
                time.sleep(60)
                api_batch = client.messages.batches.retrieve(api_batch.id)
                rc = api_batch.request_counts
                print(f"  [{api_batch.id}] processing={rc.processing} "
                      f"succeeded={rc.succeeded} errored={rc.errored}")

            for result in client.messages.batches.results(api_batch.id):
                qid = safe_to_qid[result.custom_id]
                if result.result.type == "succeeded":
                    blocks = result.result.message.content
                    raw = next((b.text for b in blocks if hasattr(b, "text")), "").strip()
                else:
                    raw = f"[ERROR] {result.result}"

                # find question/gold/iso_type from pending
                _, question, _, gold_sexp, iso_type = next(
                    p for p in batch if p[0] == qid
                )
                pred_sexp = extract_sexpr(raw)
                correct = bool(compute_exact_match(pred_sexp, gold_sexp))
                print(f"  {qid} | correct={correct}")
                results[qid] = {
                    "id":        qid,
                    "question":  question,
                    "pred_sexp": pred_sexp,
                    "gold_sexp": gold_sexp,
                    "correct":   correct,
                    "iso_type":  iso_type,
                    "iso_mode":  args.iso_mode,
                }

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            done = batch_start + len(batch)
            print(f"[Checkpoint] {done}/{len(pending)} → {out_file}")

    elif provider in {"openai", "openrouter", "vllm_server"}:
        # ── Async path ────────────────────────────────────────────────────────
        semaphore = asyncio.Semaphore(args.max_concurrent)

        from openai import AsyncOpenAI
        if provider == "openai":
            client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        elif provider == "openrouter":
            client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=args.openrouter_api_key,
            )
        else:
            client = AsyncOpenAI(base_url=args.api_base, api_key=args.api_key)

        for batch_start in range(0, len(pending), args.batch_size):
            batch = pending[batch_start: batch_start + args.batch_size]
            tasks = [
                asyncio.create_task(_call_async(
                    client, semaphore, args.model,
                    sys_prompt, user_prompt_text,
                    args.temperature, args.max_new_tokens,
                ))
                for _, _, user_prompt_text, _, _ in batch
            ]
            raw_outputs = await asyncio.gather(*tasks)

            for (qid, question, _, gold_sexp, iso_type), raw in zip(batch, raw_outputs):
                pred_sexp = extract_sexpr(raw)
                correct = bool(compute_exact_match(pred_sexp, gold_sexp))
                print(f"  {qid} | correct={correct}")
                results[qid] = {
                    "id":        qid,
                    "question":  question,
                    "pred_sexp": pred_sexp,
                    "gold_sexp": gold_sexp,
                    "correct":   correct,
                    "iso_type":  iso_type,
                    "iso_mode":  args.iso_mode,
                }

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            done = batch_start + len(batch)
            print(f"[Checkpoint] {done}/{len(pending)} → {out_file}")

    else:
        # ── Sync path (huggingface / gemini / vllm) ──────────────────────────
        from langchain.schema.messages import SystemMessage, HumanMessage
        llm = create_llm(
            provider, args.model, temperature=args.temperature,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_length,
            max_new_tokens=args.max_new_tokens,
        )
        for i, (qid, question, user_prompt_text, gold_sexp, iso_type) in enumerate(pending, 1):
            resp = llm.invoke([
                SystemMessage(content=sys_prompt),
                HumanMessage(content=user_prompt_text),
            ])
            raw = resp.content if hasattr(resp, "content") else str(resp)
            pred_sexp = extract_sexpr(raw)
            correct = bool(compute_exact_match(pred_sexp, gold_sexp))
            print(f"  [{i}/{len(pending)}] {qid} | correct={correct}")
            results[qid] = {
                "id":        qid,
                "question":  question,
                "pred_sexp": pred_sexp,
                "gold_sexp": gold_sexp,
                "correct":   correct,
                "iso_type":  iso_type,
                "iso_mode":  args.iso_mode,
            }
            if i % 50 == 0:
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                print(f"[Checkpoint] {i}/{len(pending)} → {out_file}")

    # ── Final save & summary ─────────────────────────────────────────────────
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    total   = len(results)
    correct = sum(1 for v in results.values() if v.get("correct"))
    acc     = correct / total if total else 0.0
    print(f"\n=== {run_name} ===")
    print(f"Exact match: {correct}/{total} = {acc:.4f}")
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
