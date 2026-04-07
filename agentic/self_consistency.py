#!/usr/bin/env python3
"""
agentic_clean/self_consistency.py

Self-consistency selection across none-1 to none-5 runs.

For each (dataset, model):
  1. Load none-1 … none-5 result files from results_sexp
  2. For each question, majority vote on normalized pred_sexp strings
     - Winner = the answer with the most votes, if it is strictly ahead of all others
  3. If no majority: call the same model as LLM judge
     - Give it the question + KG tuples + shuffled candidates → pick rank-1
     - "correct" is taken directly from whichever none-x produced that answer
  4. Save {dataset}_{model}_fs_sexp_SC.json  with fields:
       id, question, pred_sexp, gold_sexp, correct, iso_mode="none",
       sc_method ("majority"|"judge"), vote_counts

Usage:
  python -m agentic_clean.self_consistency                      # all pairs
  python -m agentic_clean.self_consistency --dataset webqsp --model moonshotai_kimi-k2.5
"""

import os, sys, re, json, random, argparse, asyncio

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agentic_clean.verify_loop import (
    load_dataset, render_user_prompt, read_file,
    extract_sexpr, seed_everything,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESULTS_DIR = "agentic_clean/results_sexp"
PROMPTS_DIR = "agentic_clean/prompts"

DATA_PATHS = {
    "webqsp":    "/data/shire/projects/GenQA/ISOKBQA/isomorphism_pred_KBQA/data/webqsp/KBQA_walklet_edge_label_768_dim_data.dill",
    "grailqa":   "/home/zhenwu/isomorphism_pred_KBQA/agentic/grailqa_dev_sampled_1000.dill",
    "grailqapp": "/data/shire/projects/GenQA/ISOKBQA/isomorphism_pred_KBQA/data/grailqa/KBQA_walklet_edge_label_768_dim_data_test.dill",
}

SYS_PROMPTS = {
    "webqsp":    "fs_sys_prompt_webqsp.txt",
    "grailqa":   "fs_sys_prompt_grailqa.txt",
    "grailqapp": "fs_sys_prompt_grailqa.txt",
}

TEMPLATES = {
    "webqsp":    "user_prompt.txt",
    "grailqa":   "user_prompt_grailqa.txt",
    "grailqapp": "user_prompt_grailqa.txt",
}

# model_safe → (provider, model_id, extra_kwargs)
PROVIDER_MAP = {
    "moonshotai_kimi-k2.5":              ("openrouter", "moonshotai/kimi-k2.5",                  {}),
    "gpt-5-mini":                        ("openai",     "gpt-5-mini",                             {}),
    "claude-sonnet-4-6":                 ("claude",     "claude-sonnet-4-6",                      {}),
    "Qwen_Qwen3-14B":                    ("vllm_server","Qwen/Qwen3-14B",                         {"api_base": "http://localhost:8000/v1", "api_key": "EMPTY"}),
    "Qwen_Qwen3-30B-A3B-Instruct-2507":  ("vllm_server","Qwen/Qwen3-30B-A3B-Instruct-2507",      {"api_base": "http://localhost:8000/v1", "api_key": "EMPTY"}),
    # "google_gemma-3-27b-it":             ("vllm_server","google/gemma-3-27b-it",                  {"api_base": "http://localhost:8000/v1", "api_key": "EMPTY"}),
}


# ---------------------------------------------------------------------------
# Majority vote
# ---------------------------------------------------------------------------
def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def majority_vote(candidates):
    """
    candidates: list of {"pred_sexp", "correct", "gold_sexp", "question", "source"}
    Returns (winner_entry, vote_counts) or (None, vote_counts) if no majority.
    """
    counts = {}   # norm_sexp → count
    first  = {}   # norm_sexp → first entry with that sexp
    for c in candidates:
        k = normalize(c["pred_sexp"])
        counts[k] = counts.get(k, 0) + 1
        if k not in first:
            first[k] = c

    best_k = max(counts, key=counts.__getitem__)
    best_count = counts[best_k]
    # unique winner: one answer has strictly more votes than all others
    if sum(1 for v in counts.values() if v == best_count) == 1:
        return first[best_k], counts
    return None, counts


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------
JUDGE_INSTRUCTION = (
    "\nBelow are {n} candidate S-expressions for this question. "
    "Pick the one most likely to be correct.\n"
    "Respond with ONLY the number of the best candidate (1–{n}).\n\n"
    "{candidates_block}"
)


def build_judge_prompt(base_user_prompt: str, shuffled_sexps: list) -> str:
    """Replace **S-expression:** with the judge instruction + candidates."""
    block = "\n".join(f"Candidate {i+1}: {s}" for i, s in enumerate(shuffled_sexps))
    instruction = JUDGE_INSTRUCTION.format(n=len(shuffled_sexps), candidates_block=block)
    marker = "**S-expression:**"
    if marker in base_user_prompt:
        return base_user_prompt.replace(marker, instruction)
    return base_user_prompt + instruction


def parse_judge_response(response: str, n: int) -> int:
    """Return 0-based index of top-ranked candidate, or 0 as fallback."""
    m = re.search(r"\b([1-9])\b", response)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < n:
            return idx
    return 0


# ---------------------------------------------------------------------------
# Async LLM call (OpenAI-compatible + Claude)
# ---------------------------------------------------------------------------
async def _call(client, semaphore, provider, model, sys_prompt, user_prompt,
                temperature=0.0, max_tokens=32, retries=3, backoff=2):
    from agentic_clean.verify_loop import get_model_extra_body
    async with semaphore:
        for attempt in range(retries):
            try:
                if provider == "claude":
                    t_kw = {"temperature": temperature} if temperature is not None else {}
                    resp = await client.messages.create(
                        model=model, max_tokens=max_tokens,
                        system=sys_prompt,
                        messages=[{"role": "user", "content": user_prompt}],
                        **t_kw,
                    )
                    return resp.content[0].text.strip()
                else:
                    t_kw = {"temperature": temperature} if temperature is not None else {}
                    extra_body = get_model_extra_body(model)
                    req = {"model": model, "max_tokens": max_tokens,
                           "messages": [{"role": "system", "content": sys_prompt},
                                        {"role": "user",   "content": user_prompt}],
                           **t_kw}
                    if extra_body:
                        req["extra_body"] = extra_body
                    resp = await client.chat.completions.create(**req)
                    return resp.choices[0].message.content.strip()
            except Exception as e:
                if attempt < retries - 1:
                    await asyncio.sleep(backoff * (2 ** attempt))
                else:
                    return f"[ERROR] {e}"


# ---------------------------------------------------------------------------
# Build async client
# ---------------------------------------------------------------------------
def make_client(provider, model_safe):
    _, _, extra = PROVIDER_MAP[model_safe]
    if provider == "claude":
        from anthropic import AsyncAnthropic
        return AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    elif provider == "openai":
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    elif provider == "openrouter":
        from openai import AsyncOpenAI
        return AsyncOpenAI(base_url="https://openrouter.ai/api/v1",
                           api_key=os.environ.get("OPENROUTER_API_KEY", ""))
    else:  # vllm_server
        from openai import AsyncOpenAI
        return AsyncOpenAI(base_url=extra.get("api_base", "http://localhost:8000/v1"),
                           api_key=extra.get("api_key", "EMPTY"))


# ---------------------------------------------------------------------------
# Process one (dataset, model) pair
# ---------------------------------------------------------------------------
async def process_pair(dataset, model_safe, args):
    out_name = f"{dataset}_{model_safe}_fs_sexp_SC.json"
    out_path = os.path.join(RESULTS_DIR, out_name)

    if os.path.exists(out_path) and not args.overwrite:
        print(f"[Skip] {out_name} already exists")
        return

    # ── Load none-1 … none-5 ────────────────────────────────────────────────
    none_data = {}   # {1..5: {qid: entry}}
    for i in range(1, 6):
        fpath = os.path.join(RESULTS_DIR, f"{dataset}_{model_safe}_fs_sexp_none-{i}.json")
        if os.path.exists(fpath):
            with open(fpath) as f:
                none_data[i] = json.load(f)

    if not none_data:
        print(f"[Skip] no none files found for {dataset} / {model_safe}")
        return

    all_qids = set.union(*[set(d.keys()) for d in none_data.values()])
    print(f"[{dataset}/{model_safe}] {len(none_data)} none files, {len(all_qids)} qids")

    # ── Resume ───────────────────────────────────────────────────────────────
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
    else:
        results = {}

    # ── Pass 1: majority vote ────────────────────────────────────────────────
    need_judge = []   # (qid, candidates_list)

    for qid in sorted(all_qids):
        if qid in results:
            continue
        candidates = []
        for i, d in none_data.items():
            if qid in d:
                e = d[qid]
                candidates.append({
                    "pred_sexp": e.get("pred_sexp", ""),
                    "correct":   e.get("correct", False),
                    "gold_sexp": e.get("gold_sexp", ""),
                    "question":  e.get("question", ""),
                    "source":    i,
                })

        winner, vote_counts = majority_vote(candidates)
        if winner:
            results[qid] = {
                "id":         qid,
                "question":   winner["question"],
                "pred_sexp":  winner["pred_sexp"],
                "gold_sexp":  winner["gold_sexp"],
                "correct":    winner["correct"],
                "iso_type":   None,
                "iso_mode":   "none",
                "sc_method":  "majority",
                "vote_counts": {k: v for k, v in vote_counts.items()},
            }
        else:
            need_judge.append((qid, candidates))

    n_majority = len(results)
    print(f"[{dataset}/{model_safe}] majority: {n_majority}, need judge: {len(need_judge)}")

    # ── Pass 2: LLM judge ────────────────────────────────────────────────────
    if need_judge:
        if model_safe not in PROVIDER_MAP:
            print(f"[Warn] no provider mapping for {model_safe}, skipping judge calls")
        else:
            provider, model_id, _ = PROVIDER_MAP[model_safe]

            sys_prompt    = read_file(os.path.join(PROMPTS_DIR, SYS_PROMPTS[dataset]))
            user_template = read_file(os.path.join(PROMPTS_DIR, TEMPLATES[dataset]))

            # Load dill for KG tuples
            dill_data = load_dataset(DATA_PATHS[dataset], dataset)
            kg_index = {}   # qid -> ex
            entity_key = "entities" if dataset == "webqsp" else "head_entities"
            for ex in dill_data:
                qid_ex = str(ex.get("ID", ex.get("id", "")))
                kg_index[qid_ex] = ex

            client    = make_client(provider, model_safe)
            semaphore = asyncio.Semaphore(100)

            async def judge_one(qid, candidates):
                ex = kg_index.get(qid)
                if ex is None:
                    return qid, candidates[0]   # fallback: pick first

                try:
                    base_prompt = render_user_prompt(
                        user_template, ex["question"],
                        ex.get(entity_key, []), ex["kg_tuples"], dataset)
                except Exception:
                    return qid, candidates[0]

                shuffled = candidates.copy()
                random.shuffle(shuffled)
                sexps = [c["pred_sexp"] for c in shuffled]
                judge_prompt = build_judge_prompt(base_prompt, sexps)

                response = await _call(client, semaphore, provider, model_id,
                                       sys_prompt, judge_prompt,
                                       temperature=0.0, max_tokens=16)
                idx = parse_judge_response(response, len(shuffled))
                return qid, shuffled[idx]

            tasks = [asyncio.create_task(judge_one(qid, cands))
                     for qid, cands in need_judge]
            judge_results = await asyncio.gather(*tasks)

            for qid, chosen in judge_results:
                vote_counts = {}
                for c in dict(need_judge)[qid]:
                    k = normalize(c["pred_sexp"])
                    vote_counts[k] = vote_counts.get(k, 0) + 1
                results[qid] = {
                    "id":         qid,
                    "question":   chosen["question"],
                    "pred_sexp":  chosen["pred_sexp"],
                    "gold_sexp":  chosen["gold_sexp"],
                    "correct":    chosen["correct"],
                    "iso_type":   None,
                    "iso_mode":   "none",
                    "sc_method":  "judge",
                    "vote_counts": vote_counts,
                }

    # ── Save ─────────────────────────────────────────────────────────────────
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    total   = len(results)
    correct = sum(1 for v in results.values() if v.get("correct"))
    n_judge = sum(1 for v in results.values() if v.get("sc_method") == "judge")
    print(f"[{dataset}/{model_safe}] EM={correct}/{total}={correct/total:.4f}  "
          f"majority={total-n_judge}  judge={n_judge}  → {out_name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",  default=None)
    ap.add_argument("--model",    default=None, help="model_safe name, e.g. moonshotai_kimi-k2.5")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)

    # Discover all (dataset, model_safe) pairs from existing none-1 files
    pairs = set()
    for fname in os.listdir(RESULTS_DIR):
        m = re.match(r"^(webqsp|grailqa|grailqapp)_(.+)_fs_sexp_none-1\.json$", fname)
        if m:
            pairs.add((m.group(1), m.group(2)))

    if args.dataset:
        pairs = {(d, m) for d, m in pairs if d == args.dataset}
    if args.model:
        pairs = {(d, m) for d, m in pairs if m == args.model}

    for dataset, model_safe in sorted(pairs):
        await process_pair(dataset, model_safe, args)


if __name__ == "__main__":
    asyncio.run(main())
