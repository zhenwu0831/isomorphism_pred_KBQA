#!/usr/bin/env python3
"""
agentic_clean/verify_loop.py

Unified agentic S-expression generation loop with configurable verifier and feedback.
Replaces: main.py, main_grailqa.py, main_iso_fb.py, main_grailqa_iso_fb.py

Key CLI knobs (nothing is hardcoded in this script):
  --dataset       {webqsp, grailqa, grailqapp}
  --verifier      {none, exe, double}
                    none   – always pass  (baseline / no loop)
                    exe    – executability only
                    double – ISO-type match AND executability
  --feedback      {plain, iso}
                    plain  – generic "The answer is wrong. Revise…"
                    iso    – adds reasoning-pattern guidance when ISO type is wrong
  --out_dir       PATH   explicit output directory  (required)
  --run_name      STR    filename stem; default auto-generated from key settings
  --provider      {openai, huggingface, gemini, vllm, vllm_server}
  --model         STR
  --sys_prompt    PATH
  --template      PATH
  --api_base      URL    vllm_server endpoint  (default: http://localhost:8000/v1)
  --api_key       STR    vllm_server api key   (default: EMPTY)
  --tensor_parallel_size INT   vLLM local only
  --max_length    INT    vLLM context window; falls back to tokenizer config
  --max_new_tokens INT   vLLM generation budget (default 512)

Example:
  python -m agentic_clean.verify_loop \\
      --dataset grailqa \\
      --data_path /path/to/data.dill \\
      --verifier double \\
      --feedback iso \\
      --sys_prompt agentic/fs_sys_prompt_grailqa_iso_verifier.txt \\
      --template agentic/user_prompt_grailqa.txt \\
      --out_dir agentic_clean/results/grailqa_double_iso \\
      --provider vllm \\
      --model Qwen/Qwen3-30B-A3B-Instruct-2507 \\
      --tensor_parallel_size 2 \\
      --max_length 32768 \\
      --temperature 1.0 \\
      --max_attempts 5
"""

import os
import sys
import argparse
import asyncio
import dill
import json
import re
import torch
from typing import List, Tuple

from jinja2 import Template
from langchain.schema import BaseMessage
from langchain.schema.messages import HumanMessage, AIMessage, SystemMessage

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
import logging
import transformers
transformers.logging.set_verbosity_error()
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from compute_isomorphisms.code.get_isomorphism import obtain_isomorphism_id
from isomorphism_pred_KBQA.utils.evaluation import compute_exact_match
from compute_isomorphisms.code.logic_form_util import lisp_to_sparql
from compute_isomorphisms.code.sparql_executor import execute_query

import torch._dynamo
torch._dynamo.config.recompile_limit = 128


# ---------------------------------------------------------------------------
# ISO definitions
# ---------------------------------------------------------------------------
ISO_DEFINITIONS = {
    "T-0": "A single-hop path from the constraint to the answer.",
    "T-1": "A two-hop path from the constraint to the answer.",
    "T-2": "Two single-hop paths arising from two different constraints and converging to the same answer.",
    "T-3": "Two paths (one single-hop and another two-hop) arising from two different constraints and converging to the same answer.",
    "T-4": "Two two-hop paths arising from two different constraints and converging to an intermediate common node before reaching the answer.",
    "T-5": "A three-hop path from the constraint to the answer.",
}


# ---------------------------------------------------------------------------
# 1) LLM factory
# ---------------------------------------------------------------------------
def _role(m: BaseMessage) -> str:
    if isinstance(m, SystemMessage): return "system"
    if isinstance(m, AIMessage):     return "assistant"
    return "user"


class _Resp:
    def __init__(self, content: str):
        self.content = content

def get_model_extra_body(model: str):
    model_l = model.lower()

    # Apply only to GLM-4.7-Flash family
    if "glm-4.7-flash" in model_l or "glm-4.7-flash-fp8" in model_l:
        # Replace "disabled" below with the exact disable value expected by the provider,
        # if their docs use a different literal.
        return {
            "thinking": {
                "type": "disabled"
            }
        }
    elif "glm-4.7-fp8" in model_l or "qwen3-14b" in model_l:
        return {"chat_template_kwargs": {"enable_thinking": False}}

    elif "kimi" in model_l:
        return {"chat_template_kwargs": {"thinking": False}, 
                "provider": {"sort": "latency"}
            }

    return None

def create_llm(provider: str, model: str, temperature=None, **kwargs):
    provider = provider.lower()
    temp_kwargs = {"temperature": temperature} if temperature is not None else {}

    if provider == "openai":
        from langchain_community.chat_models import ChatOpenAI
        return ChatOpenAI(model=model, **temp_kwargs)

    elif provider == "huggingface":
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline

        tok = AutoTokenizer.from_pretrained(model, use_fast=True)
        mdl = AutoModelForCausalLM.from_pretrained(model, device_map="auto", torch_dtype="auto")
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        if getattr(tok, "chat_template", None) is None:
            raise ValueError(
                f"Tokenizer for '{model}' has no chat_template. Use an Instruct/Chat checkpoint."
            )
        pipe = hf_pipeline("text-generation", model=mdl, tokenizer=tok)

        class _HFAdapter:
            def invoke(self, messages):
                chat = [{"role": _role(m), "content": m.content} for m in messages]
                prompt = tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
                do_sample = temperature is not None and temperature > 0
                out = pipe(
                    prompt, max_new_tokens=256,
                    do_sample=do_sample,
                    temperature=(temperature if do_sample else None),
                )[0]["generated_text"]
                if out.startswith(prompt):
                    out = out[len(prompt):]
                return _Resp(out)

        return _HFAdapter()

    elif provider == "gemini":
        from google import genai
        from google.genai import types

        class _GeminiAdapter:
            def __init__(self):
                self.client = genai.Client()

            def invoke(self, messages):
                system_text, contents = None, []
                for m in messages:
                    if isinstance(m, SystemMessage):
                        system_text = m.content
                    elif isinstance(m, HumanMessage):
                        contents.append(types.Content(role="user",  parts=[types.Part(text=m.content)]))
                    elif isinstance(m, AIMessage):
                        contents.append(types.Content(role="model", parts=[types.Part(text=m.content)]))
                cfg = types.GenerateContentConfig(
                    system_instruction=system_text, **temp_kwargs)
                resp = self.client.models.generate_content(
                    model=model, contents=contents, config=cfg)
                return _Resp(resp.text)

        return _GeminiAdapter()

    elif provider == "vllm":
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer as _AT

        tensor_parallel_size = kwargs.get("tensor_parallel_size", 1)
        max_new_tokens       = kwargs.get("max_new_tokens", 512)
        max_model_len        = kwargs.get("max_model_len", None)

        print("[DEBUG] entered local vllm provider branch")
        print(f"[DEBUG] model={model}")
        print(f"[DEBUG] tensor_parallel_size={tensor_parallel_size}")
        print(f"[DEBUG] max_model_len={max_model_len}")
        print(f"[DEBUG] max_new_tokens={max_new_tokens}")

        tok = _AT.from_pretrained(model, trust_remote_code=True)
        if max_model_len is None:
            max_model_len = getattr(tok, "model_max_length", None)
            if max_model_len is None or max_model_len > 1_000_000:
                raise ValueError(
                    "Cannot determine context length from tokenizer. Pass --max_length explicitly."
                )

        vllm_llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            enforce_eager=True,
            trust_remote_code=True,
        )
        sp = SamplingParams(max_tokens=max_new_tokens, **temp_kwargs)

        class _VLLMAdapter:
            def _num_tokens(self, msgs):
                chat = [{"role": _role(m), "content": m.content} for m in msgs]
                return len(tok.apply_chat_template(
                    chat,
                    add_generation_prompt=True,
                    tokenize=True,
                ))

            def _fit_messages_to_context(self, messages):
                # Leave extra slack because vLLM's final accounting can be a bit higher
                safety_margin = 256
                budget = max_model_len - max_new_tokens - safety_margin
                if budget <= 0:
                    raise ValueError(
                        f"Budget <= 0: max_model_len={max_model_len}, "
                        f"max_new_tokens={max_new_tokens}, safety_margin={safety_margin}"
                    )

                new_messages = list(messages)

                # First shrink the first user message, which is your large KG prompt.
                first_user_idx = None
                for i, m in enumerate(new_messages):
                    if isinstance(m, HumanMessage):
                        first_user_idx = i
                        break

                if first_user_idx is not None:
                    text = new_messages[first_user_idx].content
                    ids = tok.encode(text, add_special_tokens=False)

                    lo, hi = 0, len(ids)
                    best = 0
                    while lo <= hi:
                        mid = (lo + hi) // 2
                        trial = list(new_messages)
                        trial[first_user_idx] = HumanMessage(
                            content=tok.decode(ids[:mid], skip_special_tokens=True)
                        )
                        if self._num_tokens(trial) <= budget:
                            best = mid
                            lo = mid + 1
                        else:
                            hi = mid - 1

                    new_messages[first_user_idx] = HumanMessage(
                        content=tok.decode(ids[:best], skip_special_tokens=True)
                    )

                # If still too long, drop oldest non-system turns.
                while self._num_tokens(new_messages) > budget:
                    removable_idx = None
                    for i, m in enumerate(new_messages):
                        if not isinstance(m, SystemMessage):
                            removable_idx = i
                            break
                    if removable_idx is None:
                        raise ValueError("Cannot shrink messages enough to fit context window.")
                    del new_messages[removable_idx]

                return new_messages

            def invoke(self, messages):
                messages = self._fit_messages_to_context(messages)
                chat = [{"role": _role(m), "content": m.content} for m in messages]
                out = vllm_llm.chat([chat], sampling_params=sp)[0].outputs[0].text.strip()
                return _Resp(out)

        return _VLLMAdapter()

    elif provider == "vllm_server":
        from openai import OpenAI

        api_base       = kwargs.get("api_base", "http://localhost:8000/v1")
        api_key        = kwargs.get("api_key", "EMPTY")
        max_new_tokens = kwargs.get("max_new_tokens", 512)

        client = OpenAI(base_url=api_base, api_key=api_key)

        class _VLLMServerAdapter:
            def invoke(self, messages):
                chat = [{"role": _role(m), "content": m.content} for m in messages]
                extra_body = get_model_extra_body(model)

                req_kwargs = {
                    "model": model,
                    "messages": chat,
                    "max_tokens": max_new_tokens,
                    **temp_kwargs,
                }
                if extra_body is not None:
                    req_kwargs["extra_body"] = extra_body

                resp = client.chat.completions.create(**req_kwargs)
                return _Resp(resp.choices[0].message.content.strip())

        return _VLLMServerAdapter()

    else:
        raise ValueError(
            f"Unknown provider: {provider!r}. "
            "Choose from: openai, huggingface, gemini, vllm, vllm_server"
        )


def seed_everything(seed: int = 42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# 2) Data loader
# ---------------------------------------------------------------------------
def load_dataset(dill_path: str, dataset: str) -> List[dict]:
    with open(dill_path, "rb") as f:
        data = dill.load(f)
    if isinstance(data, dict) and "test" in data:
        data = data["test"]
    if not isinstance(data, list):
        raise ValueError("Expected a list (or dict with 'test' key) in dill file.")
    entity_key = "entities" if dataset == "webqsp" else "head_entities"
    required = ("question", "s_expression", entity_key, "kg_tuples")
    for ex in data:
        if not all(k in ex for k in required):
            raise ValueError(f"Example missing required keys. Has: {list(ex.keys())}")
    return data


# ---------------------------------------------------------------------------
# 3) Prompt building
# ---------------------------------------------------------------------------
def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def clean_value(v: str) -> str:
    v = str(v)
    return v[1:-1] if len(v) >= 2 and v[0] == '"' and v[-1] == '"' else v


def format_kg_tuples(tuples: List[Tuple]) -> str:
    lines = [f"[{clean_value(s)}, {clean_value(p)}, {clean_value(o)}]" for s, p, o in tuples]
    return "[\n  " + ",\n  ".join(lines) + "\n]"


def render_user_prompt(template_str: str, question: str, entities: list,
                       kg_tuples: list, dataset: str) -> str:
    tpl = Template(template_str)
    if not entities:
        raise ValueError("Empty entities list.")
    kg_str = format_kg_tuples(kg_tuples)

    if dataset == "webqsp":
        # entities[0] = (name, id)
        starting_entity    = entities[0][0]
        starting_entity_id = entities[0][1]
        return tpl.render(
            question=question,
            starting_entity=starting_entity,
            starting_entity_id=starting_entity_id,
            kg_tuples=kg_str,
        )
    else:
        # grailqa / grailqapp: entities = [(id, type, label), ...]
        parts = [f"{clean_value(e[0])} ({clean_value(e[1])}, {clean_value(e[2])})"
                 for e in entities]
        return tpl.render(
            question=question,
            starting_entity_list=", ".join(parts),
            kg_tuples=kg_str,
        )


# ---------------------------------------------------------------------------
# 4) S-expression extractor
# ---------------------------------------------------------------------------
def extract_sexpr(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    s = text.strip()
    m = re.search(r"```[a-zA-Z0-9_-]*\s*([\s\S]*?)```", s)
    if m:
        s = m.group(1).strip()
    last = s.rfind(")")
    if last != -1:
        depth = 0
        first = -1
        for i in range(last, -1, -1):
            if s[i] == ")":
                depth += 1
            elif s[i] == "(":
                depth -= 1
                if depth == 0:
                    first = i
                    break
        if first != -1:
            return s[first:last + 1].strip()
        return s[:last + 1].strip()
    for line in s.splitlines():
        line = line.strip()
        if line:
            return line
    return s


# ---------------------------------------------------------------------------
# 5) Verifier
# ---------------------------------------------------------------------------
def run_verifier(generated: str, gold: str, verifier_mode: str,
                 target_iso: str = None) -> dict:
    """
    verifier_mode:
      none   – always pass (no loop effect)
      exe    – pass iff SPARQL executes with non-empty results
      double – pass iff ISO type matches target AND SPARQL executes

    target_iso: if provided, used as the ISO target for the double check instead
                of the ISO derived from gold (allows using predicted ISO from
                an external classifier rather than gold ISO).
    """
    gold_iso = target_iso if target_iso is not None else str(obtain_isomorphism_id(gold))

    if verifier_mode == "none":
        return {"pass": True, "output": generated,
                "gold_iso": gold_iso, "pred_iso": "N/A", "answers": []}

    try:
        pred_iso = obtain_isomorphism_id(generated)
        try:
            answers = execute_query(lisp_to_sparql(generated))
        except Exception:
            answers = []

        if verifier_mode == "exe":
            passed = len(answers) > 0
        elif verifier_mode == "double":
            passed = (gold_iso == str(pred_iso)) and (len(answers) > 0)
        else:
            raise ValueError(f"Unknown verifier mode: {verifier_mode!r}")

        return {"pass": passed, "output": generated,
                "gold_iso": gold_iso, "pred_iso": str(pred_iso), "answers": answers}

    except Exception as e:
        return {"pass": False, "output": generated, "error": str(e),
                "gold_iso": gold_iso, "pred_iso": "N/A", "answers": []}


# ---------------------------------------------------------------------------
# 6) Feedback builder
# ---------------------------------------------------------------------------
def build_feedback(verifier_result: dict, feedback_mode: str) -> str:
    executable = len(verifier_result.get("answers", [])) > 0
    exec_sentence = (
        "Your previous S-expression was executable but returned incorrect results."
        if executable else
        "Your previous S-expression was not executable."
    )

    if feedback_mode == "iso":
        pred_iso = verifier_result.get("pred_iso", "N/A")
        gold_iso = verifier_result.get("gold_iso", "N/A")
        if pred_iso != gold_iso and gold_iso in ISO_DEFINITIONS:
            return (
                f"The answer is wrong because {exec_sentence} belongs to reasoning pattern {pred_iso}. "
                f"The correct reasoning pattern for this question is {gold_iso}: {ISO_DEFINITIONS[gold_iso]} "
                f"Revise and output only the correct S-expression corresponding to the {gold_iso} reasoning pattern."
            )

    return f"The answer is wrong. {exec_sentence} Revise and output only the correct S-expression."


# ---------------------------------------------------------------------------
# 7a) Sync loop  (huggingface / gemini / vllm)
# ---------------------------------------------------------------------------
def run_one(llm, sys_prompt: str, user_prompt: str, gold_sexp: str,
            max_attempts: int, verifier_mode: str, feedback_mode: str,
            target_iso: str = None) -> dict:
    history: List[BaseMessage] = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=user_prompt),
    ]
    last, last_raw, attempts_log = None, "", []

    for i in range(max_attempts):
        resp     = llm.invoke(history)
        last_raw = resp.content if hasattr(resp, "content") else str(resp)
        gen_text = extract_sexpr(last_raw)
        history.append(AIMessage(content=gen_text))

        last = run_verifier(gen_text, gold_sexp, verifier_mode, target_iso=target_iso)
        attempts_log.append({
            "attempt":  i + 1,
            "raw_output": last_raw,
            "output":   gen_text,
            "pred_iso": last.get("pred_iso", ""),
            "pass":     last["pass"],
            "answers":  last.get("answers", []),
        })

        if last["pass"]:
            return _make_result("success", gen_text, last_raw, i + 1,
                                attempts_log, gold_sexp, last)

        if i < max_attempts - 1:
            history.append(HumanMessage(content=build_feedback(last, feedback_mode)))

    return _make_result("failure", last.get("output", ""), last_raw, max_attempts,
                        attempts_log, gold_sexp, last)


# ---------------------------------------------------------------------------
# 7b) Async loop  (openai / vllm_server)
# ---------------------------------------------------------------------------
async def run_one_async(client, semaphore, model: str, temperature,
                        sys_prompt: str, user_prompt: str, gold_sexp: str,
                        max_attempts: int, verifier_mode: str, feedback_mode: str,
                        backoff: int = 2, target_iso: str = None) -> dict:
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    last, last_raw, attempts_log = None, "", []

    async with semaphore:
        for i in range(max_attempts):
            for retry in range(3):
                try:
                    t_kw = {"temperature": temperature} if temperature is not None else {}
                    extra_body = get_model_extra_body(model)

                    req_kwargs = {
                        "model": model,
                        "messages": messages,
                        **t_kw,
                    }
                    if extra_body is not None:
                        req_kwargs["extra_body"] = extra_body

                    response = await client.chat.completions.create(**req_kwargs)
                    last_raw = response.choices[0].message.content.strip()
                    break
                except Exception as e:
                    if retry < 2:
                        await asyncio.sleep(backoff * (2 ** retry))
                    else:
                        last_raw = f"[ERROR] {e}"

            gen_text = extract_sexpr(last_raw)
            messages.append({"role": "assistant", "content": gen_text})

            last = run_verifier(gen_text, gold_sexp, verifier_mode, target_iso=target_iso)
            attempts_log.append({
                "attempt":  i + 1,
                "raw_output": last_raw,
                "output":   gen_text,
                "pred_iso": last.get("pred_iso", ""),
                "pass":     last["pass"],
                "answers":  last.get("answers", []),
            })

            if last["pass"]:
                return _make_result("success", gen_text, last_raw, i + 1,
                                    attempts_log, gold_sexp, last)

            if i < max_attempts - 1:
                messages.append({"role": "user",
                                 "content": build_feedback(last, feedback_mode)})

    return _make_result("failure", last.get("output", ""), last_raw, max_attempts,
                        attempts_log, gold_sexp, last)


def _make_result(status, output, raw_output, attempts, attempts_log, gold_sexp, last):
    return {
        "status":       status,
        "output":       output,
        "raw_output":   raw_output,
        "attempts":     attempts,
        "attempts_log": attempts_log,
        "gold_sexp":    gold_sexp,
        "gold_iso":     (last or {}).get("gold_iso", ""),
        "pred_iso":     (last or {}).get("pred_iso", ""),
        "answers":      (last or {}).get("answers", []),
        "error":        (last or {}).get("error", ""),
    }


# ---------------------------------------------------------------------------
# 7c) Claude Batch API path
#
# Multi-turn loop maps to sequential batch rounds:
#   Round 1 → submit all pending as one batch → poll → verify
#   Round 2 → submit only failed examples with feedback appended → poll → verify
#   ...up to max_attempts rounds
# ---------------------------------------------------------------------------
def run_claude_batch(api_key: str, model: str, max_tokens: int,
                     sys_prompt: str, pending: list,
                     max_attempts: int, verifier_mode: str, feedback_mode: str,
                     results: dict, out_file: str, temperature=None):
    import time
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    # Claude custom_id must match ^[a-zA-Z0-9_-]{1,64}$
    # Build a sanitized-id ↔ original-qid mapping
    import re as _re
    def _safe_id(qid: str) -> str:
        return _re.sub(r"[^a-zA-Z0-9_-]", "_", qid)[:64]

    # Ensure sanitized IDs are unique (append index if collision)
    safe_to_qid = {}
    for i, (qid, _, _, _) in enumerate(pending):
        sid = _safe_id(qid)
        if sid in safe_to_qid and safe_to_qid[sid] != qid:
            sid = f"{sid[:60]}_{i}"
        safe_to_qid[sid] = qid
    qid_to_safe = {v: k for k, v in safe_to_qid.items()}

    # per-example state: messages is Claude-format (user/assistant only; system is separate)
    convs = {}
    for qid, user_prompt, gold, target_iso in pending:
        convs[qid] = {
            "messages":    [{"role": "user", "content": user_prompt}],
            "gold":        gold,
            "target_iso":  target_iso,
            "attempts":    0,
            "attempts_log": [],
            "done":        False,
        }

    for round_num in range(max_attempts):
        active = [(qid, c) for qid, c in convs.items() if not c["done"]]
        if not active:
            break

        # Submit batch
        temp_kw = {"temperature": temperature} if temperature is not None else {}
        requests = [
            {
                "custom_id": qid_to_safe[qid],
                "params": {
                    "model":      model,
                    "max_tokens": max_tokens,
                    "system":     sys_prompt,
                    "messages":   conv["messages"],
                    **temp_kw,
                },
            }
            for qid, conv in active
        ]
        batch = client.messages.batches.create(requests=requests)
        print(f"[Round {round_num+1}/{max_attempts}] Batch {batch.id} submitted "
              f"({len(requests)} requests)")

        # Poll until ended
        while batch.processing_status == "in_progress":
            time.sleep(60)
            batch = client.messages.batches.retrieve(batch.id)
            rc = batch.request_counts
            print(f"  [{batch.id}] processing={rc.processing}  "
                  f"succeeded={rc.succeeded}  errored={rc.errored}")

        # Process results
        for result in client.messages.batches.results(batch.id):
            qid  = safe_to_qid[result.custom_id]
            conv = convs[qid]

            if result.result.type == "succeeded":
                content_blocks = result.result.message.content
                raw = next((b.text for b in content_blocks if hasattr(b, "text")), "").strip()
            else:
                raw = f"[ERROR] {result.result}"

            gen_text = extract_sexpr(raw)
            conv["messages"].append({"role": "assistant", "content": gen_text})
            conv["attempts"] += 1

            vr = run_verifier(gen_text, conv["gold"], verifier_mode,
                              target_iso=conv.get("target_iso"))
            conv["attempts_log"].append({
                "attempt":    conv["attempts"],
                "raw_output": raw,
                "output":     gen_text,
                "pred_iso":   vr.get("pred_iso", ""),
                "pass":       vr["pass"],
                "answers":    vr.get("answers", []),
            })

            is_last = (round_num == max_attempts - 1)
            if vr["pass"] or is_last:
                status = "success" if vr["pass"] else "failure"
                r = _make_result(status, gen_text, raw, conv["attempts"],
                                 conv["attempts_log"], conv["gold"], vr)
                r["exact_match"] = bool(compute_exact_match(gen_text, conv["gold"]))
                print(f"  {qid} | {status} | attempts={conv['attempts']} | em={r['exact_match']}")
                results[qid] = r
                conv["done"] = True
            else:
                conv["messages"].append({
                    "role":    "user",
                    "content": build_feedback(vr, feedback_mode),
                })

        # Checkpoint after each round
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        done_n = sum(1 for c in convs.values() if c["done"])
        print(f"[Checkpoint] round {round_num+1}: {done_n}/{len(convs)} done → {out_file}")


# ---------------------------------------------------------------------------
# 8) Main
# ---------------------------------------------------------------------------
async def main():
    ap = argparse.ArgumentParser(
        description="Unified agentic KBQA loop with configurable verifier and feedback.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Data ---
    ap.add_argument("--data_path",  type=str, required=True,
                    help="Path to dill dataset file.")
    ap.add_argument("--dataset",    type=str, required=True,
                    choices=["webqsp", "grailqa", "grailqapp"],
                    help="Dataset name. Controls entity format (webqsp vs grailqa/grailqapp).")
    ap.add_argument("--filter_ids", type=str, default=None,
                    help="JSON file of example IDs to restrict evaluation to.")

    # --- Prompt ---
    ap.add_argument("--sys_prompt", type=str, required=True,
                    help="Path to system prompt file.")
    ap.add_argument("--template",   type=str, required=True,
                    help="Path to Jinja2 user prompt template.")
    ap.add_argument("--shotmode",   type=str, required=True, choices=["zs", "fs"],
                    help="Shot mode label (for metadata only; actual content is in sys_prompt).")

    # --- Verifier & feedback ---
    ap.add_argument("--verifier",   type=str, default="exe",
                    choices=["none", "exe", "double"],
                    help="Verification strategy.")
    ap.add_argument("--feedback",   type=str, default="plain",
                    choices=["plain", "iso"],
                    help="Feedback style when verifier rejects a generation.")
    ap.add_argument("--pred_iso_path", type=str, default=None,
                    help="JSON file of per-example predicted ISOs "
                         "(list of {id, predicted_iso, correct, ...}). "
                         "When set, uses predicted_iso as the double-verifier target "
                         "and restricts to examples where correct==false.")

    # --- LLM ---
    ap.add_argument("--provider",   type=str, default="openai",
                    choices=["openai", "huggingface", "gemini", "vllm", "vllm_server", "claude", "openrouter"])
    ap.add_argument("--model",      type=str, default="gpt-3.5-turbo")
    ap.add_argument("--temperature", type=float, default=None,
                    help="Sampling temperature. Omit to use each model's default.")

    # --- Loop ---
    ap.add_argument("--max_attempts",  type=int, default=3)
    ap.add_argument("--seed",          type=int, default=42)

    # --- OpenAI / vllm_server async ---
    ap.add_argument("--batch_size",     type=int, default=200,
                    help="Examples per async batch (openai / vllm_server / openrouter).")
    ap.add_argument("--max_concurrent", type=int, default=100,
                    help="Max concurrent async conversations (openai / vllm_server / openrouter).")
    ap.add_argument("--api_base",       type=str, default="http://localhost:8000/v1",
                    help="Base URL for vllm_server provider.")
    ap.add_argument("--api_key",        type=str, default="EMPTY",
                    help="API key for vllm_server provider.")

    # --- Claude batch ---
    ap.add_argument("--anthropic_api_key", type=str,
                    default=os.environ.get("ANTHROPIC_API_KEY", ""),
                    help="Anthropic API key (default: $ANTHROPIC_API_KEY env var).")

    # --- OpenRouter ---
    ap.add_argument("--openrouter_api_key", type=str,
                    default=os.environ.get("OPENROUTER_API_KEY", ""),
                    help="OpenRouter API key (default: $OPENROUTER_API_KEY env var).")

    # --- vLLM local ---
    ap.add_argument("--tensor_parallel_size", type=int, default=1,
                    help="Number of GPUs for vLLM tensor parallelism.")
    ap.add_argument("--max_length",     type=int, default=None,
                    help="vLLM context window. Falls back to tokenizer config if not set.")
    ap.add_argument("--max_new_tokens", type=int, default=512,
                    help="Max tokens to generate per turn (vLLM local only).")

    # --- Output ---
    ap.add_argument("--out_dir",    type=str, required=True,
                    help="Directory to write results JSON and metrics.json.")
    ap.add_argument("--run_name",   type=str, default=None,
                    help="Output filename stem. "
                         "Default: {dataset}_{model}_{shotmode}_{verifier}_{feedback}")

    args = ap.parse_args()
    seed_everything(args.seed)

    # ---- Derived names --------------------------------------------------------
    model_safe = args.model.replace("/", "_")
    pred_iso_safe = "_pred_iso" if args.pred_iso_path else ""
    run_name   = args.run_name or (
        f"{args.dataset}_{model_safe}_{args.shotmode}_{args.verifier}_{args.feedback}{pred_iso_safe}"
    )
    os.makedirs(args.out_dir, exist_ok=True)
    out_file     = os.path.join(args.out_dir, f"{run_name}.json")
    metrics_file = os.path.join(args.out_dir, "metrics.json")

    # ---- Load assets ----------------------------------------------------------
    sys_prompt    = read_file(args.sys_prompt)
    user_template = read_file(args.template)
    data          = load_dataset(args.data_path, args.dataset)

    # ---- Load pred ISO (optional) ---------------------------------------------
    pred_iso_map = {}   # {qid: predicted_iso}  — empty means use gold ISO
    if args.pred_iso_path:
        with open(args.pred_iso_path, "r", encoding="utf-8") as f:
            pred_iso_data = json.load(f)
        if isinstance(pred_iso_data, list):
            for item in pred_iso_data:
                if not item.get("correct", True):   
                    iso = item.get("predicted_iso") or item.get("pred_rp") or item.get("predicted")
                    pred_iso_map[str(item["id"])] = iso
        elif isinstance(pred_iso_data, dict):
            for qid, item in pred_iso_data.items():
                if not item.get("correct", True):
                    iso = item.get("predicted_iso") or item.get("pred_rp") or item.get("predicted")
                    pred_iso_map[str(qid)] = iso
        print(f"[PredISO] {len(pred_iso_map)} incorrect examples loaded from {args.pred_iso_path}")

        # Auto-load the corresponding double-iso results to copy correct examples from.
        # Path is derived from current args: {out_dir}/{dataset}_{model_safe}_fs_double_iso.json
        double_iso_results_path = os.path.join(
            args.out_dir,
            f"{args.dataset}_{model_safe}_{args.shotmode}_double_iso.json"
        )
        if os.path.exists(double_iso_results_path):
            with open(double_iso_results_path, "r", encoding="utf-8") as f:
                double_iso_results = json.load(f)
            if isinstance(double_iso_results, list):
                double_iso_results = {str(item[0]): item[1] for item in double_iso_results}
            print(f"[PredISO] Loaded {len(double_iso_results)} double-iso results from {double_iso_results_path}")
        else:
            double_iso_results = {}
            print(f"[PredISO] No double-iso results found at {double_iso_results_path}, correct examples will be skipped.")
    else:
        double_iso_results = {}

    # ---- Resume ---------------------------------------------------------------
    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        if isinstance(results, list):
            results = {str(item[0]): item[1] for item in results}
        print(f"[Resume] Loaded {len(results)} existing results from {out_file}")
    else:
        results = {}

    # ---- Filter IDs -----------------------------------------------------------
    allowed_ids = None
    if args.filter_ids:
        with open(args.filter_ids, "r", encoding="utf-8") as f:
            fd = json.load(f)
        if isinstance(fd, list):
            allowed_ids = {str(e.get("id", e.get("ID", ""))) for e in fd}
        elif isinstance(fd, dict):
            allowed_ids = {str(k) for k in fd.keys()}
        print(f"[Filter] Restricting to {len(allowed_ids)} IDs from {args.filter_ids}")

    # ---- Build pending list ---------------------------------------------------
    entity_key = "entities" if args.dataset == "webqsp" else "head_entities"
    pending = []
    n_already = len(results)   # cached before any copying
    n_copied  = 0
    for idx, ex in enumerate(data, 1):
        qid = str(ex.get("ID", ex.get("id", f"ex_{idx}")))
        if qid in results:
            continue
        if allowed_ids is not None and qid not in allowed_ids:
            continue
        # When pred_iso_path is given, skip correct examples (copy from double_iso_results if available)
        if pred_iso_map and qid not in pred_iso_map:
            if qid in double_iso_results:
                results[qid] = double_iso_results[qid]
                n_copied += 1
            continue
        gold = ex["s_expression"]
        if not isinstance(gold, str) or gold.strip().lower() == "none":
            continue
        entities = ex.get(entity_key, [])
        if not entities:
            print(f"[Skip] {qid}: no entities.")
            continue
        try:
            user_prompt = render_user_prompt(
                user_template, ex["question"], entities, ex["kg_tuples"], args.dataset)
        except Exception as e:
            print(f"[Skip] {qid}: prompt build failed → {e}")
            continue
        target_iso = pred_iso_map.get(qid)   # None when not using pred_iso_path
        pending.append((qid, user_prompt, gold, target_iso))

    info_parts = [f"{len(pending)} to process", f"{n_already} already done"]
    if n_copied:
        info_parts.append(f"{n_copied} copied from double_iso (correct pred_iso)")
    print(
        f"[Info] {', '.join(info_parts)}. "
        f"verifier={args.verifier}  feedback={args.feedback}  provider={args.provider}"
    )

    # ==========================================================================
    # Claude Batch API path
    # ==========================================================================
    if args.provider.lower() == "claude":
        run_claude_batch(
            api_key        = args.anthropic_api_key,
            model          = args.model,
            max_tokens     = args.max_new_tokens,
            sys_prompt     = sys_prompt,
            pending        = pending,
            max_attempts   = args.max_attempts,
            verifier_mode  = args.verifier,
            feedback_mode  = args.feedback,
            results        = results,
            out_file       = out_file,
            temperature    = args.temperature,
        )

    # ==========================================================================
    # Async path  (openai / vllm_server / openrouter)
    # ==========================================================================
    elif args.provider.lower() in {"openai", "vllm_server", "openrouter"}:
        from openai import AsyncOpenAI

        if args.provider.lower() == "openai":
            client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        elif args.provider.lower() == "openrouter":
            client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=args.openrouter_api_key,
            )
        else:
            client = AsyncOpenAI(base_url=args.api_base, api_key=args.api_key)

        semaphore = asyncio.Semaphore(args.max_concurrent)

        for batch_start in range(0, len(pending), args.batch_size):
            batch = pending[batch_start: batch_start + args.batch_size]
            tasks = [
                asyncio.create_task(run_one_async(
                    client, semaphore, args.model, args.temperature,
                    sys_prompt, user_prompt, gold,
                    args.max_attempts, args.verifier, args.feedback,
                    target_iso=target_iso,
                ))
                for _, user_prompt, gold, target_iso in batch
            ]
            batch_results = await asyncio.gather(*tasks)

            for (qid, _, _, _), res in zip(batch, batch_results):
                res["exact_match"] = bool(compute_exact_match(res["output"], res["gold_sexp"]))
                print(f"  {qid} | {res['status']} | attempts={res['attempts']} | em={res['exact_match']}")
                results[qid] = res

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"[Checkpoint] {batch_start + len(batch)}/{len(pending)} → {out_file}")

    # ==========================================================================
    # Sync path  (huggingface / gemini / vllm)
    # ==========================================================================
    else:
        llm = create_llm(
            args.provider, args.model, temperature=args.temperature,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_length,
            max_new_tokens=args.max_new_tokens,
            api_base=args.api_base,
            api_key=args.api_key,
        )
        for i, (qid, user_prompt, gold, target_iso) in enumerate(pending, 1):
            res = run_one(llm, sys_prompt, user_prompt, gold,
                          args.max_attempts, args.verifier, args.feedback,
                          target_iso=target_iso)
            res["exact_match"] = bool(compute_exact_match(res["output"], res["gold_sexp"]))
            print(f"  {qid} | {res['status']} | attempts={res['attempts']} | em={res['exact_match']}")
            results[qid] = res

            if i % 50 == 0:
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                print(f"[Checkpoint] {i}/{len(pending)} → {out_file}")

    # ---- Final save -----------------------------------------------------------
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # ---- Metrics --------------------------------------------------------------
    total    = len(results)
    correct  = sum(1 for r in results.values() if r.get("exact_match", False))
    acc      = correct / total if total else 0.0
    baseline_correct = sum(
        1 for r in results.values()
        if r.get("attempts_log")
        and bool(compute_exact_match(r["attempts_log"][0]["output"], r["gold_sexp"]))
    )
    baseline_acc = baseline_correct / total if total else 0.0

    print(f"\n[Results] {run_name}")
    print(f"  Overall  EM: {correct}/{total} = {acc:.2%}")
    print(f"  Baseline EM: {baseline_correct}/{total} = {baseline_acc:.2%}")

    all_metrics = {}
    if os.path.exists(metrics_file):
        with open(metrics_file, "r", encoding="utf-8") as f:
            all_metrics = json.load(f)
    all_metrics[run_name] = {
        "dataset":                  args.dataset,
        "model":                    args.model,
        "provider":                 args.provider,
        "shotmode":                 args.shotmode,
        "verifier":                 args.verifier,
        "feedback":                 args.feedback,
        "total":                    total,
        "exact_match":              correct,
        "exact_match_acc":          round(acc, 4),
        "baseline_exact_match":     baseline_correct,
        "baseline_exact_match_acc": round(baseline_acc, 4),
    }
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[Metrics] saved → {metrics_file}")


if __name__ == "__main__":
    asyncio.run(main())
