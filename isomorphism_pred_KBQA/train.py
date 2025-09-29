# train.py

import os
import sys
import json
import argparse
import dill
import random
import numpy as np
import re
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
import wandb
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import f1_score

from utils.utils import (
    compute_global_label_map,
    seed_everything,
    pad_entity_labels,
    kco_info_nce_loss,
    execute_sparql
)
from utils.evaluation import hits_at_k_text, hits_at_k_graph, compute_hits_at_k, compute_exact_match, compute_f1
from utils.parse_sql import lisp_to_sparql
from utils.early_stopping import EarlyStopping
from dataloader import get_dataloader
from dataloader_with_prefix import get_dataloader as get_dataloader_prefix

from models.text_model import (
    BertMultiTaskModel,
    T5MultiTaskModel,
    GPT2MultiTaskModel,
)
from models.graph_model import GraphMultiTaskModel
from models.cod import CoDModel

# from accelerate import Accelerator
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

from torch.utils import checkpoint

from bitsandbytes.optim import Adam8bit

def train_epoch_text_prefix(model, dataloader, optimizer, tokenizer, device, args, epoch=0):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for step, batch in enumerate(tqdm(dataloader, desc="Training", leave=False), start=0):
        input_ids = batch["text_view"]["input_ids"].to(device)
        attention_mask = batch["text_view"]["attention_mask"].to(device)
        labels = batch["text_view"]["labels"].to(device)

        optimizer.zero_grad()
        outputs, _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs['gen_outputs'].loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        wandb.log({
            "train/loss": loss.item(),
            "step": step + epoch * len(dataloader)
        })

    avg_loss = total_loss / max(1, num_batches)
    wandb.log({"train/epoch_loss": avg_loss, "epoch": epoch})
    return avg_loss

from sklearn.metrics import f1_score

def eval_epoch_text_prefix(model, dataloader, tokenizer, device, args, epoch=0):
    model.eval()

    gen_correct, gen_total = 0, 0
    all_iso_preds = []
    all_iso_gold = []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, desc="Evaluating", leave=False), start=0):
            input_ids = batch["text_view"]["input_ids"].to(device)
            attention_mask = batch["text_view"]["attention_mask"].to(device)

            preds = model.generate(input_ids=input_ids, attention_mask=attention_mask)
            pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)

            task_types = batch["text_view"]["task_type"]
            gold_sexps_raw = batch["text_view"].get("gold_sexp_raw", [None] * len(task_types))
            gold_iso = batch["text_view"].get("gold_iso", [None] * len(task_types))

            for i, task in enumerate(task_types):
                pred = pred_texts[i].strip()
                if task == "generation":
                    gold = gold_sexps_raw[i].strip()
                    gen_correct += compute_exact_match(pred, gold)
                    gen_total += 1
                elif task == "iso":
                    all_iso_preds.append(pred)
                    all_iso_gold.append(gold_iso[i].strip())

    metrics = {
        "exact_match": gen_correct / max(1, gen_total),
        "iso_f1": f1_score(all_iso_gold, all_iso_preds, average="macro", zero_division=0)
    }

    wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=epoch)
    metrics["epoch"] = epoch  # optional: include in return for tracking

    return metrics

def train_epoch_text(
    args, model, loader, optimizer, scheduler,
    global_step, device, tasks, weights
):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        
        optimizer.zero_grad()
        tv = batch["text_view"]
        in_ids = tv["input_ids"].to(device)
        attn   = tv["attention_mask"].to(device)

        gen_l = tv.get("labels")
        iso_l = tv.get("iso_label")
        spans= tv.get("entity_spans")
        labs = tv.get("entity_labels")

        kwargs = {"input_ids": in_ids, "attention_mask": attn}
        if "generation" in tasks:
            kwargs["labels"] = gen_l.to(device)
        if "ranking" in tasks:
            kwargs["entity_spans"] = spans

        out, _ = model(**kwargs)
        loss = torch.tensor(0.0, device=device)

        gen_loss, iso_loss, ranking_loss = None, None, None

        if "generation" in tasks:
            gen_loss = out["gen_outputs"].loss
            loss += weights["generation"] * gen_loss
        elif "iso" in tasks:
            iso_loss = F.cross_entropy(out["iso_logits"], iso_l.to(device))
            loss += weights["iso"] * iso_loss
        elif "ranking" in tasks:
            scores, mask = pad_entity_labels(labs, device)
            ranking_loss = F.binary_cross_entropy(out["entity_scores"] * mask, scores)
            loss += weights["ranking"] * ranking_loss

        loss.backward()
        optimizer.step()

        if args.scheduler is not None and args.scheduler in ("linear", "onecycle"):
            scheduler.step()

        global_step += 1

        if scheduler is not None and hasattr(scheduler, "get_last_lr"):
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        wandb.log({
            "train/generation_loss": gen_loss.item()     if gen_loss     is not None else None,
            "train/iso_loss":        iso_loss.item()     if iso_loss     is not None else None,
            "train/ranking_loss":    ranking_loss.item() if ranking_loss is not None else None,
            "train/lr":              current_lr,
        }, step=global_step)

        total_loss += loss.item()
    

    return total_loss / len(loader), global_step


def eval_epoch_text(args, model, loader, device, tasks, tokenizer, eps=1e-5):
    model.eval()
    iso_preds, iso_labels = [], []
    rank_hits, rank_count = 0.0, 0
    gen_hits, gen_count   = 0.0, 0
    em_sum,  f1_sum    = 0, 0

    with torch.no_grad():
        for b_idx, batch in enumerate(tqdm(loader, desc="Evaluating", leave=False), start=0):
            tv = batch["text_view"]
            in_ids = tv["input_ids"].to(device)
            attn   = tv["attention_mask"].to(device)
            iso_l  = tv.get("iso_label")
            spans  = tv.get("entity_spans", None)
            gold_spans = tv.get("answers")
            gold_sexps = tv.get("gold_sexp_raw", None)
            gold_sexps_mapped = tv.get("gold_sexp", None)

            if "generation" in tasks:
                gen_inputs = {"input_ids": in_ids, "attention_mask": attn}

            forward_out = None
            if any(t in tasks for t in ("iso","ranking")):
                fw_kwargs = {"input_ids": in_ids, "attention_mask": attn}
                if "ranking" in tasks:
                    fw_kwargs["entity_spans"] = spans
                forward_out, _ = model(**fw_kwargs)

            if "iso" in tasks:
                iso_logits = forward_out["iso_logits"]
                preds = iso_logits.argmax(-1).cpu().tolist()
                iso_preds.extend(preds)
                iso_labels.extend(iso_l.tolist())

            if "ranking" in tasks:
                scores = forward_out["entity_scores"].cpu()
                labs, _ = pad_entity_labels(tv["entity_labels"], device)
                labs = labs.cpu()
                batch_hits = compute_hits_at_k(scores, labs)
                rank_hits += batch_hits
                rank_count += (labs.sum(dim=1) > 0).sum().item()

            if "generation" in tasks:
                beams = args.num_beams
                gens = model.generate(
                    **gen_inputs,
                    max_length=args.generation_max_length,
                    num_beams=beams
                )
                decoded = tokenizer.batch_decode(gens, skip_special_tokens=True)
                batch_size = in_ids.size(0)

                for i in range(batch_size):
                    sample_sexps = decoded[i*beams:(i+1)*beams]
                    # gold_entities = [a for a in gold_spans[i]]
                    # count_by_ent = Counter()

                    # import pdb; pdb.set_trace()

                    # remap each <E#> back to original id before SPARQL
                    if args.convert_entities:
                        node_map = tv["entity_ids"][i]
                        mapped_sexps = []
                        for sexp in sample_sexps:
                            # replace placeholders
                            if b_idx == 0 and i == 0:
                                print(f"Pred S-exp before mapping: {sexp}")
                            for idx, orig_ent in node_map.items():
                                sexp = sexp.replace(f"<E{idx}>", orig_ent)
                            mapped_sexps.append(sexp)
                        sample_sexps = mapped_sexps
                        # try:
                        #     sparql = lisp_to_sparql(sexp)
                        #     for ent in execute_sparql(sparql):
                        #         count_by_ent[ent] += 1
                        # except Exception:
                        #     continue

                    # compute Hits@K for generation
                    # ranked = [e for e,_ in count_by_ent.most_common()]
                    # k = len(gold_entities)
                    # if k:
                    #     hit = sum(1 for e in ranked[:k] if e in gold_entities) / (k + eps)
                    #     gen_hits += hit

                    # compute em
                    em = 0
                    for p_sexp in sample_sexps:
                        em += compute_exact_match(p_sexp, gold_sexps_mapped[i])
                    em_sum += em

                    # compute f1
                    # if k:
                    #     f1_sum += compute_f1(ranked[:k], gold_entities)
                    gen_count += 1

    # aggregate
    metrics = {}
    if "iso" in tasks:
        from sklearn.metrics import f1_score
        metrics["iso_f1"] = f1_score(iso_labels, iso_preds, average="macro")
    if "ranking" in tasks:
        metrics["hits@k_ranking"] = rank_hits / max(rank_count, 1)
    if "generation" in tasks:
        # metrics["hits@k_generation"] = gen_hits / max(gen_count, 1)
        metrics["exact_match"] = em_sum    / max(gen_count, 1)
        # metrics["f1_score"] = f1_sum / max(gen_count, 1)

    return metrics

def train_epoch_graph(model, loader, optimizer, weights, device):
    model.train()
    total_loss = 0.0
    rank_loss_fn = torch.nn.BCELoss()
    iso_loss_fn  = torch.nn.CrossEntropyLoss()

    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad(set_to_none=True)
        gv = batch["graph_view"]

        # with autocast():
        # def forward_fn(gv):
            # Use checkpointing to save memory
            # return model(gv)
        out, _ = model(gv)

        # out, _ = checkpoint.checkpoint(forward_fn, gv)

        loss = torch.tensor(0.0, device=device)

        # entity ranking loss (node‐level BCE)
        if "ranking" in model.modes:
            scores = out["entity_scores"]
            gdata  = gv["gnn_data"].to(device)
            labels = gdata.answer_x.squeeze(-1).float()
            l_rank = rank_loss_fn(scores, labels)
            loss  += weights.get("ranking", 1.0) * l_rank

        # isomorphism classification loss (graph‐level CE)
        if "iso" in model.modes:
            iso_lbl = batch["text_view"]["iso_label"].to(device)
            logits  = out["iso_logits"]
            l_iso   = iso_loss_fn(logits, iso_lbl)
            loss   += weights.get("iso", 1.0) * l_iso

        loss.backward()
        # try:
        optimizer.step()
        # except:
        #     import pdb; pdb.set_trace()

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        wandb.log({
            "train/ranking_loss": l_rank.item() if "ranking" in model.modes else None,
            "train/iso_loss":     l_iso.item()    if "iso"     in model.modes else None,
        })

        total_loss += loss.item()

        # clear cache 
        torch.cuda.empty_cache()

    return total_loss / len(loader)


def eval_epoch_graph(model, loader, device):
    model.eval()
    rank_hits, rank_count = 0.0, 0
    iso_preds, iso_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating (graph)", leave=False):
            gv = batch["graph_view"]

            # gdata = gv["gnn_data"]
            # et = gdata.edge_type
            # ei = gdata.edge_index

            # print(f"  ▶ edge_type dtype={et.dtype}   min={int(et.min())}   max={int(et.max())}")
            # print(f"  ▶ expected num_rels = {model.params.num_rels}")
            # print(f"  ▶ edge_index shape = {tuple(ei.shape)}   (should be [2, E])")


            for k, v in gv.items():
                # we know gnn_data is a torch_geometric Batch
                if hasattr(v, "to"):
                    gv[k] = v.to(device)

            try:
                out, _ = model(gv)
            except Exception as e:
                import pdb; pdb.set_trace()

            # ranking hits@k over each graph
            if "ranking" in model.modes:
                scores_flat = out["entity_scores"].cpu()
                gdata       = gv["gnn_data"]
                batch_idx   = gdata.batch.cpu()
                gold_mask   = gdata.answer_x.cpu().squeeze(-1)

                for gid in range(int(batch_idx.max()) + 1):
                    node_mask = (batch_idx == gid)
                    sc = scores_flat[node_mask]
                    gm = gold_mask[node_mask]

                    ranked   = torch.argsort(sc, descending=True).tolist()
                    gold_idx = torch.where(gm == 1)[0].tolist()
                    k = len(gold_idx)
                    if k > 0:
                        rank_hits  += len(set(ranked[:k]) & set(gold_idx)) / k
                        rank_count += 1

            # iso classification F1
            if "iso" in model.modes:
                logits = out["iso_logits"]
                preds  = logits.argmax(-1).cpu().tolist()
                gold   = batch["text_view"]["iso_label"].tolist()
                iso_preds.extend(preds)
                iso_labels.extend(gold)

    # aggregate
    metrics = {}
    if "ranking" in model.modes:
        metrics["hits@k_graph"] = rank_hits / max(rank_count, 1)
    if "iso" in model.modes:
        metrics["iso_f1"] = f1_score(iso_labels, iso_preds, average="macro")
    return metrics

def train_epoch_cod(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        iso_l = batch["text_view"]["iso_label"].to(device)
        logits, z_gcn, z_text = model(batch)
        cl = F.cross_entropy(logits, iso_l)
        nc = (kco_info_nce_loss(z_gcn, z_text) + kco_info_nce_loss(z_text, z_gcn)) / 2.0
        loss = cl + nc
        loss.backward()
        optimizer.step()

        wandb.log({
            "train/iso_loss": cl.item(),
            "train/contrastive_loss": nc.item(),
            "train/loss": loss.item(),
        })

        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch_cod(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            iso_l = batch["text_view"]["iso_label"].to(device)
            logits, _, _ = model(batch)
            preds = logits.argmax(-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(iso_l.cpu().tolist())
        
        
    
    return {"iso_f1": f1_score(all_labels, all_preds, average="macro")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["text","graph","cod","text_multi"], required=True)
    parser.add_argument("--backbone",    choices=["bert","t5","gpt2"], default="t5")
    parser.add_argument("--tasks",       type=str, default="iso")
    parser.add_argument("--weights",     type=str, default=None)
    parser.add_argument("--dataset",     required=True)
    parser.add_argument("--data_path",   required=True)
    parser.add_argument("--relations_path", default=None)
    parser.add_argument("--output_dir",  required=True)

    parser.add_argument("--input_max_length",      type=int, default=512)
    parser.add_argument("--generation_max_length", type=int, default=128)
    parser.add_argument("--max_seq_len",           type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=1,)

    parser.add_argument("--ptlm_model",  default="t5-base")
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--epochs",      type=int, default=10)
    parser.add_argument("--lr",          type=float, default=5e-5)
    parser.add_argument("--lr_text", type=float, default=5e-5)
    parser.add_argument("--lr_graph", type=float, default=8e-5)
    parser.add_argument(
        "--scheduler",
        choices=["none", "linear", "step", "cosine", "cosine_restart", "reduce_on_plateau", "onecycle"],
        default="linear"
    )
    parser.add_argument("--step_size",    type=int,   default=1,
                        help="StepLR step size (in epochs)")
    parser.add_argument("--warmup_steps", type=int,   default=0,
                        help="Warmup (in steps) for linear scheduler")
    parser.add_argument("--gamma",        type=float, default=0.9,
                        help="Decay factor for Step/Cosine LR")
    parser.add_argument("--t_max",        type=int,   default=10,
                        help="T_max for CosineAnnealingLR")
    parser.add_argument("--eta_min", type=float, default=1e-6, help="Minimum LR for cosine schedulers")
    parser.add_argument("--pct_start",    type=float, default=0.3,
                        help="Warmup fraction for OneCycleLR")

    parser.add_argument("--gnn_model",    choices=["rgcn","rgat","gcn","gat"], default="rgcn")
    parser.add_argument("--gnn_depth",    type=int, default=2)
    parser.add_argument("--node_emb_dim", type=int, default=768)
    parser.add_argument("--hidden_dim",  type=int, default=768)

    parser.add_argument("--shared_dim",   type=int, default=2048)
    parser.add_argument("--dropout",      type=float, default=0.2)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--wandb_proj",  type=str, default="isomorphism_pred_KBQA")
    parser.add_argument("--wandb_run",   type=str, default="run1")
    parser.add_argument("--patience",    type=int, default=5)

    parser.add_argument("--debug", action="store_true", help="Run in debug mode with smaller dataset")
    parser.add_argument("--convert_entities", action="store_true",)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # --- setup ---
    device = seed_everything(args.seed)
    # accelerator = Accelerator()
    # device = accelerator.device
    os.makedirs(args.output_dir, exist_ok=True)
    wandb.init(project=args.wandb_proj,
               name=args.wandb_run,
               config=vars(args))

    tasks = args.tasks.split(",") if args.tasks else []
    if args.weights:
        w = list(map(float, args.weights.split(",")))
        weights = dict(zip(tasks, w))
    else:
        weights = {t: 1.0 for t in tasks}

    # --- data ---
    with open(args.data_path, "rb") as f:
        data = dill.load(f)

    if args.dataset != "grailqa":
        train, dev, test = data["train"], data["dev"], data["test"]
    else:
        if args.debug:
            train = data["train"][:200]
            dev = data["train"][200:300]
        else:
            train = data["train"][:int(len(data["train"])*0.7)]
            dev = data["train"][int(len(data["train"])*0.7):]
        test = data["dev"]

    all_samples = train + dev + test
    max_entities = max(len(e["node_data"]["node_dict"]) for e in all_samples)

    # --- labels & relations ---
    label_map = None
    if args.model_type in ("text","graph","cod"):
        if "iso" in tasks or args.model_type=="cod":
            label_map, nc = compute_global_label_map(train, dev, test)
            args.num_classes = nc
    if args.model_type in ("graph","cod"):
        assert args.relations_path, "--relations_path required"
        with open(args.relations_path) as rf:
            rels = json.load(rf)
        args.num_rels = len(rels)

    # --- tokenizer & loaders ---
    tokenizer = AutoTokenizer.from_pretrained(args.ptlm_model)
    if args.backbone == "gpt2" and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    special_tokens = {
            'less': ' <',
            'less_or_equal': ' <='
        }

    print("DEBUGGING: convert_entities is", args.convert_entities)

    if args.convert_entities:
        entity_tokens = [f"<E{i}>" for i in range(max_entities)]
        tokenizer.add_tokens(entity_tokens + list(special_tokens.values()))
    else:
        tokenizer.add_tokens(list(special_tokens.values()))

    get_loader = get_dataloader_prefix if args.model_type == "text_multi" else get_dataloader

    train_loader = get_loader(
        train, args, tokenizer, modes=tasks, convert_entities=args.convert_entities,
        global_label_map=label_map,
        batch_size=args.batch_size, shuffle=True
    )
    dev_loader = get_loader(
        dev, args, tokenizer, modes=tasks, convert_entities=args.convert_entities,
        global_label_map=label_map,
        batch_size=args.batch_size, shuffle=False
    )
    test_loader = get_loader(
        test, args, tokenizer, modes=tasks, convert_entities=args.convert_entities,
        global_label_map=label_map,
        batch_size=args.batch_size, shuffle=False
    )

    # all_lengths = []
    # for batch in train_loader:
    #     # batch["text_view"]["attention_mask"] is (B, L)
    #     lengths = batch["text_view"]["attention_mask"].sum(dim=1).tolist()
    #     all_lengths.extend(lengths)

    # print(f"Dataset size:            {len(all_lengths)} sequences")
    # print(f"Max un-padded length:   {max(all_lengths)}")
    # print(f"Min un-padded length:   {min(all_lengths)}")
    # print(f"Average un-padded length: {sum(all_lengths)/len(all_lengths):.1f}")

    print(args.model_type, "tasks:", tasks)

    # --- model ---
    if args.model_type == "text":
        if args.backbone == "bert":
            model = BertMultiTaskModel(args, tokenizer, modes=tasks)
        elif args.backbone == "t5":
            model = T5MultiTaskModel(args, tokenizer, modes=tasks)
        else:
            model = GPT2MultiTaskModel(args, tokenizer, modes=tasks)
        model.backbone.resize_token_embeddings(len(tokenizer))
        # model.backbone.gradient_checkpointing_enable()

    elif args.model_type == "graph":
        model = GraphMultiTaskModel(args, modes=tasks)
        # model.text_encoder.gradient_checkpointing_enable()
    elif args.model_type == "text_multi":
        if args.backbone == "bert":
            model = BertMultiTaskModel(args, tokenizer, modes=["generation"])
        elif args.backbone == "t5":
            model = T5MultiTaskModel(args, tokenizer, modes=["generation"])
        else:
            model = GPT2MultiTaskModel(args, tokenizer, modes=["generation"])
        model.backbone.resize_token_embeddings(len(tokenizer))
        # model.backbone.gradient_checkpointing_enable()

    else:  # CoD
        if args.backbone == "bert":
            txt = BertMultiTaskModel(args, tokenizer, modes=["iso"])
        elif args.backbone == "t5":
            txt = T5MultiTaskModel(args, tokenizer, modes=["iso"])
        else:
            txt = GPT2MultiTaskModel(args, tokenizer, modes=["iso"])
        txt.backbone.resize_token_embeddings(len(tokenizer))
        graph = GraphMultiTaskModel(args, modes=["iso"])
        model = CoDModel(args, txt, graph)

    model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # optimizer = Adam8bit(trainable_params, lr=args.lr)
    # optimizer = AdamW(trainable_params, lr=args.lr)
    if args.model_type == "cod" and args.gnn_model == "gcn":
        optimizer = torch.optim.AdamW([
            {
                "params": model.text_model.parameters(),
                "lr": args.lr_text
            },
            {
                "params": model.text_head.parameters(),
                "lr": args.lr_text
            },
            {
                "params": model.graph_model.parameters(),
                "lr": args.lr_graph
            },
            {
                "params": model.gcn_head.parameters(),
                "lr": args.lr_graph
            },
            {
                "params": model.classifier.parameters(),
                "lr": args.lr_graph   # tie to graph branch for classification
            }
        ])
        wandb.config.update({
            "lr_text": args.lr_text,
            "lr_graph": args.lr_graph
        })
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # optimizer = Adam8bit(trainable_params, lr=args.lr)
    # if args.debug:
    #     optimizer = Adam8bit(trainable_params, lr=args.lr)
    # else:
    #     optimizer = AdamW(trainable_params, lr=args.lr)
    # scaler = GradScaler()
    # model, optimizer, train_loader, dev_loader, test_loader = accelerator.prepare(
    #     model, optimizer, train_loader, dev_loader, test_loader
    # )

    # --- scheduler setup ---
    total_steps = len(train_loader) * args.epochs
    if args.scheduler == "none":
        scheduler = None
    elif args.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps
        )
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.t_max,
            eta_min=args.eta_min
        )
    elif args.scheduler == "cosine_restart":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.t_max,
            T_mult=1,
            eta_min=args.eta_min,
        )
    elif args.scheduler == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max" if "ranking" in tasks else "min",
            factor=args.gamma,
            patience=args.step_size
        )
    elif args.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=total_steps,
            pct_start=args.pct_start
        )
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")

    global_step = 0
    stopper = EarlyStopping(patience=args.patience)
    best_val = -float("inf")
    best_ckpt = os.path.join(args.output_dir, "best_model.pt")

    # --- training loop ---
    for ep in range(1, args.epochs + 1):
        if args.model_type == "cod":
            train_loss = train_epoch_cod(model, train_loader, optimizer, device)
            metrics = eval_epoch_cod(model, dev_loader, device)
        elif args.model_type == "graph":
            train_loss = train_epoch_graph(model, train_loader, optimizer, weights, device)
            metrics = eval_epoch_graph(model, dev_loader, device)
        elif args.model_type == "text_multi":
            train_loss = train_epoch_text_prefix(
                model, train_loader, optimizer, tokenizer, device, args, epoch=ep
            )
            metrics = eval_epoch_text_prefix(
                model, dev_loader, tokenizer, device, args, epoch=ep
            )
        else:
            train_loss, global_step = train_epoch_text(
                args, model, train_loader,
                optimizer, scheduler, global_step,
                device, tasks, weights
            )
            metrics = eval_epoch_text(args, model, dev_loader, device, tasks, tokenizer)

        primary = ("iso_f1" if args.model_type=="graph"
                    else "iso_f1" if args.model_type=="cod"
                    else "hits@k_ranking" if "ranking" in tasks
                    else "exact_match" if "generation" in tasks or "text_multi" in args.model_type
                    else "iso_f1")

        # epoch‐wise scheduler stepping
        if scheduler is not None:
            if args.scheduler in ("step", "cosine", "cosine_restart"):
                scheduler.step()
            elif args.scheduler == "reduce_on_plateau":
                scheduler.step(metrics.get(primary, 0.0))

        # log & print
        log_dict = {"epoch": ep, "train_loss": train_loss}
        for k,v in metrics.items():
            log_dict[f"dev_{k}"] = v
        wandb.log(log_dict)

        metric_str = "  ".join(f"dev_{k}={v:.4f}" for k,v in metrics.items())
        print(f"Epoch {ep}: train_loss={train_loss:.4f}  {metric_str}")

        # import pdb; pdb.set_trace()

        # checkpoint on primary
        primary_val = metrics.get(primary, 0.0)
        if primary_val > best_val:
            best_val = primary_val
            torch.save(model, best_ckpt)
            print(f"↳ saved best {primary}={best_val:.4f}")
        
        stopper(primary_val)

        if stopper.should_stop():
            print("Early stopping")
            break

    # --- final evaluation + save preds ---
    best_state_dict = torch.load(best_ckpt, weights_only=False)
    model.load_state_dict(best_state_dict)
    
    model.to(device)
    if args.model_type == "cod":
        test_metrics = eval_epoch_cod(model, test_loader, device)
    elif args.model_type == "graph":
        test_metrics = eval_epoch_graph(model, test_loader, device)
    elif args.model_type == "text_multi":
        test_metrics = eval_epoch_text_prefix(model, test_loader, tokenizer, device, args)
    else:
        test_metrics = eval_epoch_text(args, model, test_loader, device, tasks, tokenizer)

    # log final metrics
    for k,v in test_metrics.items():
        wandb.log({f"test_{k}": v})

    wandb.finish()
    # exit()


    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting", leave=False):
            IDs = batch["text_view"]["ID"]
            recs = [{"ID": qid} for qid in IDs]

            if args.model_type == "graph":
                out, _ = model(batch["graph_view"])
                if "ranking" in tasks:
                    scores      = out["entity_scores"].cpu()                 # (N_total,)
                    gdata       = batch["graph_view"]["gnn_data"]
                    batch_idx   = gdata.batch.cpu()                        
                    answer_mask = gdata.answer_x.cpu().squeeze(-1).bool() 

                    batch_size = int(batch_idx.max().item()) + 1
                    for gid in range(batch_size):
                        node_mask = (batch_idx == gid)
                        sc        = scores[node_mask]
                        gm        = answer_mask[node_mask]
                        k         = int(gm.sum().item())
                        topk      = torch.argsort(sc, descending=True)[:k].tolist()
                        recs[gid]["rank_pred"] = topk
            elif args.model_type == "text_multi":
                questions  = batch["text_view"]["question"]
                task_types = batch["text_view"]["task_type"]
                in_ids     = batch["text_view"]["input_ids"].to(device)
                attn       = batch["text_view"]["attention_mask"].to(device)

                gens = model.generate(
                    input_ids=in_ids,
                    attention_mask=attn,
                    max_length=args.generation_max_length,
                    num_beams=args.num_beams
                )
                decoded = tokenizer.batch_decode(gens, skip_special_tokens=True)

                gold_sexps = batch["text_view"].get("gold_sexp_raw", [None] * len(task_types))
                gold_isos  = batch["text_view"].get("gold_iso", [None] * len(task_types))

                for i, r in enumerate(recs):
                    beam_preds = decoded[i * args.num_beams: (i + 1) * args.num_beams]
                    r["gen_pred"] = beam_preds

                    if task_types[i] == "generation":
                        r["gold"] = gold_sexps[i]
                    elif task_types[i] == "iso":
                        r["gold"] = gold_isos[i]

            else:  # text or cod
                tv     = batch["text_view"]
                in_ids = tv["input_ids"].to(device)
                attn   = tv["attention_mask"].to(device)
                spans  = tv.get("entity_spans", None)

                if args.model_type == "cod":
                    logits, _, _ = model(batch)
                    out = {"iso_logits": logits}
                else:
                    fw = {"input_ids": in_ids, "attention_mask": attn}
                    if "ranking" in tasks:
                        fw["entity_spans"] = spans
                    out, _ = model(**fw)

                if "iso" in tasks:
                    iso_logits = out["iso_logits"].cpu().tolist()
                    recs = [
                        { **r, "iso_pred": logit_vec }
                        for r, logit_vec in zip(recs, iso_logits)
                    ]

                if "ranking" in tasks and args.model_type != "cod":  # cod doesn’t do ranking here
                    scores, mask = pad_entity_labels(tv["entity_labels"], device)
                    scores = out["entity_scores"].cpu()
                    recs = [
                        {
                            **r,
                            "rank_pred": torch.argsort(scores[i][mask[i].bool()], descending=True).tolist()
                        }
                        for i, r in enumerate(recs)
                    ]

                if "generation" in tasks and args.model_type != "cod":  # cod doesn’t do generation here
                    beams = args.num_beams
                    gens  = model.generate(
                        input_ids=in_ids,
                        attention_mask=attn,
                        max_length=args.generation_max_length,
                        num_beams=beams
                    )
                    decoded = tokenizer.batch_decode(gens, skip_special_tokens=True)
                    for i, r in enumerate(recs):
                        r["gen_pred"] = decoded[i*beams : (i+1)*beams]

            predictions.extend(recs)

    # Write out metrics + predictions
    out = {"metrics": test_metrics, "predictions": predictions}
    with open(os.path.join(args.output_dir, "test_predictions.json"), "w") as wf:
        json.dump(out, wf, indent=2)

    print("Test metrics:", test_metrics)
    wandb.log({f"test_{k}": v for k, v in test_metrics.items()})
    wandb.finish()


if __name__ == "__main__":
    main()