# train.py

import os
import json
import argparse
import dill
import random
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
import wandb
from tqdm import tqdm

from utils.utils import (
    compute_global_label_map,
    seed_everything,
    pad_entity_labels,
    kco_info_nce_loss,
    execute_sparql
)
from utils.evaluation import hits_at_k_text, hits_at_k_graph, compute_hits_at_k
from utils.webqsp_sql import lisp_to_sparql
from utils.early_stopping import EarlyStopping
from dataloader import get_dataloader

from models.text_model import (
    BertMultiTaskModel,
    T5MultiTaskModel,
    GPT2MultiTaskModel,
)
from models.graph_model import GraphMultiTaskModel
from models.cod import CoDModel


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

        out = model(**kwargs)
        loss = torch.tensor(0.0, device=device)

        gen_loss, iso_loss, ranking_loss = None, None, None

        if "generation" in tasks:
            gen_loss = out["gen_outputs"].loss
            loss += weights["generation"] * gen_loss
        if "iso" in tasks:
            iso_loss = F.cross_entropy(out["iso_logits"], iso_l.to(device))
            loss += weights["iso"] * iso_loss
        if "ranking" in tasks:
            scores, mask = pad_entity_labels(labs, device)
            ranking_loss = F.binary_cross_entropy(out["entity_scores"] * mask, scores)
            loss += weights["ranking"] * ranking_loss

        loss.backward()
        optimizer.step()

        # step scheduler per batch only for linear & onecycle
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


def eval_epoch_text(model, loader, device, tasks, tokenizer, eps=1e-5):
    model.eval()
    iso_preds, iso_labels = [], []
    rank_hits, rank_count = 0.0, 0
    gen_hits, gen_count = 0.0, 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            tv = batch["text_view"]
            in_ids = tv["input_ids"].to(device)
            attn   = tv["attention_mask"].to(device)
            iso_l  = tv.get("iso_label")
            spans  = tv.get("entity_spans")
            gold_spans = tv.get("answers")
            gold_sexps = tv.get("gold_sexp")

            kwargs = {"input_ids": in_ids, "attention_mask": attn}
            if "generation" in tasks:
                gen_inputs = kwargs.copy()
            if "ranking" in tasks:
                kwargs["entity_spans"] = spans

            out = model(**kwargs)

            # ISO
            if "iso" in tasks:
                preds = out["iso_logits"].argmax(-1).cpu().tolist()
                iso_preds.extend(preds)
                iso_labels.extend(iso_l.tolist())

            # RANKING
            if "ranking" in tasks:
                scores = out["entity_scores"].cpu()
                labs, mask = pad_entity_labels(tv["entity_labels"], device)
                labs = labs.cpu()
                batch_hits = compute_hits_at_k(scores, labs)
                rank_hits += batch_hits
                rank_count += (labs.sum(dim=1) > 0).sum().item()

            # GENERATION → SPARQL → HITS@K
            if "generation" in tasks:
                beams = getattr(model.config, "num_beams", 1)
                gens = model.generate(
                    **gen_inputs,
                    max_length=model.config.max_length,
                    num_beams=beams
                )
                decoded = tokenizer.batch_decode(gens, skip_special_tokens=True)
                bs = in_ids.size(0)
                for i in range(bs):
                    preds_sexps = decoded[i*beams:(i+1)*beams]
                    gold_ents = [a[0] for a in gold_spans[i]]
                    counter = Counter()
                    for sexp in preds_sexps:
                        try:
                            sparql = lisp_to_sparql(sexp)
                            for ent in execute_sparql(sparql):
                                counter[ent] += 1
                        except:
                            continue
                    ranked = [e for e,_ in counter.most_common()]
                    k = len(gold_ents)
                    if k:
                        gen_hits += sum(e in gold_ents for e in ranked[:k]) / (k + eps)
                    gen_count += 1

    metrics = {}
    if "iso" in tasks:
        from sklearn.metrics import f1_score
        metrics["iso_f1"] = f1_score(iso_labels, iso_preds, average="macro")
    if "ranking" in tasks:
        metrics["hits@k_ranking"] = rank_hits / max(rank_count, 1)
    if "generation" in tasks:
        metrics["hits@k_generation"] = gen_hits / max(gen_count, 1)
    return metrics


def train_epoch_graph(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    loss_fn = nn.BCELoss()

    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        gv = batch["graph_view"]
        out = model(gv)
        scores = out["entity_scores"]
        gdata = gv["gnn_data"].to(device)
        labels= gdata.answer_x.squeeze(-1).float()
        loss = loss_fn(scores, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def eval_epoch_graph(model, loader, device):
    model.eval()
    total_hits, count = 0.0, 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            gv = batch["graph_view"]
            out = model(gv)
            scores = out["entity_scores"].cpu()
            gdata = gv["gnn_data"]
            hits, _, _ = hits_at_k_graph(scores, gdata.answer_x.cpu())
            total_hits += hits
            count += 1

    return {"hits@k": total_hits / max(count, 1)}


def train_epoch_cod(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        iso_l = batch["text_view"]["iso_label"].to(device)
        logits, z_gcn, z_text = model(batch)
        cl = F.cross_entropy(logits, iso_l)
        nc = kco_info_nce_loss(z_gcn, z_text)
        loss = cl + nc
        loss.backward()
        optimizer.step()
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
    from sklearn.metrics import f1_score
    return {"iso_f1": f1_score(all_labels, all_preds, average="macro")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["text","graph","cod"], required=True)
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

    parser.add_argument("--ptlm_model",  default="t5-base")
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--epochs",      type=int, default=10)
    parser.add_argument("--lr",          type=float, default=5e-5)
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

    parser.add_argument("--gnn_model",    choices=["rgcn","rgat","gcn"], default="rgcn")
    parser.add_argument("--gnn_depth",    type=int, default=2)
    parser.add_argument("--node_emb_dim", type=int, default=768)

    parser.add_argument("--shared_dim",   type=int, default=2048)
    parser.add_argument("--dropout",      type=float, default=0.2)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--wandb_proj",  type=str, default="isomorphism_pred_KBQA")
    parser.add_argument("--wandb_run",   type=str, default="run1")
    parser.add_argument("--patience",    type=int, default=5)
    args = parser.parse_args()

    # --- setup ---
    device = seed_everything(args.seed)
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
    train, dev, test = data["train"], data["dev"], data["test"]

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

    train_loader = get_dataloader(
        train, args, tokenizer, modes=tasks, convert_entities=True,
        global_label_map=label_map,
        batch_size=args.batch_size, shuffle=True
    )
    dev_loader = get_dataloader(
        dev, args, tokenizer, modes=tasks, convert_entities=True,
        global_label_map=label_map,
        batch_size=args.batch_size, shuffle=False
    )
    test_loader = get_dataloader(
        test, args, tokenizer, modes=tasks, convert_entities=True,
        global_label_map=label_map,
        batch_size=args.batch_size, shuffle=False
    )

    # --- model ---
    if args.model_type == "text":
        if args.backbone == "bert":
            model = BertMultiTaskModel(args, tokenizer, modes=tasks)
        elif args.backbone == "t5":
            model = T5MultiTaskModel(args, tokenizer, modes=tasks)
        else:
            model = GPT2MultiTaskModel(args, tokenizer, modes=tasks)

    elif args.model_type == "graph":
        model = GraphMultiTaskModel(args, modes=tasks)

    else:  # CoD
        if args.backbone == "bert":
            txt = BertMultiTaskModel(args, tokenizer, modes=["iso"])
        elif args.backbone == "t5":
            txt = T5MultiTaskModel(args, tokenizer, modes=["iso"])
        else:
            txt = GPT2MultiTaskModel(args, tokenizer, modes=["iso"])
        graph = GraphMultiTaskModel(args, modes=["ranking"])
        model = CoDModel(args, txt, graph)

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

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
            train_loss = train_epoch_graph(model, train_loader, optimizer, device)
            metrics = eval_epoch_graph(model, dev_loader, device)
        else:
            train_loss, global_step = train_epoch_text(
                args, model, train_loader,
                optimizer, scheduler, global_step,
                device, tasks, weights
            )
            metrics = eval_epoch_text(model, dev_loader, device, tasks, tokenizer)

        primary = ("hits@k" if args.model_type=="graph"
                    else "iso_f1" if args.model_type=="cod"
                    else "hits@k_ranking" if "ranking" in tasks
                    else "hits@k_generation" if "generation" in tasks
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

        # checkpoint on primary
        primary_val = metrics.get(primary, 0.0)
        if primary_val > best_val:
            best_val = primary_val
            torch.save(model.state_dict(), best_ckpt)
            print(f"↳ saved best {primary}={best_val:.4f}")

        stopper(primary_val)
        if stopper.should_stop():
            print("Early stopping")
            break

    # --- final evaluation + save preds ---
    model.load_state_dict(torch.load(best_ckpt))
    model.to(device)
    if args.model_type == "cod":
        test_metrics = eval_epoch_cod(model, test_loader, device)
    elif args.model_type == "graph":
        test_metrics = eval_epoch_graph(model, test_loader, device)
    else:
        test_metrics = eval_epoch_text(model, test_loader, device, tasks, tokenizer)

    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting", leave=False):
            tv = batch["text_view"]
            IDs = tv["ID"]
            in_ids = tv["input_ids"].to(device)
            attn   = tv["attention_mask"].to(device)
            spans  = tv.get("entity_spans")

            kwargs = {"input_ids": in_ids, "attention_mask": attn}
            if "generation" in tasks:
                gen_inputs = kwargs.copy()
            if "ranking" in tasks:
                kwargs["entity_spans"] = spans

            out = model(**kwargs)

            # ISO
            iso_p = (out["iso_logits"].argmax(-1).cpu().tolist()
                     if "iso" in tasks else None)
            # RANKING
            rank_p = None
            if "ranking" in tasks:
                scores = out["entity_scores"].cpu()
                labs, _ = pad_entity_labels(tv["entity_labels"], device)
                rank_p = [
                    torch.argsort(scores[i], descending=True)[:int(labs[i].sum())].tolist()
                    for i in range(scores.size(0))
                ]
            # GENERATION
            gen_p = None
            if "generation" in tasks:
                beams = getattr(model.config, "num_beams", 1)
                gens = model.generate(
                    **gen_inputs,
                    max_length=model.config.max_length,
                    num_beams=beams
                )
                dec = tokenizer.batch_decode(gens, skip_special_tokens=True)
                gen_p = [dec[i*beams:(i+1)*beams] for i in range(len(IDs))]

            for i,qid in enumerate(IDs):
                rec = {"ID": qid}
                if iso_p is not None:    rec["iso_pred"] = iso_p[i]
                if rank_p is not None:   rec["rank_pred"] = rank_p[i]
                if gen_p is not None:    rec["gen_pred"] = gen_p[i]
                predictions.append(rec)

    out = {"metrics": test_metrics, "predictions": predictions}
    with open(os.path.join(args.output_dir, "test_predictions.json"), "w") as wf:
        json.dump(out, wf, indent=2)

    print("Test metrics:", test_metrics)
    wandb.log({f"test_{k}": v for k,v in test_metrics.items()})
    wandb.finish()


if __name__ == "__main__":
    main()