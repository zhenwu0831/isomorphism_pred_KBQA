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
from transformers import AutoTokenizer, AdamW
import wandb
from tqdm import tqdm

from utils.utils import (
    compute_global_label_map,
    seed_everything,
    pad_entity_labels,
    kco_info_nce_loss,
)
from utils.evaluation import hits_at_k_text, hits_at_k_graph, compute_hits_at_k
from utils.webqsp_sql import lisp_to_sparql, execute_sparql
from utils.early_stopping import EarlyStopping
from dataloader import get_dataloader

from models.text_model import (
    BertMultiTaskModel,
    T5MultiTaskModel,
    GPT2MultiTaskModel,
)
from models.graph_model import GraphMultiTaskModel
from models.cod import CoDModel


def train_epoch_text(model, loader, optimizer, device, tasks, weights):
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

        if "generation" in tasks:
            gen_loss = out["gen_outputs"].loss
            loss += weights["generation"] * gen_loss
            wandb.log({
                "batch_generation_loss": gen_loss.item()
            })
        if "iso" in tasks:
            iso_loss = F.cross_entropy(out["iso_logits"], iso_l.to(device))
            loss += weights["iso"] * iso_loss
            wandb.log({
                "batch_iso_loss": iso_loss.item()
            })
        if "ranking" in tasks:
            scores, mask = pad_entity_labels(labs, device)
            ranking_loss = F.binary_cross_entropy(out["entity_scores"] * mask, scores)
            loss += weights["ranking"] * ranking_loss
            wandb.log({
                "batch_ranking_loss": ranking_loss.item()
            })

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def eval_epoch_text(model, loader, device, tasks, tokenizer, eps=1e-5):
    """
    Evaluate generation→SPARQL→Hits@K, iso F1, and ranking Hits@K.
    """
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

            # common forward args
            kwargs = {"input_ids": in_ids, "attention_mask": attn}
            if "generation" in tasks:
                # will use generate()
                gen_inputs = kwargs.copy()
            if "ranking" in tasks:
                kwargs["entity_spans"] = spans

            out = model(**kwargs)

            # --- ISO ---
            if "iso" in tasks:
                preds = out["iso_logits"].argmax(-1).cpu().tolist()
                iso_preds.extend(preds)
                iso_labels.extend(iso_l.tolist())

            # --- RANKING Hits@K ---
            if "ranking" in tasks:
                scores = out["entity_scores"].cpu()
                labs, mask = pad_entity_labels(tv["entity_labels"], device)
                labs = labs.cpu()
                batch_hits = compute_hits_at_k(scores, labs)
                rank_hits += batch_hits
                rank_count += (labs.sum(dim=1) > 0).sum().item()

            # --- GENERATION → SPARQL → Hits@K ---
            if "generation" in tasks:
                beams = getattr(model.config, "num_beams", 1)
                gens = model.generate(**gen_inputs,
                                      max_length=model.config.max_length,
                                      num_beams=beams)
                decoded = tokenizer.batch_decode(gens, skip_special_tokens=True)
                batch_size = in_ids.size(0)
                for i in range(batch_size):
                    preds_sexps = decoded[i*beams:(i+1)*beams]
                    gold_entities = [a[0] for a in gold_spans[i]]
                    count_by_ent = Counter()
                    for sexp in preds_sexps:
                        try:
                            sparql = lisp_to_sparql(sexp)
                            for ent in execute_sparql(sparql):
                                count_by_ent[ent] += 1
                        except Exception:
                            continue
                    ranked_ents = [e for e,_ in count_by_ent.most_common()]
                    k = len(gold_entities)
                    if k:
                        hit = sum(1 for e in ranked_ents[:k] if e in gold_entities) / (k + eps)
                        gen_hits += hit
                    gen_count += 1

    # aggregate metrics
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
            scores= out["entity_scores"].cpu()
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
    parser.add_argument("--dataset",   required=True)
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

    device = seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    wandb.init(project=args.wandb_proj, name=args.wandb_run, config=vars(args))

    tasks = args.tasks.split(",") if args.tasks else []
    if args.weights:
        w = list(map(float, args.weights.split(",")))
        weights = dict(zip(tasks,w))
    else:
        weights = {t:1.0 for t in tasks}

    with open(args.data_path,"rb") as f:
        data = dill.load(f)
    train, dev, test = data["train"], data["dev"], data["test"]

    label_map = None
    if args.model_type in ("text","graph","cod"):
        if "iso" in tasks or args.model_type=="cod":
            label_map, nc = compute_global_label_map(train,dev,test)
            args.num_classes = nc

    if args.model_type in ("graph","cod"):
        assert args.relations_path, "--relations_path required"
        with open(args.relations_path) as rf:
            rels = json.load(rf)
        args.num_rels = len(rels)

    tokenizer = AutoTokenizer.from_pretrained(args.ptlm_model)
    if args.backbone=="gpt2" and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader = get_dataloader(train,args,tokenizer,
                                  modes=tasks,convert_entities=True,
                                  global_label_map=label_map,
                                  batch_size=args.batch_size,shuffle=True)
    dev_loader   = get_dataloader(dev,  args,tokenizer,
                                  modes=tasks,convert_entities=True,
                                  global_label_map=label_map,
                                  batch_size=args.batch_size,shuffle=False)
    test_loader  = get_dataloader(test, args,tokenizer,
                                  modes=tasks,convert_entities=True,
                                  global_label_map=label_map,
                                  batch_size=args.batch_size,shuffle=False)

    if args.model_type=="text":
        if args.backbone=="bert":
            model = BertMultiTaskModel(args,tokenizer,modes=tasks)
        elif args.backbone=="t5":
            model = T5MultiTaskModel(args,tokenizer,modes=tasks)
        else:
            model = GPT2MultiTaskModel(args,tokenizer,modes=tasks)

    elif args.model_type=="graph":
        model = GraphMultiTaskModel(args,modes=tasks)

    else:
        if args.backbone=="bert":
            txt = BertMultiTaskModel(args,tokenizer,modes=["iso"])
        elif args.backbone=="t5":
            txt = T5MultiTaskModel(args,tokenizer,modes=["iso"])
        else:
            txt = GPT2MultiTaskModel(args,tokenizer,modes=["iso"])
        graph = GraphMultiTaskModel(args,modes=["ranking"])
        model = CoDModel(args, txt, graph)

    model.to(device)
    optimizer = AdamW(model.parameters(),lr=args.lr)
    stopper   = EarlyStopping(patience=args.patience)
    best_val, best_ckpt = -1.0, os.path.join(args.output_dir,"best_model.pt")

    for ep in range(1, args.epochs+1):
        if args.model_type=="cod":
            train_loss = train_epoch_cod(model, train_loader, optimizer, device)
            metrics    = eval_epoch_cod (model, dev_loader,   device)
        elif args.model_type=="graph":
            train_loss = train_epoch_graph(model, train_loader, optimizer, device)
            metrics    = eval_epoch_graph (model, dev_loader,   device)
        else:
            train_loss = train_epoch_text(model, train_loader, optimizer, device, tasks, weights)
            metrics    = eval_epoch_text (model, dev_loader,   device, tasks)

        # log all metrics
        log_dict = {"epoch": ep, "train_loss": train_loss}
        for name, score in metrics.items():
            log_dict[f"dev_{name}"] = score
        wandb.log(log_dict)

        # print them
        metric_str = "  ".join(f"dev_{n}={s:.4f}" for n, s in metrics.items())
        print(f"Epoch {ep}: train_loss={train_loss:.4f}  {metric_str}")

        # pick primary metric: prefer hits@k if ranking is one of the tasks
        if "ranking" in tasks:
            primary = "hits@k"
        else:
            primary = "iso_f1"
        primary_val = metrics.get(primary, 0.0)

        # checkpoint if improved
        if primary_val > best_val:
            best_val = primary_val
            torch.save(model.state_dict(), best_ckpt)
            print(f"↳ saved best {primary}={primary_val:.4f}")

        # early stopping on primary metric
        stopper(primary_val)
        if stopper.should_stop():
            print("Early stopping")
            break

    # load best model and run on test set
    model.load_state_dict(torch.load(best_ckpt))
    model.to(device)

    if args.model_type == "cod":
        test_metrics = eval_epoch_cod(model, test_loader, device)
    elif args.model_type == "graph":
        test_metrics = eval_epoch_graph(model, test_loader, device)
    else:
        test_metrics = eval_epoch_text(model, test_loader, device, tasks)

    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting", leave=False):
            tv = batch["text_view"]
            IDs = tv["ID"]
            in_ids = tv["input_ids"].to(device)
            attn   = tv["attention_mask"].to(device)
            spans  = tv.get("entity_spans")

            # common kwargs
            kwargs = {"input_ids": in_ids, "attention_mask": attn}
            if "generation" in tasks:
                gen_inputs = kwargs.copy()
            if "ranking" in tasks:
                kwargs["entity_spans"] = spans

            out = model(**kwargs)

            # ISO preds
            iso_p = None
            if "iso" in tasks:
                iso_p = out["iso_logits"].argmax(-1).cpu().tolist()

            # ranking preds
            rank_p = None
            if "ranking" in tasks:
                scores = out["entity_scores"].cpu()
                labs, mask = pad_entity_labels(tv["entity_labels"], device)
                # per‐sample top-k
                rank_p = []
                for i in range(scores.size(0)):
                    gold_k = int((labs[i]>0).sum().item())
                    topk = torch.argsort(scores[i], descending=True)[:gold_k].tolist()
                    rank_p.append(topk)

            # generation preds
            gen_p = None
            if "generation" in tasks:
                beams = getattr(model.config, "num_beams", 1)
                gens = model.generate(**gen_inputs,
                                      max_length=model.config.max_length,
                                      num_beams=beams)
                decoded = tokenizer.batch_decode(gens, skip_special_tokens=True)
                gen_p = [decoded[i*beams:(i+1)*beams] for i in range(len(IDs))]

            # record
            for idx, qid in enumerate(IDs):
                rec = {"ID": qid}
                if iso_p is not None:
                    rec["iso_pred"] = iso_p[idx]
                if rank_p is not None:
                    rec["rank_pred"] = rank_p[idx]
                if gen_p is not None:
                    rec["gen_pred"] = gen_p[idx]
                predictions.append(rec)

    # dump both metrics + predictions
    out = {"metrics": test_metrics, "predictions": predictions}
    path = os.path.join(args.output_dir, "test_predictions.json")
    with open(path, "w") as wf:
        json.dump(out, wf, indent=2)

    print("Test metrics:", test_metrics)
    wandb.log({f"test_{k}": v for k,v in test_metrics.items()})
    wandb.finish()


if __name__=="__main__":
    main()