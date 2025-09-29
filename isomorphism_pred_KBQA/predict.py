# train.py

import os
import sys
import json
import argparse
import dill
import random
import numpy as np
import pandas as pd
import re
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.optim as optim
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    T5TokenizerFast,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
import wandb
from tqdm import tqdm
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
from dataloader import get_dataloader, get_T5_dataloader

from models.text_model import (
    BertMultiTaskModel,
    T5MultiTaskModel,
    GPT2MultiTaskModel,
    T5MultiTaskGenModel
)
from models.graph_model import GraphMultiTaskModel
from models.cod import CoDModel

# from accelerate import Accelerator
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

from torch.utils import checkpoint

from bitsandbytes.optim import Adam8bit


'''
Train, Eval, and Predict for Text-based models
'''

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


def predict_epoch_text(args, model, loader, device, tasks, tokenizer, eps=1e-5):
    model.eval()
    iso_preds, iso_labels = [], []
    pred_sexps, true_sexps = [], []
    rank_hits, rank_count = 0.0, 0
    gen_hits, gen_count   = 0.0, 0
    em_sum,  f1_sum    = 0, 0
    iso_logits_arr = []

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
                # import pdb; pdb.set_trace()
                iso_logits_arr.extend(iso_logits.cpu().tolist())

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

                        pred_sexps.append(p_sexp)
                        true_sexps.append(gold_sexps_mapped[i])

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
    
    

    
    return {'y_true': iso_labels, 'y_pred': iso_preds, 'pred_sexps': pred_sexps, 'true_sexps': true_sexps, 'iso_logits': iso_logits_arr}


'''
Train, Eval, and Predict for Graph-based models
'''

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

        total_loss += loss.item()

    return total_loss / len(loader)


def eval_epoch_graph(model, loader, device):
    model.eval()
    rank_hits, rank_count = 0.0, 0
    iso_preds, iso_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating (graph)", leave=False):
            gv = batch["graph_view"]

            for k, v in gv.items():
                # we know gnn_data is a torch_geometric Batch
                if hasattr(v, "to"):
                    gv[k] = v.to(device)

            out, _ = model(gv)

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


def predict_epoch_graph(model, loader, device):
    model.eval()
    rank_hits, rank_count = 0.0, 0
    iso_preds, iso_labels = [], []
    iso_logits = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating (graph)", leave=False):
            gv = batch["graph_view"]

            for k, v in gv.items():
                # we know gnn_data is a torch_geometric Batch
                if hasattr(v, "to"):
                    gv[k] = v.to(device)

            out, _ = model(gv)

            # iso classification F1
            if "iso" in model.modes:
                logits = out["iso_logits"]

                preds  = logits.argmax(-1).cpu().tolist()
                gold   = batch["text_view"]["iso_label"].tolist()
                iso_logits.extend(logits.cpu().tolist())
                iso_preds.extend(preds)
                iso_labels.extend(gold)

    # aggregate
    metrics = {}
    if "ranking" in model.modes:
        metrics["hits@k_graph"] = rank_hits / max(rank_count, 1)
    if "iso" in model.modes:
        metrics["iso_f1"] = f1_score(iso_labels, iso_preds, average="macro")
    
    if "iso" in model.modes:
        return {'y_true': iso_labels, 'y_pred': iso_preds, 'iso_logits': iso_logits}
    return None


'''
Train, Eval, and Predict for CoD-based models
'''


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

def predict_epoch_cod(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    all_logits            = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            iso_l = batch["text_view"]["iso_label"].to(device)
            logits, _, _ = model(batch)
            preds = logits.argmax(-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(iso_l.cpu().tolist())
            all_logits.extend(logits.cpu().tolist())
        
    return {'y_true': all_labels, 'y_pred': all_preds, 'iso_logits': all_logits}    
    
    # return {"iso_f1": f1_score(all_labels, all_preds, average="macro")}


'''
Eval, and Predict for Multi-Task Learning (MTL) text-based models
'''

def eval_epoch_MTL_text(args, model, loader, device, tasks, tokenizer, eps=1e-5):
    model.eval()
    iso_preds, iso_labels = [], []
    rank_hits, rank_count = 0.0, 0
    gen_hits, gen_count   = 0.0, 0
    em_sum,  f1_sum    = 0, 0

    prefix_dict = {
            'iso': 'classify',
            'lf' : 'generate'
    }
    
    with torch.no_grad():
        for b_idx, tv in enumerate(tqdm(loader, desc="Evaluating", leave=False), start=0):
            # tv = batch["text_view"]
            in_ids = tv["input_ids"].to(device)
            attn   = tv["attention_mask"].to(device)
            iso_l  = tv.get("isomorphism")
            prefix = tv.get("prefix", None)

            spans  = tv.get("entity_spans", None)
            gold_spans = tv.get("answers")
            gold_sexps = tv.get("gold_sexp_raw", None)
            gold_sexps_mapped = tv.get("gold_sexp", None)

            if "generation" in tasks:
                gen_inputs = {"input_ids": in_ids, "attention_mask": attn}

            forward_out = None
            
            
            beams = args.num_beams
            gens = model.generate(
                **gen_inputs,
                max_length=args.generation_max_length,
                num_beams=beams
            )
            decoded = tokenizer.batch_decode(gens, skip_special_tokens=True)
            batch_size = in_ids.size(0)

            for i in range(batch_size):
                sample_gens = decoded[i*beams:(i+1)*beams]
                

                if prefix[i] == 'iso':
                    iso_labels.append(iso_l[i])
                    
                    poss_lbls = []
                    for gen_lbl in sample_gens:
                        poss_lbls.append(gen_lbl.strip())
                    
                    # choose the lbl with the greatest frequency
                    
                    most_common_lbl, _ = Counter(poss_lbls).most_common(1)[0]
                    iso_preds.append(most_common_lbl)
                    
                else:

                    em = 0
                    for p_sexp in sample_gens:
                        em += compute_exact_match(p_sexp, gold_sexps_mapped[i])
                    em_sum += em

                    # compute f1
                    # if k:
                    #     f1_sum += compute_f1(ranked[:k], gold_entities)
                    gen_count += 1

    # aggregate
    metrics = {}
    # import pdb; pdb.set_trace()
    metrics["iso_f1"] = f1_score(iso_labels, iso_preds, average="macro")


    metrics["exact_match"] = em_sum    / max(gen_count, 1)
    

    return metrics



def predict_epoch_MTL_text(args, model, loader, device, tasks, tokenizer, eps=1e-5):
    model.eval()
    iso_preds, iso_labels = [], []
    pred_sexps, true_sexps = [], []

    rank_hits, rank_count = 0.0, 0
    gen_hits, gen_count   = 0.0, 0
    em_sum,  f1_sum    = 0, 0

    prefix_dict = {
            'iso': 'classify',
            'lf' : 'generate'
    }
    
    with torch.no_grad():
        for b_idx, tv in enumerate(tqdm(loader, desc="Evaluating", leave=False), start=0):
            # tv = batch["text_view"]
            in_ids = tv["input_ids"].to(device)
            attn   = tv["attention_mask"].to(device)
            iso_l  = tv.get("isomorphism")
            prefix = tv.get("prefix", None)

            spans  = tv.get("entity_spans", None)
            gold_spans = tv.get("answers")
            gold_sexps = tv.get("gold_sexp_raw", None)
            gold_sexps_mapped = tv.get("gold_sexp", None)

            if "generation" in tasks:
                gen_inputs = {"input_ids": in_ids, "attention_mask": attn}

            forward_out = None
            
            
            beams = args.num_beams
            gens = model.generate(
                **gen_inputs,
                max_length=args.generation_max_length,
                num_beams=beams
            )
            decoded = tokenizer.batch_decode(gens, skip_special_tokens=True)
            batch_size = in_ids.size(0)

            for i in range(batch_size):
                sample_gens = decoded[i*beams:(i+1)*beams]
                

                if prefix[i] == 'iso':
                    iso_labels.append(iso_l[i])
                    
                    poss_lbls = []
                    for gen_lbl in sample_gens:
                        poss_lbls.append(gen_lbl.strip())
                    
                    # choose the lbl with the greatest frequency
                    
                    most_common_lbl, _ = Counter(poss_lbls).most_common(1)[0]
                    iso_preds.append(most_common_lbl)
                    
                else:

                    em = 0
                    for p_sexp in sample_gens:
                        em += compute_exact_match(p_sexp, gold_sexps_mapped[i])

                        pred_sexps.append(p_sexp)
                        true_sexps.append(gold_sexps_mapped[i])

                    em_sum += em

                    # compute f1
                    # if k:
                    #     f1_sum += compute_f1(ranked[:k], gold_entities)
                    gen_count += 1

    # aggregate
    metrics = {}
    # import pdb; pdb.set_trace()
    metrics["iso_f1"] = f1_score(iso_labels, iso_preds, average="macro")

    metrics["exact_match"] = em_sum / max(gen_count, 1)

    return {'y_true': iso_labels, 'y_pred': iso_preds, 'pred_sexps': pred_sexps, 'true_sexps': true_sexps}

    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type",  choices=["text","graph","cod"], required=True)
    parser.add_argument("--backbone",    choices=["bert","t5","gpt2"], default="t5")
    parser.add_argument("--tasks",       type=str, default="iso")
    parser.add_argument("--weights",     type=str, default=None)
    parser.add_argument("--dataset",     default='grailqapp')
    parser.add_argument("--data_path",   type=str)
    parser.add_argument("--relations_path", default=None)
    parser.add_argument("--output_dir",  type=str)

    parser.add_argument("--input_max_length",      type=int, default=512)
    parser.add_argument("--generation_max_length", type=int, default=128)
    parser.add_argument("--max_seq_len",           type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=1,)

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

    # Loaded the data pile

    if args.dataset == 'grailqa':

        args.data_path = '/data/shire/projects/GenQA/ISOKBQA/isomorphism_pred_KBQA/data/grailqa/KBQA_walklet_edge_label_768_dim_data.dill'

        with open(args.data_path, "rb") as f:
            data = dill.load(f)


        train = data["train"][:int(len(data["train"])*0.7)]
        dev = data["train"][int(len(data["train"])*0.7):]
        test = data["dev"]
    
    if args.dataset == 'grailqapp':

        args.data_path = '/data/shire/projects/GenQA/ISOKBQA/isomorphism_pred_KBQA/data/grailqa/KBQA_walklet_edge_label_768_dim_data_test.dill'

        with open(args.data_path, "rb") as f:
            data = dill.load(f)


        train   = data['train']
        dev     = data['dev']
        test    = data["test"]

    all_samples = train + dev + test

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    wandb_df = pd.read_csv('wandb_runs.csv')
    # remove cases where the test iso f1 is NaN
    # wandb_df = wandb_df[~wandb_df['test_iso_f1'].isna()]

    wandb_df = wandb_df[(wandb_df['tasks']=='iso') & (wandb_df['State']=='finished') & (wandb_df['model_type']=='cod')]

    print(f"Found {len(wandb_df)} wandb runs to evaluate.")

    cols    = wandb_df.columns.tolist()

    mtl_flag = False
   
    for idx, row in wandb_df.iterrows():

        for col in cols:
            if col =='dataset': 
                continue
            if col in args:
                args.__dict__[col] = row[col]
        
        if 'MTL' in row['Name']:
            mtl_flag = True
        

        # OUTPUT_DIR_BASE="/data/shire/projects/GenQA/ISOKBQA/isomorphism_pred_KBQA/ckpts_rdutt"    
        pred_dir      = args.output_dir.replace('ckpts_rdutt', "predictions")
        print(pred_dir)

        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir, exist_ok=True)
        
        # import pdb; pdb.set_trace()
        # --- setup ---
        device = seed_everything(args.seed)
        # accelerator = Accelerator()
        # device = accelerator.device

        tasks = args.tasks.split(",") if args.tasks else []
        if args.weights:
            w = list(map(float, args.weights.split(",")))
            weights = dict(zip(tasks, w))
        else:
            weights = {t: 1.0 for t in tasks}

        print(f'Choice of the args is {args}')
        # --- data ---
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

        max_entities = max(len(e["node_data"]["node_dict"]) for e in all_samples)

        print("DEBUGGING: convert_entities is", args.convert_entities)

        if args.convert_entities:
            entity_tokens = [f"<E{i}>" for i in range(max_entities)]
            tokenizer.add_tokens(entity_tokens + list(special_tokens.values()))
        else:
            tokenizer.add_tokens(list(special_tokens.values()))

        if mtl_flag:
             test_loader = get_T5_dataloader(
                test, args, tokenizer, modes=tasks, convert_entities=args.convert_entities,
                global_label_map=label_map,
                batch_size=args.batch_size, shuffle=False
            )
        else:
            
            test_loader = get_dataloader(
                test, args, tokenizer, modes=tasks, convert_entities=args.convert_entities,
                global_label_map=label_map,
                batch_size=args.batch_size, shuffle=False
            )

        # --- model ---

        ### using the MTL scenario for the text model in a multi-task setting

        if not mtl_flag:
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


        ### using the MTL scenario for the T5 model in a multi-task setting

        if args.model_type == "text" and mtl_flag:
            if args.backbone == "t5":
                model = T5MultiTaskGenModel(args, tokenizer, modes=tasks)            
            model.backbone.resize_token_embeddings(len(tokenizer))
            # model.backbone.gradient_checkpointing_enable()

        model.to(device)
        
        global_step = 0
        stopper = EarlyStopping(patience=args.patience)
        best_val = -float("inf")

        best_ckpt = os.path.join(args.output_dir, "best_model.pt")

        if os.path.exists(best_ckpt) is False:
            print(f"Best checkpoint {best_ckpt} does not exist. Skipping evaluation.")
            continue

        # --- final evaluation + save preds ---
        best_state_dict = torch.load(best_ckpt, weights_only=False)
        model.load_state_dict(best_state_dict)
        
        model.to(device)
        if args.model_type == "cod":
            # test_metrics = eval_epoch_cod(model, test_loader, device)
            results      = predict_epoch_cod(model, test_loader, device)

        elif args.model_type == "graph":
            # test_metrics = eval_epoch_graph(model, test_loader, device)
            results      = predict_epoch_graph(model, test_loader, device)
        elif args.model_type == 'text':

            if mtl_flag:
                # test_metrics = eval_epoch_MTL_text(args, model, test_loader, device, tasks, tokenizer)
                results      = predict_epoch_MTL_text(args, model, test_loader, device, tasks, tokenizer)
            
            else:
                # test_metrics = eval_epoch_text(args, model, test_loader, device, tasks, tokenizer)
                results      = predict_epoch_text(args, model, test_loader, device, tasks, tokenizer)

        # print("Test metrics:", test_metrics)
        # print(f"Current row metrics {row['test_iso_f1'] if 'test_iso_f1' in row else 'N/A'}")

        y_true, y_pred = results['y_true'], results['y_pred']

        print("Overall Iso F1:", f1_score(y_true, y_pred, average="macro"))

        # assert os.path.exists(args.output_dir), f"Output dir {args.output_dir} does not exist."

        # pred_dir    = args.output_dir.replace(OUTPUT_DIR_BASE, "predictions/")
        
        if args.dataset == 'grailqa':
            with open(os.path.join(pred_dir, "grailqa_dev_predictions.json"), "w") as f:
                json.dump(results, f, indent=4)

        elif args.dataset == 'grailqapp':
            with open(os.path.join(pred_dir, "grailqapp_test_predictions_with_isologits.json"), "w") as f:
                    json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()