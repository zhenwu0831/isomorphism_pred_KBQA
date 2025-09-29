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
from typing import Dict, List, Union
from dataclasses import dataclass


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data import DataLoader
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
# from datasets import load_dataset, concatenate_datasets


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




def train_epoch_text(
    args, model, loader, optimizer, scheduler,
    global_step, device, tasks, weights
):
    model.train()
    total_loss = 0.0

    for tv in tqdm(loader, desc="Training", leave=False):
        
        optimizer.zero_grad()
        
        in_ids = tv["input_ids"].to(device)
        attn   = tv["attention_mask"].to(device)

        gen_l = tv.get("labels").to(device)
        
        kwargs = {"input_ids": in_ids, "attention_mask": attn, "labels": gen_l}
        
        # kwargs["labels"] = gen_l.to(device)
        
        out, _ = model(**kwargs)
        loss = torch.tensor(0.0, device=device)

        gen_loss, iso_loss, ranking_loss = None, None, None

        
        gen_loss = out["gen_outputs"].loss
        loss += gen_loss

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

    from sklearn.metrics import f1_score
    metrics["iso_f1"] = f1_score(iso_labels, iso_preds, average="macro")


    metrics["exact_match"] = em_sum    / max(gen_count, 1)
    

    return metrics





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

    label_map, nc = compute_global_label_map(train, dev, test)
    args.num_classes = nc
    
    # --- tokenizer & loaders ---
    tokenizer = AutoTokenizer.from_pretrained(args.ptlm_model)
   
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

    train_loader = get_T5_dataloader(
        train, args, tokenizer, modes=tasks, convert_entities=args.convert_entities,
        global_label_map=label_map,
        batch_size=args.batch_size, shuffle=True
    )
    dev_loader = get_T5_dataloader(
        dev, args, tokenizer, modes=tasks, convert_entities=args.convert_entities,
        global_label_map=label_map,
        batch_size=args.batch_size, shuffle=False
    )
    
    test_loader = get_T5_dataloader(
        test, args, tokenizer, modes=tasks, convert_entities=args.convert_entities,
        global_label_map=label_map,
        batch_size=args.batch_size, shuffle=False
    )

    # import pdb; pdb.set_trace()

    # --- model ---
    if args.model_type == "text":
        if args.backbone == "t5":
            model = T5MultiTaskGenModel(args, tokenizer, modes=tasks)
        
        model.backbone.resize_token_embeddings(len(tokenizer))
        # model.backbone.gradient_checkpointing_enable()

    
    model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = AdamW(trainable_params, lr=args.lr)

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
        train_loss, global_step = train_epoch_text(
            args, model, train_loader,
            optimizer, scheduler, global_step,
            device, tasks, weights
        )
        metrics = eval_epoch_text(args, model, dev_loader, device, tasks, tokenizer)

        primary = ("iso_f1" if args.model_type=="graph"
                    else "iso_f1" if args.model_type=="cod"
                    else "hits@k_ranking" if "ranking" in tasks
                    else "exact_match" if "generation" in tasks
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
            torch.save(model.state_dict(), best_ckpt)
            print(f"↳ saved best {primary}={best_val:.4f}")
        
        stopper(primary_val)

        if stopper.should_stop():
            print("Early stopping")
            break

    # --- final evaluation + save preds ---
    best_state_dict = torch.load(best_ckpt, weights_only=False)
    model.load_state_dict(best_state_dict)
    
    model.to(device)
    
    test_metrics = eval_epoch_text(args, model, test_loader, device, tasks, tokenizer)

    # log final metrics
    for k,v in test_metrics.items():
        wandb.log({f"test_{k}": v})

    wandb.finish()
    # exit()


if __name__ == "__main__":
    main()