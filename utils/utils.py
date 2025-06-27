import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import numpy as np
from typing import List, Dict

def compute_global_label_map(
    train_data: List[Dict], dev_data: List[Dict], test_data: List[Dict]
) -> (Dict[str, int], int):
    """
    Build a global mapping from isomorphism labels to integers.
    """
    unique = sorted({e["isomorphism"] for e in (train_data + dev_data + test_data)})
    label_map = {iso: idx for idx, iso in enumerate(unique)}
    return label_map, len(label_map)

def extract_entity_embeddings(hidden_states: torch.Tensor,
                              entity_spans: list) -> list:
    """
    Given encoder hidden states (batch_size, seq_len, hidden_dim)
    and entity_spans: list of dicts [{"<E0>": [[i,j],...], ...}, ...] per batch,
    return list of tensors (num_entities, hidden_dim) per sample.
    """
    batch_embeddings = []
    for b_idx, spans in enumerate(entity_spans):
        emb_list = []
        for tag, ranges in spans.items():
            # average token embeddings for each span
            span_embs = []
            for (start, end) in ranges:
                span_embs.append(hidden_states[b_idx, start:end+1].mean(dim=0))
            if span_embs:
                emb_list.append(torch.stack(span_embs).mean(dim=0))
        if emb_list:
            batch_embeddings.append(torch.stack(emb_list))
        else:
            # no entities found: zero vector
            hidden_dim = hidden_states.size(-1)
            batch_embeddings.append(torch.zeros((1, hidden_dim), device=hidden_states.device))
    return batch_embeddings

def seed_everything(seed: int = 42) -> torch.device:
    """
    Set all relevant random seeds and return the torch device.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pad_entity_labels(
    entity_labels: List[Dict[str,int]],
    device: torch.device
) -> (torch.Tensor, torch.Tensor):
    """
    Given a list of per-sample `{entity: label}`, pad them to a uniform tensor:
      - returns (labels, mask), both shaped (batch, max_entities).
    """
    max_e = max(len(d) for d in entity_labels)
    labels, masks = [], []
    for d in entity_labels:
        vals = list(d.values())
        pad_len = max_e - len(vals)
        lab = torch.tensor(vals + [0]*pad_len, dtype=torch.float, device=device)
        m   = torch.tensor([1]*len(vals) + [0]*pad_len, dtype=torch.float, device=device)
        labels.append(lab)
        masks.append(m)
    return torch.stack(labels), torch.stack(masks)

def kco_info_nce_loss(A: torch.Tensor, B: torch.Tensor, temp=0.1, allow_gradient=False):
    """
    Your KCo-InfoNCE loss between two sets of reps A and B.
    """
    A_norm = F.normalize(A, dim=1)
    B_norm = (
        F.normalize(B.detach(), dim=1)
        if not allow_gradient
        else F.normalize(B, dim=1)
    )
    S = torch.matmul(A_norm, B_norm.T)
    positives = torch.diag(S).unsqueeze(1)
    mask = ~torch.eye(S.size(0), dtype=torch.bool, device=S.device)
    negatives = S[mask].view(S.size(0), -1)
    logits = torch.cat([positives, negatives], dim=1) / temp
    labels = torch.zeros(S.size(0), dtype=torch.long, device=S.device)
    return nn.CrossEntropyLoss()(logits, labels)

def execute_sparql(sparql):
    """
    Executes the SPARQL query and returns the answers.
    """
    try:
        qres = g.query(sparql)
        answers = [str(a[0]) for a in qres]
        return answers
    except Exception as e:
        print(e)
        return []