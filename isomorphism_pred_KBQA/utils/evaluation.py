# utils/evaluation.py
import os,sys 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append("../")

import torch
from typing import List, Any
from compute_isomorphisms.code.create_sem_matcher import *

eps = 1e-5

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ONTOLOGY_DIR = os.path.join("../", "ontology")

FB_ROLES_FILE      = os.path.join(ONTOLOGY_DIR, "fb_roles")
FB_TYPES_FILE      = os.path.join(ONTOLOGY_DIR, "fb_types")
REVERSE_PROPS_FILE = os.path.join(ONTOLOGY_DIR, "reverse_properties")

# load once
reverse_properties, relation_dr, relations, upper_types, types, common_rels = process_ontology(
    fb_roles_file=FB_ROLES_FILE,
    fb_types_file=FB_TYPES_FILE,
    reverse_properties_file=REVERSE_PROPS_FILE
)

matcher = SemanticMatcher(
    reverse_properties,
    relation_dr,
    relations,
    upper_types,
    types
)

def hits_at_k_graph(
    node_scores: torch.Tensor,
    answer_nodes: torch.Tensor
) -> (float, List[List[int]], List[List[int]]):
    """
    Compute Hits@K for graph-based ranking.

    Returns:
        graph_acc: Sum of per-sample Hits@K accuracy.
        preds: List of predicted top-K indices per sample.
        answers: List of gold answer indices per sample.
    """
    graph_acc = 0.0
    preds, answers = [], []
    batch_size = node_scores.size(0)

    for i in range(batch_size):
        scores = node_scores[i]
        gold_mask = answer_nodes[i].squeeze(-1)
        sorted_idx = torch.argsort(scores, descending=True).cpu().tolist()
        gold_idx = torch.where(gold_mask == 1)[0].cpu().tolist()
        k = len(gold_idx)

        if k > 0:
            topk = sorted_idx[:k]
            graph_acc += len(set(topk) & set(gold_idx)) / (k + eps)

        preds.append(sorted_idx)
        answers.append(gold_idx)

    return graph_acc, preds, answers


def hits_at_k_text(
    batch_gold_entities: List[set],
    pred_ranked_entities: List[List[Any]]
) -> (float, List[List[Any]], List[List[Any]]):
    """
    Compute Hits@K for text-based entity ranking.

    Returns:
        text_acc: Mean Hits@K accuracy over batch.
        preds: List of full ranked predictions per sample.
        answers: List of gold answer lists per sample.
    """
    text_acc = 0.0
    preds, answers = [], []
    total = len(batch_gold_entities)

    for gold_set, ranked in zip(batch_gold_entities, pred_ranked_entities):
        preds.append(ranked)
        answers.append(list(gold_set))
        k = len(gold_set)
        if k > 0:
            topk = ranked[:k]
            text_acc += len(set(topk) & set(gold_set)) / k

    text_acc /= max(total, 1)
    return text_acc, preds, answers

def compute_hits_at_k(all_predictions, all_gold_labels):
    """
    Compute HITS@k where k is dynamically set to the number of gold answers per query.

    Args:
        all_predictions (torch.Tensor): Shape (num_samples, num_entities) with model scores.
        all_gold_labels (torch.Tensor): Shape (num_samples, num_entities), binary labels (1 for gold answers, 0 otherwise).

    Returns:
        float: Mean HITS@k score across all queries.
    """
    num_samples, num_entities = all_predictions.shape
    hits_at_k_scores = 0

    for i in range(num_samples):
        # import pdb; pdb.set_trace()  # Debugging line to inspect variables
        gold_indices = (all_gold_labels[i] > 0).nonzero(as_tuple=True)[0]  # Indices of gold answers
        k = len(gold_indices)  # k = Number of gold answers

        if k == 0:
            continue  # Skip queries with no gold answers

        # Get top-k predicted indices
        top_k_indices = torch.topk(all_predictions[i], k=k, largest=True, sorted=True).indices

        # Compute intersection
        intersection = set(top_k_indices.tolist()) & set(gold_indices.tolist())

        # Compute HITS@k score for this sample
        hit_score = len(intersection) / k
        hits_at_k_scores += hit_score

    return hits_at_k_scores

def compute_exact_match(pred, ref):
    # import pdb; pdb.set_trace() 
    em = int(matcher.same_logical_form(pred, ref))
    return em

def compute_f1(pred_list, gold_list):
    """
    Compute precision, recall, and F1-score for a single prediction-gold pair.
    Args:
        pred: A prediction string (e.g., "A|B|C").
        gold: A gold label string (e.g., "B|C|D").
    Returns:
        A dictionary with precision, recall, and F1-score.
    """
    # pred_list = [p.strip() for p in pred.split("|")]
    # gold_list = [g.strip() for g in gold.split("|")]

    pred_set = set(pred_list)
    gold_set = set(gold_list)

    true_positives = len(pred_set & gold_set)  # Intersection of prediction and gold
    precision = true_positives / len(pred_set) if len(pred_set) > 0 else 0
    recall = true_positives / len(gold_set) if len(gold_set) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1