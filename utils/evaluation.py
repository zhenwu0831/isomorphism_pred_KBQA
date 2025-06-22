# utils/evaluation.py

import torch
from typing import List, Any

eps = 1e-5

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