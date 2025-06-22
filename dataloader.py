from typing import List, Dict, Any, Optional, Sequence
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch.nn.utils.rnn import pad_sequence
import torch
from copy import deepcopy
from tqdm import tqdm

class UnifiedDataset(Dataset):
    """
    A single Dataset for three modes:
      - generation: token-to-S-expression
      - iso: classification of isomorphism
      - ranking: entity-ranking as BCE over entity spans

    Args:
      raw_data:          list of raw examples
      params:            has input_max_length, generation_max_length, max_seq_len
      tokenizer:         a HuggingFace tokenizer
      modes:             subset of {"generation","iso","ranking"}
      convert_entities:  if True, replace IDs with <E#>
      global_label_map:  required if "iso" in modes
    """
    def __init__(
        self,
        raw_data: List[Dict],
        params: Any,
        tokenizer: PreTrainedTokenizer,
        modes: Sequence[str],
        convert_entities: bool = False,
        global_label_map: Optional[Dict[str, int]] = None
    ):
        assert set(modes).issubset({"generation", "iso", "ranking"})
        if "iso" in modes:
            assert global_label_map is not None, "Need label_map for isomorphism mode"
        self.params = params
        self.tokenizer = tokenizer
        self.modes = set(modes)
        self.convert_entities = convert_entities
        self.label_map = global_label_map or {}

        # 1) filter & annotate raw examples
        self.samples = self._preprocess(raw_data)

        # 2) tokenize everything up-front
        self.samples = self._tokenize_all()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

    def _preprocess(self, data: List[Dict]) -> List[Dict]:
        """
        Build text_in and struct_in but do not drop samples unless missing necessary fields for _any_ enabled mode.
        """
        out = []
        for e in data:
            if not e.get("question") or not e.get("kg_tuples") or not e.get("entities"):
                continue
            if "generation" in self.modes and not e.get("s_expression"):
                continue
            if ("iso" in self.modes or "ranking" in self.modes) and not e.get("answers"):
                continue

            q = e["question"].strip()
            ents = " ".join(f"{subj}: {obj}" for subj, obj in e["entities"])
            kg = " | ".join(" ".join(t) for t in e["kg_tuples"])
            e["text_in"] = q
            e["struct_in"] = f"{ents} | {kg}"
            out.append(deepcopy(e))
        return out

    def _map_entities(self, text: str, node_dict: Dict[str, int]) -> str:
        for ent, idx in node_dict.items():
            text = text.replace(ent, f"<E{idx}>")
        return text

    def _tokenize_all(self) -> List[Dict]:
        toks = []
        for e in tqdm(self.samples, desc="Tokenizing unified dataset"):
            node_dict = e["node_data"]["node_dict"]
            struct = e["struct_in"]
            if self.convert_entities:
                struct = self._map_entities(struct, node_dict)

            # text+KG view
            seq = f"{e['text_in']} ; structured knowledge: {struct}"
            tkn = self.tokenizer(
                seq,
                padding="max_length",
                truncation=True,
                max_length=self.params.input_max_length
            )

            tv = {
                "ID": e.get("ID", e.get("id")),
                "question": e["question"],
                "input_ids": torch.LongTensor(tkn["input_ids"]),
                "attention_mask": torch.LongTensor(tkn["attention_mask"])
            }

            if "generation" in self.modes:
                sexp = e["s_expression"]
                t2 = self.tokenizer(
                    sexp,
                    padding="max_length",
                    truncation=True,
                    max_length=self.params.generation_max_length
                )
                labels = torch.LongTensor(t2["input_ids"])
                labels[labels == self.tokenizer.pad_token_id] = -100
                tv["labels"] = labels
                tv["gold_sexp"] = sexp

            if "iso" in self.modes:
                idx = self.label_map[e["isomorphism"]]
                tv["iso_label"] = torch.tensor(idx, dtype=torch.long)

            if "ranking" in self.modes:
                ans = [a[0][3:] if a[0].startswith("ns:") else a[0] for a in e["answers"]]
                if self.convert_entities:
                    ans = [self._map_entities(a, node_dict) for a in ans]
                spans = self._extract_entity_spans(
                    tkn["input_ids"], node_dict
                )
                labels = {k: int(k in ans) for k in spans}
                sorted_keys = sorted(spans, key=lambda k: int(k[2:-1]))
                tv["entity_spans"] = {k: spans[k] for k in sorted_keys}
                tv["entity_labels"] = {k: labels[k] for k in sorted_keys}
                tv["answers"] = ans

            # graph view
            q2 = self.tokenizer(
                e["question"],
                padding="max_length",
                truncation=True,
                max_length=self.params.max_seq_len
            )
            gv = {
                "input_ids": torch.LongTensor(q2["input_ids"]),
                "attention_mask": torch.LongTensor(q2["attention_mask"]),
                "gnn_data": Data(
                    x=torch.tensor(e["node_data"]["x"]),
                    edge_index=torch.tensor(e["node_data"]["edge_index"]),
                    edge_type=torch.tensor(e["node_data"]["edge_type"]),
                    answer_x=torch.tensor(e["node_data"]["answer_x"]),
                    head_x=torch.tensor(e["node_data"]["head_x"])
                ),
                "node_dict": {v: k for k, v in node_dict.items()}
            }

            toks.append({"text_view": tv, "graph_view": gv})
        return toks

    def _extract_entity_spans(
        self,
        input_ids: List[int],
        node_dict: Dict[str, int]
    ) -> Dict[str, List[List[int]]]:
        ent2span = {}
        for ent, idx in node_dict.items():
            tag = f"<E{idx}>"
            tok = self.tokenizer.tokenize(tag)
            ids = self.tokenizer.convert_tokens_to_ids(tok)
            spans = []
            for i in range(len(input_ids) - len(ids) + 1):
                if input_ids[i:i + len(ids)] == ids:
                    spans.append([i, i + len(ids) - 1])
            if spans:
                ent2span[tag] = spans
        return ent2span


def collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch = {"text_view": {}, "graph_view": {}}
    tvs = [s["text_view"] for s in samples]
    batch["text_view"]["ID"] = [t["ID"] for t in tvs]
    batch["text_view"]["question"] = [t["question"] for t in tvs]
    batch["text_view"]["input_ids"] = pad_sequence(
        [t["input_ids"] for t in tvs], batch_first=True
    )
    batch["text_view"]["attention_mask"] = pad_sequence(
        [t["attention_mask"] for t in tvs], batch_first=True
    )
    if "labels" in tvs[0]:
        batch["text_view"]["labels"] = pad_sequence(
            [t["labels"] for t in tvs], batch_first=True, padding_value=-100
        )
    if "iso_label" in tvs[0]:
        batch["text_view"]["iso_label"] = torch.stack([
            t["iso_label"] for t in tvs
        ])
    if "entity_spans" in tvs[0]:
        batch["text_view"]["entity_spans"] = [t["entity_spans"] for t in tvs]
        batch["text_view"]["entity_labels"] = [t["entity_labels"] for t in tvs]
        batch["text_view"]["answers"] = [t["answers"] for t in tvs]

    gvs = [s["graph_view"] for s in samples]
    batch["graph_view"]["input_ids"] = pad_sequence(
        [g["input_ids"] for g in gvs], batch_first=True
    )
    batch["graph_view"]["attention_mask"] = pad_sequence(
        [g["attention_mask"] for g in gvs], batch_first=True
    )
    batch["graph_view"]["gnn_data"] = Batch.from_data_list(
        [g["gnn_data"] for g in gvs]
    )
    batch["graph_view"]["node_dict"] = [g["node_dict"] for g in gvs]
    return batch


def get_dataloader(
    raw_data: List[Dict],
    params: Any,
    tokenizer: PreTrainedTokenizer,
    modes: Sequence[str],
    convert_entities: bool = False,
    global_label_map: Optional[Dict[str, int]] = None,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 2
) -> DataLoader:
    ds = UnifiedDataset(
        raw_data, params, tokenizer,
        modes=modes,
        convert_entities=convert_entities,
        global_label_map=global_label_map
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )