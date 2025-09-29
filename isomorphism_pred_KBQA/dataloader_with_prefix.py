from typing import List, Dict, Any, Optional, Sequence
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch.nn.utils.rnn import pad_sequence
import torch
from copy import deepcopy
from tqdm import tqdm


class UnifiedDataset(Dataset):
    def __init__(
        self,
        raw_data: List[Dict],
        params: Any,
        tokenizer: PreTrainedTokenizer,
        modes: Sequence[str],
        convert_entities: bool = False,
        global_label_map: Optional[Dict[str, int]] = None
    ):
        assert set(modes).issubset({"generation", "iso"})
        # assert global_label_map is not None, "Need label_map for isomorphism mode"
        self.params = params
        self.tokenizer = tokenizer
        self.modes = set(modes)
        self.convert_entities = convert_entities
        self.label_map = global_label_map

        self.samples = self._preprocess(raw_data)
        self.samples = self._tokenize_all()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

    def _preprocess(self, data: List[Dict]) -> List[Dict]:
        out = []
        for e in data:
            if not e.get("question") or not e.get("kg_tuples"):
                continue
            if not e.get("s_expression") or not e.get("isomorphism"):
                continue

            q = e["question"].strip()
            if e.get("entities"):
                ents = " ".join(f"{subj}: {obj}" for subj, obj in e["entities"])
            elif e.get("head_entities"):
                ents = " | ".join(f"{subj}: {obj}: {name}" for subj, obj, name in e["head_entities"])
            else:
                print(f"Warning: No entities in {e.get('ID', e.get('id'))}, skipping.")
                continue
            kg = " | ".join(" ".join(t) for t in e["kg_tuples"])
            struct = f"{ents} | {kg}"

            eg = deepcopy(e)
            eg["task_type"] = "generation"
            eg["text_in"] = f"Generate an s-expression to answer the question: {q}"
            eg["struct_in"] = struct
            out.append(eg)

            ei = deepcopy(e)
            ei["task_type"] = "iso"
            ei["text_in"] = f"Predict the isomorphism for this question: {q}"
            ei["struct_in"] = struct
            out.append(ei)
        return out

    def _map_entities(self, text: str, node_dict: Dict[str, int]) -> str:
        for ent, idx in node_dict.items():
            text = text.replace(ent, f"<E{idx}>")
        return text

    def _tokenize_all(self) -> List[Dict]:
        toks = []
        for e in tqdm(self.samples, desc="Tokenizing unified dataset"):
            node_dict = e["node_data"]["node_dict"]
            struct = self._map_entities(e["struct_in"], node_dict) if self.convert_entities else e["struct_in"]

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
                "task_type": e["task_type"],
                "input_ids": torch.LongTensor(tkn["input_ids"]),
                "attention_mask": torch.LongTensor(tkn["attention_mask"])
            }

            # Unified target processing for generation and iso
            if e["task_type"] == "generation":
                target_str = e["s_expression"]
            elif e["task_type"] == "iso":
                target_str = e["isomorphism"]  # e.g., "T-0", "T-1", etc.

            if self.convert_entities:
                target_str = self._map_entities(target_str, node_dict)

            t2 = self.tokenizer(
                target_str,
                padding="max_length",
                truncation=True,
                max_length=self.params.generation_max_length
            )
            labels = torch.LongTensor(t2["input_ids"])
            labels[labels == self.tokenizer.pad_token_id] = -100
            tv["labels"] = labels

            # Store gold text for optional logging/eval
            if e["task_type"] == "generation":
                tv["gold_sexp"] = target_str
                tv["gold_sexp_raw"] = e["s_expression"]
                tv["gold_iso"] = "[DUMMY_ISO]"  # Placeholder to ensure consistent keys
            elif e["task_type"] == "iso":
                tv["gold_iso"] = target_str
                tv["gold_sexp"] = "[DUMMY_SEXP]"
                tv["gold_sexp_raw"] = "[DUMMY_SEXP_RAW]"

            q2 = self.tokenizer(
                e["question"],
                padding="max_length",
                truncation=True,
                max_length=self.params.max_seq_len
            )
            # if e["task_type"] == "iso":
            #     import pdb; pdb.set_trace()

            gv = {
                "input_ids": torch.LongTensor(q2["input_ids"]),
                "attention_mask": torch.LongTensor(q2["attention_mask"]),
                "gnn_data": Data(
                    x=torch.tensor(e["node_data"]["x"], dtype=torch.float),
                    edge_index=torch.tensor(e["node_data"]["edge_index"], dtype=torch.long),
                    edge_type=torch.tensor(e["node_data"]["edge_type"], dtype=torch.long),
                    answer_x=torch.tensor(e["node_data"]["answer_x"], dtype=torch.long),
                    head_x=torch.tensor(e["node_data"]["head_x"], dtype=torch.long),
                ),
            }

            toks.append({"text_view": tv, "graph_view": gv, "node_dict": node_dict})
        return toks


def collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch = {"text_view": {}, "graph_view": {}, "node_dict": []}
    tvs = [s["text_view"] for s in samples]

    batch["text_view"]["ID"] = [t["ID"] for t in tvs]
    batch["text_view"]["question"] = [t["question"] for t in tvs]
    batch["text_view"]["task_type"] = [t["task_type"] for t in tvs]

    batch["text_view"]["input_ids"] = pad_sequence(
        [t["input_ids"] for t in tvs], batch_first=True
    )
    batch["text_view"]["attention_mask"] = pad_sequence(
        [t["attention_mask"] for t in tvs], batch_first=True
    )

    labels = []
    for t in tvs:
        l = t["labels"]
        labels.append(l if l.ndim > 0 else l.unsqueeze(0))
    batch["text_view"]["labels"] = pad_sequence(
        labels, batch_first=True, padding_value=-100
    )

    # if "gold_sexp" in tvs[0]:
    #     batch["text_view"]["gold_sexp"] = [t["gold_sexp"] for t in tvs]
    #     batch["text_view"]["gold_sexp_raw"] = [t["gold_sexp_raw"] for t in tvs]
    # if "gold_iso" in tvs[0]:
    #     batch["text_view"]["gold_iso"] = [t["gold_iso"] for t in tvs]

    batch["text_view"]["gold_sexp"] = [t["gold_sexp"] for t in tvs if "gold_sexp" in t]
    batch["text_view"]["gold_sexp_raw"] = [t["gold_sexp_raw"] for t in tvs if "gold_sexp_raw" in t]
    batch["text_view"]["gold_iso"] = [t["gold_iso"] for t in tvs if "gold_iso" in t]

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
    batch["node_dict"] = [s["node_dict"] for s in samples]
    return batch


def get_dataloader(
    raw_data: List[Dict],
    params: Any,
    tokenizer: PreTrainedTokenizer,
    modes: Sequence[str],  # e.g., ["generation", "iso"]
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