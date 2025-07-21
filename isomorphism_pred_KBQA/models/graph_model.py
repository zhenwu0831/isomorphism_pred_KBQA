# models/graph_model.py

import torch
import torch.nn as nn
from torch_geometric.nn import FastRGCNConv, RGATConv, GCNConv, global_max_pool, global_mean_pool
from torch_geometric.data import Batch
from transformers import AutoModel, T5EncoderModel


class GNNDropout(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.drop = nn.Dropout(dropout)

    def forward(self, inp):
        x, edge_index, edge_type = inp
        return self.drop(x), edge_index, edge_type


class GNNSoftmax(nn.Module):
    def forward(self, inp):
        x, edge_index, edge_type = inp
        # apply softmax over features
        return torch.softmax(x, dim=-1), edge_index, edge_type


class GNNConv(nn.Module):
    """
    Single GNN layer: RGCN, RGAT, or simple GCN.
    """
    def __init__(self, gnn_model: str, in_dim: int, out_dim: int, num_rels: int):
        super().__init__()
        self.gnn_model = gnn_model
        if gnn_model == 'rgcn':
            self.gnn = FastRGCNConv(in_channels=in_dim, out_channels=out_dim, num_relations=num_rels)
        elif gnn_model == 'rgat':
            self.gnn = RGATConv(in_channels=in_dim, out_channels=out_dim, num_relations=num_rels)
        elif gnn_model == 'gcn':
            self.gnn = GCNConv(in_channels=in_dim, out_channels=out_dim)
        else:
            raise ValueError(f"Unknown GNN model '{gnn_model}'")

    def forward(self, inp):
        x, edge_index, edge_type = inp
        # FastRGCNConv & RGATConv expect edge_type; GCNConv ignores it
        if self.gnn_model == 'gcn':
            return self.gnn(x, edge_index), edge_index, edge_type
        else:
            return self.gnn(x, edge_index, edge_type), edge_index, edge_type


class DeepNet(nn.Module):
    """
    Stacks multiple GNNConv + GNNSoftmax + GNNDropout layers.
    """
    def __init__(self, params):
        super().__init__()
        layers = []
        # first layer: input_dim = text_emb_dim + node_emb_dim + 1
        if "iso" in params.modes:
            in0 = params.text_emb_dim + params.node_emb_dim
        else:
            in0 = params.text_emb_dim + params.node_emb_dim + 1
        hid = params.hidden_dim
        # add first conv block
        layers.append(GNNConv(params.gnn_model, in0, hid, params.num_rels))
        layers.append(GNNSoftmax())
        layers.append(GNNDropout(params.dropout))
        # add intermediate blocks
        for _ in range(params.gnn_depth - 2):
            layers.append(GNNConv(params.gnn_model, hid, hid, params.num_rels))
            layers.append(GNNSoftmax())
            layers.append(GNNDropout(params.dropout))
        # final conv block
        layers.append(GNNConv(params.gnn_model, hid, hid, params.num_rels))
        layers.append(GNNSoftmax())
        layers.append(GNNDropout(params.dropout))

        self.gnn_module = nn.Sequential(*layers)

    def forward(self, x_edge):
        """
        x_edge: tuple (x, edge_index, edge_type)
        returns: x' same shape as x but transformed
        """
        x, edge_index, edge_type = x_edge
        x, _, _ = self.gnn_module((x, edge_index, edge_type))
        return x


class GraphMultiTaskModel(nn.Module):
    """
    Graph-based multi-task model supporting:
      - entity ranking ("ranking")
      - isomorphism classification ("iso")
    """
    def __init__(self, params, modes=("ranking", "iso")):
        super().__init__()
        assert set(modes).issubset({"ranking", "iso"}), "Unsupported mode"
        self.modes = set(modes)
        self.params = params
        self.params.modes = modes

        # -- question encoder --
        model_name = params.ptlm_model
        if "t5" in model_name:
            self.text_encoder = T5EncoderModel.from_pretrained(model_name)
        else:
            self.text_encoder = AutoModel.from_pretrained(model_name)

        self.text_encoder.gradient_checkpointing_enable()
        self.text_encoder.config.use_cache = False  # disable cache for training

        # import torch.nn as nn
        # for m in self.text_encoder.modules():
        #     if isinstance(m, nn.Dropout):
        #         m.p = 0.0
        # self.text_encoder.gradient_checkpointing_disable()

        text_hid = self.text_encoder.config.hidden_size
        # propagate text_emb_dim for GNN input size
        self.params.text_emb_dim = text_hid

        # -- GNN encoder --
        self.gnn = DeepNet(params)
        self.layer_norm = nn.LayerNorm(params.hidden_dim)

        self.pooling = global_max_pool

        # -- isomorphism classification head --
        if "iso" in self.modes:
            self.iso_dropout = nn.Dropout(params.dropout)
            self.iso_classifier = nn.Linear(params.hidden_dim, params.num_classes)

        # -- entity ranking head --
        if "ranking" in self.modes:
            # takes [gnn_out || head_entity]
            self.ans_classifier = nn.Linear(params.hidden_dim, 1)
            self.ans_sigmoid = nn.Sigmoid()

    def forward(self, batch: dict) -> dict:
        """
        batch should contain:
          - 'input_ids', 'attention_mask' for question
          - 'gnn_data': a PyG Batch with .x, .edge_index, .edge_type, .head_x, .answer_x, .batch
        Returns dict with any of:
          - 'iso_logits': (B, num_classes)
          - 'entity_scores': (total_nodes,)
        """
        device = next(self.parameters()).device

        # -- encode question --
        q_ids = batch["input_ids"].to(device)
        q_mask = batch["attention_mask"].to(device)
        txt_out = self.text_encoder(input_ids=q_ids, attention_mask=q_mask, return_dict=True)
        q_emb = txt_out.last_hidden_state.mean(dim=1)  # (B, H)

        out = {}

        # iso and ranking tasks are similar, just that iso is graph classification and ranking is node classification
        gdata = batch["gnn_data"].to(device)
        # expand question to each node
        b_idx = gdata.batch
        q_exp = q_emb[b_idx]                   # (N, H)
        node_feats = gdata.x                   # (N, node_emb_dim)
        head_flag  = gdata.head_x.unsqueeze(-1).float()  # (N,1)

        # concat features
        if "iso" in self.modes:
            g_in = torch.cat([q_exp, node_feats], dim=-1)  # (N, H+feat)
        elif "ranking" in self.modes:
            g_in = torch.cat([q_exp, node_feats, head_flag], dim=-1)  # (N, H+feat+1)

        g_out = self.gnn((g_in, gdata.edge_index, gdata.edge_type))
        g_norm = self.layer_norm(g_out)         # (N, hidden_dim)

        graph_representation = self.pooling(g_norm, gdata.batch)  # (B, hidden_dim)
        # -- isomorphism classification --
        if "iso" in self.modes:
            logits = self.iso_classifier(self.iso_dropout(graph_representation))
            out["iso_logits"] = logits  # (B, num_classes)

        # score nodes
        if "ranking" in self.modes:
            node_scores = self.ans_classifier(g_norm)
            node_scores = self.ans_sigmoid(node_scores).squeeze(-1)  #
            # mask out head nodes
            node_scores = node_scores * (1 - head_flag.squeeze(-1))
            out["entity_scores"] = node_scores  # (N,)

        return out, graph_representation