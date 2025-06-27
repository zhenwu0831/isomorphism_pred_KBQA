import torch
import torch.nn as nn
from transformers import AutoModel, T5ForConditionalGeneration, GPT2LMHeadModel, PreTrainedTokenizer
from utils.utils import extract_entity_embeddings


class BertMultiTaskModel(nn.Module):
    """
    BERT backbone supporting:
      - isomorphism classification ("iso")
      - entity ranking ("ranking")
    """
    def __init__(self, params, tokenizer: PreTrainedTokenizer, modes=("iso", "ranking")):
        super().__init__()
        assert set(modes).issubset({"iso", "ranking"}), "Unsupported mode"
        self.modes = set(modes)
        self.tokenizer = tokenizer
        self.backbone = AutoModel.from_pretrained(params.ptlm_model)
        self.hidden = self.backbone.config.hidden_size
        if "iso" in self.modes:
            self.classifier = nn.Linear(self.hidden, params.num_classes)
            self.dropout = nn.Dropout(params.dropout)
        if "ranking" in self.modes:
            self.entity_scorer = nn.Linear(self.hidden, 1)
            self.sigmoid = nn.Sigmoid()

    def forward(self,
                input_ids: torch.LongTensor,
                attention_mask: torch.LongTensor,
                entity_spans: list = None):
        outputs = self.backbone(input_ids=input_ids,
                             attention_mask=attention_mask,
                             return_dict=True)
        seq_hidden = outputs.last_hidden_state  # (B, L, H)
        # pooled = seq_hidden[:, 0]  # CLS token
        pooled = outputs.pooler_output
        results = {}
        if "iso" in self.modes:
            x = self.dropout(pooled)
            results['iso_logits'] = self.classifier(x)
        if "ranking" in self.modes:
            ent_embs = extract_entity_embeddings(seq_hidden, entity_spans)
            padded = nn.utils.rnn.pad_sequence(ent_embs, batch_first=True)
            scores = self.entity_scorer(padded).squeeze(-1)
            results['entity_scores'] = self.sigmoid(scores)
        return results, pooled


class T5MultiTaskModel(nn.Module):
    """
    T5 backbone supporting:
      - s-expression generation ("generation")
      - isomorphism classification ("iso")
      - entity ranking ("ranking")
    """
    def __init__(self, params, tokenizer: PreTrainedTokenizer,
                 modes=("generation", "iso", "ranking")):
        super().__init__()
        assert set(modes).issubset({"generation", "iso", "ranking"}), "Unsupported mode"
        self.modes = set(modes)
        self.tokenizer = tokenizer
        self.backbone = T5ForConditionalGeneration.from_pretrained(params.ptlm_model)
        self.backbone.resize_token_embeddings(len(tokenizer))
        self.hidden = self.backbone.config.d_model
        if "iso" in self.modes:
            self.classifier = nn.Linear(self.hidden, params.num_classes)
        if "ranking" in self.modes:
            self.entity_scorer = nn.Linear(self.hidden, 1)
            self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(params.dropout)

    def forward(self,
                input_ids: torch.LongTensor,
                attention_mask: torch.LongTensor,
                labels: torch.LongTensor = None,
                entity_spans: list = None):
        gen_outputs = None
        if "generation" in self.modes:
            gen_outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
        encoder_hidden = self.backbone.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        ).last_hidden_state  # (B, L, H)
        pooled = encoder_hidden.mean(dim=1)
        results = {'gen_outputs': gen_outputs}
        if "iso" in self.modes:
            pooled = self.dropout(pooled)
            iso_logits = self.classifier(pooled)
            results['iso_logits'] = iso_logits
        if "ranking" in self.modes:
            ent_embs = extract_entity_embeddings(encoder_hidden, entity_spans)
            padded = nn.utils.rnn.pad_sequence(ent_embs, batch_first=True)
            scores = self.entity_scorer(padded).squeeze(-1)
            results['entity_scores'] = self.sigmoid(scores)
        return results, pooled
    
    def generate(self, *args, **kwargs):
        return self.backbone.generate(*args, **kwargs)


class GPT2MultiTaskModel(nn.Module):
    """
    GPT2 backbone supporting:
      - s-expression generation ("generation")
      - isomorphism classification ("iso") via final token hidden
    """
    def __init__(self, params, tokenizer: PreTrainedTokenizer,
                 modes=("generation", "iso")):
        super().__init__()
        assert set(modes).issubset({"generation", "iso"}), "Unsupported mode"
        self.modes = set(modes)
        self.tokenizer = tokenizer
        self.backbone = GPT2LMHeadModel.from_pretrained(params.ptlm_model)
        self.backbone.resize_token_embeddings(len(tokenizer))
        self.hidden = self.backbone.config.hidden_size
        if "iso" in self.modes:
            self.classifier = nn.Linear(self.hidden, params.num_classes)
        self.dropout = nn.Dropout(params.dropout)

    def forward(self,
                input_ids: torch.LongTensor,
                attention_mask: torch.LongTensor,
                labels: torch.LongTensor = None):
        gen_outputs = None
        if "generation" in self.modes:
            gen_outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
        hidden = self.backbone.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        ).last_hidden_state  # (B, L, H)
        pooled = hidden[torch.arange(hidden.size(0)), attention_mask.sum(-1)-1]
        results = {'gen_outputs': gen_outputs}
        if "iso" in self.modes:
            results['iso_logits'] = self.classifier(self.dropout(pooled))
        return results, pooled
    
    def generate(self, *args, **kwargs):
        return self.backbone.generate(*args, **kwargs)