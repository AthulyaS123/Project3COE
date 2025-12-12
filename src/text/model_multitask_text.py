# src/text/model_multitask_text.py

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class MultitaskTextESAFCAModel(nn.Module):
    """
    One shared BERT encoder with two classification heads:
      - ESA: 2 classes (0/1)
      - FCA: 3 classes (0/1/2)
    """

    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        # Classification heads
        self.dropout = nn.Dropout(0.1)
        self.esa_classifier = nn.Linear(hidden_size, 2)  # 2 classes
        self.fca_classifier = nn.Linear(hidden_size, 3)  # 3 classes

        self.esa_loss_fn = nn.CrossEntropyLoss()
        self.fca_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids,
        attention_mask,
        esa_labels=None,
        fca_labels=None,
    ):
        # BERT encoder
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output  # [batch_size, hidden_size]
        pooled = self.dropout(pooled)

        # Task-specific logits
        esa_logits = self.esa_classifier(pooled)   # [batch, 2]
        fca_logits = self.fca_classifier(pooled)   # [batch, 3]

        loss = None
        loss_esa = None
        loss_fca = None

        if esa_labels is not None and fca_labels is not None:
            loss_esa = self.esa_loss_fn(esa_logits, esa_labels)
            loss_fca = self.fca_loss_fn(fca_logits, fca_labels)
            # Simple equal-weighted multitask loss
            loss = loss_esa + loss_fca

        return {
            "loss": loss,
            "loss_esa": loss_esa,
            "loss_fca": loss_fca,
            "esa_logits": esa_logits,
            "fca_logits": fca_logits,
        }


def load_tokenizer(model_name: str = "bert-base-uncased"):
    return BertTokenizer.from_pretrained(model_name)