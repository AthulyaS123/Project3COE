import os
import time
from collections import Counter
import pickle  

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizerFast,
    DistilBertModel,
)
from sklearn.metrics import classification_report


# --------------------------
# Dataset
# --------------------------
class TextMultitaskDataset(Dataset):
    def __init__(self, texts, esa_labels, fca_labels, tokenizer, max_length=256):
        self.texts = texts
        self.esa_labels = esa_labels
        self.fca_labels = fca_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        esa = int(self.esa_labels[idx])
        fca = int(self.fca_labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["esa_label"] = torch.tensor(esa, dtype=torch.long)
        item["fca_label"] = torch.tensor(fca, dtype=torch.long)
        return item


# --------------------------
# Multitask DistilBERT Model
# --------------------------
class DistilBERTMultitask(nn.Module):
    def __init__(self, num_esa_classes=2, num_fca_classes=3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        hidden = self.bert.config.dim  # 768

        # Two classification heads
        self.esa_classifier = nn.Linear(hidden, num_esa_classes)
        self.fca_classifier = nn.Linear(hidden, num_fca_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token representation

        esa_logits = self.esa_classifier(pooled)
        fca_logits = self.fca_classifier(pooled)

        return esa_logits, fca_logits


# --------------------------
# Training Loop (with class weights)
# --------------------------
def train_epoch(model, dataloader, optimizer, device, esa_weights, fca_weights):
    model.train()
    total_loss = 0.0

    criterion_esa = nn.CrossEntropyLoss(weight=esa_weights)
    criterion_fca = nn.CrossEntropyLoss(weight=fca_weights)

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        esa_labels = batch["esa_label"].to(device)
        fca_labels = batch["fca_label"].to(device)

        optimizer.zero_grad()
        esa_logits, fca_logits = model(input_ids, attention_mask)

        esa_loss = criterion_esa(esa_logits, esa_labels)
        fca_loss = criterion_fca(fca_logits, fca_labels)

        loss = esa_loss + fca_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 5 == 0:
            print(
                f"[TRAIN] Batch {batch_idx}/{len(dataloader)} "
                f"Loss: {loss.item():.4f} (ESA {esa_loss.item():.4f}, FCA {fca_loss.item():.4f})"
            )

    return total_loss / max(len(dataloader), 1)


# --------------------------
# Evaluation
# --------------------------
def evaluate(model, dataloader, device):
    model.eval()
    all_esa_preds, all_esa_true = [], []
    all_fca_preds, all_fca_true = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            esa_labels = batch["esa_label"]
            fca_labels = batch["fca_label"]

            esa_logits, fca_logits = model(input_ids, attention_mask)

            esa_pred = torch.argmax(esa_logits, dim=1).cpu()
            fca_pred = torch.argmax(fca_logits, dim=1).cpu()

            all_esa_preds.extend(esa_pred.tolist())
            all_esa_true.extend(esa_labels.tolist())
            all_fca_preds.extend(fca_pred.tolist())
            all_fca_true.extend(fca_labels.tolist())

    print("\n===== ESA Classification Report =====")
    print(classification_report(all_esa_true, all_esa_preds, zero_division=0))

    print("\n===== FCA Classification Report =====")
    print(classification_report(all_fca_true, all_fca_preds, zero_division=0))

    # Return predictions for visualization
    return all_esa_true, all_esa_preds, all_fca_true, all_fca_preds


# --------------------------
# Main
# --------------------------
# --------------------------
# Main
# --------------------------
def main():
    print("[INFO] Loading dataset...")
    df = pd.read_csv("../data/processed/text_dataset.csv")

    # --------------------------
    # OVERSAMPLING RARE CLASSES
    # --------------------------
    print("[INFO] Oversampling rare classes...")

    esa_minority = df[df["ESA_Class"] == 0]
    if len(esa_minority) > 0:
        df = pd.concat([df, esa_minority] * 5, ignore_index=True)

    fca_minority = df[df["FCA_Class"] == 0]
    if len(fca_minority) > 0:
        df = pd.concat([df, fca_minority] * 5, ignore_index=True)

    print("[INFO] New ESA counts:", df["ESA_Class"].value_counts().to_dict())
    print("[INFO] New FCA counts:", df["FCA_Class"].value_counts().to_dict())

    # Extract columns
    texts = df["text"].tolist()
    esa = df["ESA_Class"].tolist()
    fca = df["FCA_Class"].tolist()

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    dataset = TextMultitaskDataset(texts, esa, fca, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    model = DistilBERTMultitask().to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # --------------------------
    # CLASS WEIGHTS
    # --------------------------
    esa_counts = Counter(esa)
    fca_counts = Counter(fca)

    esa_total = sum(esa_counts.values())
    esa_weights_list = [esa_total / esa_counts[i] for i in sorted(esa_counts.keys())]
    esa_weights = torch.tensor(esa_weights_list, dtype=torch.float).to(device)

    fca_total = sum(fca_counts.values())
    fca_weights_list = [fca_total / fca_counts[i] for i in sorted(fca_counts.keys())]
    fca_weights = torch.tensor(fca_weights_list, dtype=torch.float).to(device)

    print(f"[INFO] ESA class counts: {esa_counts}")
    print(f"[INFO] ESA class weights: {esa_weights_list}")
    print(f"[INFO] FCA class counts: {fca_counts}")
    print(f"[INFO] FCA class weights: {fca_weights_list}")

    # --------------------------
    # Training
    # --------------------------
    print("\n[INFO] Starting training...")
    start = time.time()
    num_epochs = 5

    for epoch in range(num_epochs):
        print(f"\n===== Epoch {epoch + 1}/{num_epochs} =====")
        loss = train_epoch(model, dataloader, optimizer, device, esa_weights, fca_weights)
        print(f"[INFO] Epoch {epoch + 1} Loss: {loss:.4f}")

    end = time.time()
    print(f"\n[INFO] Training time: {end - start:.2f} seconds")

    # --------------------------
    # Evaluation
    # --------------------------
    print("\n[INFO] Evaluating model...")
    all_esa_true, all_esa_preds, all_fca_true, all_fca_preds = evaluate(
        model, dataloader, device
    )

    # --------------------------
    # Save model + tokenizer + eval results
    # --------------------------
    save_path = "../models/distilbert_multitask/"
    os.makedirs(save_path, exist_ok=True)

    # 1) Save eval results so the notebook can load them
    results_path = os.path.join(save_path, "eval_results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(
            {
                "esa_true": all_esa_true,
                "esa_pred": all_esa_preds,
                "fca_true": all_fca_true,
                "fca_pred": all_fca_preds,
            },
            f,
        )
    print(f"[OK] Saved evaluation results to: {results_path}")

    # 2) Save model + tokenizer as before
    print(f"[INFO] Saving model to: {save_path}")
    torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
    tokenizer.save_pretrained(save_path)
    print("[OK] Model + tokenizer saved!")


if __name__ == "__main__":
    main()
