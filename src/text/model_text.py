import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

class TextESAClassifier:
    def __init__(self, num_labels=5):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=num_labels
        )

    def tokenize(self, texts):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

    def train(self, texts, labels, epochs=1):
        self.model.train()
        inputs = self.tokenize(texts)
        labels = torch.tensor(labels)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-5)

        for epoch in range(epochs):
            optimizer.zero_grad()

            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels
            )

            loss = outputs.loss
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
            loss.backward()
            optimizer.step()

    def evaluate(self, texts, labels):
        self.model.eval()
        inputs = self.tokenize(texts)
        labels = torch.tensor(labels)

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

        preds = torch.argmax(outputs.logits, dim=1)
        print(classification_report(labels, preds, zero_division=0))
        print("Macro F1:", f1_score(labels, preds, average="macro", zero_division=0))
