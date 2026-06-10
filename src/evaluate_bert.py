import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config import PROCESSED_DATA_PATH, TEXT_COLUMN, ORGANS
from src.utils.metrics import plot_confusion_matrix  


class ClinicalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def evaluate_model_for_organ(organ):
    print(f"\n🔹 Evaluating {organ} ...")

    # Load data
    df = pd.read_csv(PROCESSED_DATA_PATH)
    texts = df[TEXT_COLUMN].tolist()
    labels = df[organ].tolist()

    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    # Split data (we only need test here)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        texts, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
    )

    # Load tokenizer & model
    MODEL_DIR = f"models/bert_models/{organ}"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    test_dataset = ClinicalDataset(X_test, y_test, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=8)

    all_preds = []
    all_labels = []

    # Predict
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels_batch = batch["labels"]

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.tolist())
            all_labels.extend(labels_batch.tolist())

    # Accuracy + Classification Report
    acc = accuracy_score(all_labels, all_preds)
    print(f"✅ Accuracy for {organ}: {acc*100:.2f}%")
    print(classification_report(all_labels, all_preds, target_names=le.classes_))

    # Confusion Matrix
    plot_confusion_matrix(all_labels, all_preds, le.classes_, organ)


if __name__ == "__main__":
    for organ in ORGANS:
        evaluate_model_for_organ(organ)
