import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from src.config import (
    PROCESSED_DATA_PATH,
    TEXT_COLUMN,
    ORGANS,
    RANDOM_STATE
)


MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
NUM_LABELS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}


def train_clinicalbert_for_organ(organ):
    print(f"\n🚀 Training ClinicalBERT for: {organ}")

    df = pd.read_csv(PROCESSED_DATA_PATH)
    texts = df[TEXT_COLUMN].tolist()
    labels = df[organ].tolist()

    # ✅ FIX: Save label mapping so decision_layer uses the SAME order
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # LabelEncoder sorts alphabetically → missing=0, present=1, removed=2
    label_mapping = {i: cls for i, cls in enumerate(label_encoder.classes_)}
    print(f"  Label mapping for {organ}: {label_mapping}")
    # Expected: {0: 'missing', 1: 'present', 2: 'removed'}

    # ✅ Save the label mapping alongside the model so it can be loaded later
    output_dir = f"models/bert_models/{organ}"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "label_mapping.json"), "w") as f:
        json.dump(label_mapping, f)
    print(f"  Label mapping saved to {output_dir}/label_mapping.json")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels_encoded,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels_encoded
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )
    model.to(DEVICE)

    train_dataset = ClinicalDataset(X_train, y_train, tokenizer)
    test_dataset  = ClinicalDataset(X_test,  y_test,  tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=20,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    print("\n📊 Final Evaluation:")
    metrics = trainer.evaluate()
    print(metrics)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Finished training for {organ}")
    print(f"   Label order: {list(label_encoder.classes_)}")


if __name__ == "__main__":
    for organ in ORGANS:
        train_clinicalbert_for_organ(organ)