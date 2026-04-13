import pandas as pd
import numpy as np
import os
import torch
import gc
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.model_selection import train_test_split

BASE_DIR = "./SemEval_Data/test_phase"
OUTPUT_DIR = os.path.join(BASE_DIR, "submissions/ita_final")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "saved_models/Italian")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(BASE_DIR, "merged/ita.csv")
TEST_FILE = os.path.join(BASE_DIR, "test/ita.csv")


def load_data():
    df_full = pd.read_csv(TRAIN_FILE)
    df_test = pd.read_csv(TEST_FILE)

    if "polarization" in df_full.columns:
        df_full.rename(columns={"polarization": "label"}, inplace=True)
    elif "labels" in df_full.columns:
        df_full.rename(columns={"labels": "label"}, inplace=True)

    df_full = df_full.dropna(subset=["label"])
    df_full["label"] = df_full["label"].astype(int)

    df_train, df_val = train_test_split(df_full, test_size=0.05, random_state=42)
    return df_train, df_val, df_test


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return {"acc": (preds == labels).mean()}


def main():
    print("Starting Italian Specialist Processing (GilBERTo)...")
    df_train, df_val, df_test = load_data()

    model_name = "idb-ita/gilberto-uncased-from-camembert"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    train_ds = Dataset.from_pandas(df_train).map(tokenize_fn, batched=True)
    val_ds = Dataset.from_pandas(df_val).map(tokenize_fn, batched=True)
    test_ds = Dataset.from_pandas(df_test).map(tokenize_fn, batched=True)

    keep_cols = ["input_ids", "attention_mask", "label"]
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])
    val_ds = val_ds.remove_columns([c for c in val_ds.column_names if c not in keep_cols])
    test_ds = test_ds.remove_columns([c for c in test_ds.column_names if c not in ["input_ids", "attention_mask"]])

    args = TrainingArguments(
        output_dir="./temp_ita_gilberto",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        num_train_epochs=5,
        weight_decay=0.01,
        fp16=True,
        logging_steps=50,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )

    print("Training model...")
    trainer.train()

    print("Saving final Italian model...")
    trainer.save_model(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)

    print("Generating predictions...")
    preds = trainer.predict(test_ds)

    probs = torch.nn.functional.softmax(torch.tensor(preds.predictions), dim=-1)[:, 1].numpy()
    labels_final = np.argmax(preds.predictions, axis=1)

    pred_path = os.path.join(OUTPUT_DIR, "pred_ita.csv")
    pd.DataFrame({"id": df_test["id"], "polarization": labels_final}).to_csv(pred_path, index=False)

    probs_path = os.path.join(OUTPUT_DIR, "probs_ita.csv")
    pd.DataFrame({"id": df_test["id"], "prob1": probs}).to_csv(probs_path, index=False)

    print(f"Predictions saved to {pred_path}")
    print(f"Probabilities saved to {probs_path}")

    del model, trainer, tokenizer
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
