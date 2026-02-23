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
OUTPUT_DIR = os.path.join(BASE_DIR, "submissions/urd_final")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "saved_models/Urdu")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(BASE_DIR, "merged/urd.csv")
TEST_FILE  = os.path.join(BASE_DIR, "test/urd.csv")

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

def train_and_predict(model_name, alias, df_train, df_val, df_test, epochs):
    print(f"\n--- Training {alias} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # MuRIL stability settings (as in your file’s code style)
    if alias.lower() == "muril":
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    train_ds = Dataset.from_pandas(df_train).map(tokenize_fn, batched=True)
    val_ds   = Dataset.from_pandas(df_val).map(tokenize_fn, batched=True)
    test_ds  = Dataset.from_pandas(df_test).map(tokenize_fn, batched=True)

    keep_cols = ["input_ids", "attention_mask", "label"]
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])
    val_ds   = val_ds.remove_columns([c for c in val_ds.column_names if c not in keep_cols])
    test_ds  = test_ds.remove_columns([c for c in test_ds.column_names if c not in ["input_ids", "attention_mask"]])

    args = TrainingArguments(
        output_dir=f"./temp_urd_{alias}",
        eval_strategy="epoch",
        save_strategy="no",
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        num_train_epochs=epochs,
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

    trainer.train()

    preds = trainer.predict(test_ds)
    probs = torch.nn.functional.softmax(torch.tensor(preds.predictions), dim=-1)[:, 1].cpu().numpy()

    del model, trainer, tokenizer, train_ds, val_ds, test_ds
    torch.cuda.empty_cache()
    gc.collect()

    return probs

def main():
    print("Starting Urdu Grand Finale: XLM-R + MuRIL (50/50), threshold 0.5 ...")
    df_train, df_val, df_test = load_data()

    probs_xlmr  = train_and_predict("xlm-roberta-base", "xlmr",  df_train, df_val, df_test, epochs=4)
    probs_muril = train_and_predict("google/muril-base-cased", "muril", df_train, df_val, df_test, epochs=5)

    print("\nMixing: 50% XLM-R + 50% MuRIL")
    final_probs = (probs_xlmr + probs_muril) / 2.0

    # Standard threshold
    final_preds = (final_probs > 0.5).astype(int)

    pred_path = os.path.join(OUTPUT_DIR, "pred_urd.csv")
    pd.DataFrame({"id": df_test["id"], "polarization": final_preds}).to_csv(pred_path, index=False)

    prob_path = os.path.join(OUTPUT_DIR, "probs_urd.csv")
    pd.DataFrame({"id": df_test["id"], "prob1": final_probs}).to_csv(prob_path, index=False)

    print(f"Saved: {pred_path}")
    print(f"Saved: {prob_path}")

if __name__ == "__main__":
    main()
