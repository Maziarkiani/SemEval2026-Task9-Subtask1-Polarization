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
OUTPUT_DIR = os.path.join(BASE_DIR, "submissions/eng_final")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "saved_models/English")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(BASE_DIR, "merged/eng.csv")
TEST_FILE = os.path.join(BASE_DIR, "test/eng.csv")

def load_data():
    df_full = pd.read_csv(TRAIN_FILE)
    df_test = pd.read_csv(TEST_FILE)

    if 'polarization' in df_full.columns: 
        df_full.rename(columns={'polarization': 'label'}, inplace=True)
    elif 'labels' in df_full.columns: 
        df_full.rename(columns={'labels': 'label'}, inplace=True)
        
    df_full = df_full.dropna(subset=['label'])
    df_full['label'] = df_full['label'].astype(int)

    df_train, df_val = train_test_split(df_full, test_size=0.05, random_state=42)
    return df_train, df_val, df_test

def train_and_predict(model_name, alias, df_train, df_val, df_test, use_fast_tok=True):
    print(f"Loading {alias} ({model_name})...")
    
    if alias == "bertweet":
        tokenizer = AutoTokenizer.from_pretrained(model_name, normalization=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast_tok)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2,
    )

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    train_ds = Dataset.from_pandas(df_train).map(tokenize_fn, batched=True)
    val_ds = Dataset.from_pandas(df_val).map(tokenize_fn, batched=True)
    test_ds = Dataset.from_pandas(df_test).map(tokenize_fn, batched=True)

    keep_cols = ['input_ids', 'attention_mask', 'label']
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])
    val_ds = val_ds.remove_columns([c for c in val_ds.column_names if c not in keep_cols])
    test_ds = test_ds.remove_columns([c for c in test_ds.column_names if c not in ['input_ids', 'attention_mask']])

    if alias == "deberta":
        args = TrainingArguments(
            output_dir=f"./temp_{alias}",
            eval_strategy="epoch",
            save_strategy="no",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            learning_rate=5e-6,
            num_train_epochs=4,
            weight_decay=0.01,
            fp16=True,
            logging_steps=50,
            report_to="none"
        )
    else:
        args = TrainingArguments(
            output_dir=f"./temp_{alias}",
            eval_strategy="epoch",
            save_strategy="no",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            learning_rate=7e-6,
            num_train_epochs=4,
            weight_decay=0.01,
            fp16=True,
            logging_steps=50,
            report_to="none"
        )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)
        return {"acc": (preds == labels).mean()}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )

    print(f"Training {alias}...")
    trainer.train()
    
    print(f"Saving final model for {alias}...")
    trainer.save_model(os.path.join(MODEL_SAVE_DIR, alias))
    tokenizer.save_pretrained(os.path.join(MODEL_SAVE_DIR, alias))

    print(f"Getting predictions for {alias}...")
    preds = trainer.predict(test_ds)
    probs = torch.nn.functional.softmax(torch.tensor(preds.predictions), dim=-1)[:, 1].numpy()

    del model, trainer, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return probs

def main():
    print("Starting English Ensemble Processing...")
    df_train, df_val, df_test = load_data()

    probs_deberta = train_and_predict("microsoft/deberta-v3-large", "deberta", df_train, df_val, df_test, use_fast_tok=False)
    probs_bertweet = train_and_predict("vinai/bertweet-large", "bertweet", df_train, df_val, df_test, use_fast_tok=True)

    print("Ensembling predictions...")
    final_probs = (probs_deberta * 0.65) + (probs_bertweet * 0.35)
    
    labels_final = (final_probs > 0.45).astype(int)

    out_df = pd.DataFrame({'id': df_test['id'], 'polarization': labels_final})
    pred_path = os.path.join(OUTPUT_DIR, "pred_eng.csv")
    out_df.to_csv(pred_path, index=False)
    
    print(f"Predictions successfully saved to {pred_path}")

if __name__ == "__main__":
    main()
