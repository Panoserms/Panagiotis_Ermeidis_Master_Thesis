"""Fine-tune BERT-Large on the subtitle pairs and save the embeddings of all three splits (needs a GPU).

python gpu/finetune_bert.py --pairs data/pairs --out work
Reads train.csv, validation.csv and test.csv with sub1_text, sub2_text and silence_label (made by the notebook, section 1).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          get_linear_schedule_with_warmup)

MODEL, MAX_LEN, SEED = "bert-large-uncased", 128, 42
BATCH_TRAIN, BATCH_EVAL, GRAD_ACCUM, EPOCHS = 8, 32, 4, 3
LEARNING_RATE, WEIGHT_DECAY, WARMUP_STEPS, LABEL_SMOOTHING = 1e-5, 0.02, 500, 0.1
LAST_LAYERS = 4


def make_loader(table, tokenizer, batch, shuffle=False):
    rows = [{**tokenizer(a, b, truncation=True, max_length=MAX_LEN), "labels": int(y)}
            for a, b, y in zip(table["sub1_text"], table["sub2_text"], table["silence_label"])]
    return DataLoader(rows, batch_size=batch, shuffle=shuffle, collate_fn=DataCollatorWithPadding(tokenizer))


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    gold, predicted = [], []
    for batch in loader:
        gold += batch.pop("labels").tolist()
        predicted += model(**batch.to(device)).logits.argmax(-1).tolist()
    return accuracy_score(gold, predicted)


@torch.no_grad()
def embed(model, loader, device):
    """[CLS] of the last four layers, one after the other: 4 x 1,024 = 4,096 numbers per pair."""
    model.eval()
    out = []
    for batch in loader:
        batch.pop("labels")
        layers = model(**batch.to(device), output_hidden_states=True).hidden_states[-LAST_LAYERS:]
        out.append(torch.cat([layer[:, 0] for layer in layers], dim=-1).cpu().numpy())
    return np.vstack(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda")

    tables = {s: pd.read_csv(args.pairs / f"{s}.csv", keep_default_na=False) for s in ("train", "validation", "test")}
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(device)
    train = make_loader(tables["train"], tokenizer, BATCH_TRAIN, shuffle=True)
    validation = make_loader(tables["validation"], tokenizer, BATCH_EVAL)

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    schedule = get_linear_schedule_with_warmup(optimizer, WARMUP_STEPS, EPOCHS * (len(train) // GRAD_ACCUM))
    scaler = torch.amp.GradScaler("cuda")
    best = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for step, batch in enumerate(train, start=1):
            labels = batch.pop("labels").to(device)
            with torch.amp.autocast("cuda"):
                loss = loss_fn(model(**batch.to(device)).logits, labels) / GRAD_ACCUM
            scaler.scale(loss).backward()
            if step % GRAD_ACCUM == 0 or step == len(train):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                if step % GRAD_ACCUM == 0:
                    schedule.step()
                optimizer.zero_grad(set_to_none=True)
        score = accuracy(model, validation, device)
        print(f"epoch {epoch}: validation accuracy {score:.4f}")
        if score > best:
            best = score
            model.save_pretrained(args.out / "bert")

    model = AutoModelForSequenceClassification.from_pretrained(args.out / "bert").to(device)
    arrays = {}
    for name, short in (("train", "train"), ("validation", "val"), ("test", "test")):
        arrays[f"{short}_embeddings"] = embed(model, make_loader(tables[name], tokenizer, BATCH_EVAL), device)
        arrays[f"y_{short}"] = tables[name]["silence_label"].to_numpy()
    np.savez_compressed(args.out / "bert_embeddings.npz", **arrays)


if __name__ == "__main__":
    main()
