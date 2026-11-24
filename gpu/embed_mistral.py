"""Mistral-7B embeddings of the subtitle pairs: frozen model, mean of the last hidden layer (needs a GPU).

python gpu/embed_mistral.py --pairs data/pairs --out work
The mean runs over every position of the batch, padding included, as in the thesis.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

MODEL, MAX_LEN, BATCH = "mistralai/Mistral-7B-v0.3", 64, 32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto").eval()

    arrays = {}
    for name, short in (("train", "train"), ("validation", "val"), ("test", "test")):
        table = pd.read_csv(args.pairs / f"{name}.csv", keep_default_na=False)
        texts = [f"{a} </s> {b}".strip().replace("\n", " ") for a, b in zip(table["sub1_text"], table["sub2_text"])]
        out = []
        for i in range(0, len(texts), BATCH):
            inputs = tokenizer(texts[i:i + BATCH], return_tensors="pt", truncation=True, max_length=MAX_LEN, padding=True)
            with torch.no_grad():
                out.append(model(**inputs.to(model.device)).last_hidden_state.mean(dim=1).float().cpu().numpy())
        arrays[f"{short}_embeddings"] = np.vstack(out)
        arrays[f"y_{short}"] = table["silence_label"].to_numpy()
    np.savez_compressed(args.out / "mistral_embeddings.npz", **arrays)


if __name__ == "__main__":
    main()
