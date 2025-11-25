# src/predict.py
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import os

from src.labels import ID2LABEL, label_is_pii  # <-- use src package path


def bio_to_spans(text, offsets, label_ids):
    spans = []
    current_label = None
    current_start = None
    current_end = None

    for (start, end), lid in zip(offsets, label_ids):
        # offsets may be ints or floats; ensure ints
        start = int(start)
        end = int(end)

        # skip special tokens with (0,0)
        if start == 0 and end == 0:
            continue

        label = ID2LABEL.get(int(lid), "O")
        if label == "O":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
            continue

        # label is like "B-CREDIT_CARD" or "I-EMAIL"
        parts = label.split("-", 1)
        if len(parts) != 2:
            # unexpected format; treat as O
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
            continue
        prefix, ent_type = parts

        if prefix == "B":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
            current_label = ent_type
            current_start = start
            current_end = end
        elif prefix == "I":
            if current_label == ent_type and current_label is not None:
                current_end = end
            else:
                # treat I without correct previous as a B
                if current_label is not None:
                    spans.append((current_start, current_end, current_label))
                current_label = ent_type
                current_start = start
                current_end = end
        else:
            # unknown prefix: flush current
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None

    if current_label is not None:
        spans.append((current_start, current_end, current_label))

    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # prefer tokenizer saved in model_dir (if you saved tokenizer there)
    tokenizer_source = args.model_dir if args.model_name is None else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)

    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    results = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            # offset_mapping is a tensor when return_tensors="pt"; convert to list
            offsets = enc["offset_mapping"][0].cpu().tolist()
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[0]                    # (seq_len, num_labels)
                pred_ids = logits.argmax(dim=-1).cpu().tolist()

            spans = bio_to_spans(text, offsets, pred_ids)

            ents = []
            for s, e, lab in spans:
                # clamp bounds and convert to ints
                s = max(0, int(s))
                e = min(len(text), int(e))
                ents.append(
                    {
                        "start": s,
                        "end": e,
                        "label": lab,
                        "pii": bool(label_is_pii(lab)),
                    }
                )
            results[uid] = ents

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")


if __name__ == "__main__":
    main()
