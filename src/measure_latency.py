# src/measure_latency.py

import argparse
import json
import time

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out", help="Directory with trained model + tokenizer")
    ap.add_argument("--input", default="data/dev.jsonl", help="JSONL file to measure latency on")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    texts = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj["text"])

    latencies_ms = []

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(args.device) for k, v in enc.items()}

            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(**enc)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            lat_ms = (t1 - t0) * 1000.0
            latencies_ms.append(lat_ms)

    if not latencies_ms:
        print("No samples found in input file.")
        return

    latencies_ms.sort()
    n = len(latencies_ms)
    p95 = latencies_ms[int(0.95 * n) - 1]

    mean_lat = sum(latencies_ms) / n

    print(f"Number of samples: {n}")
    print(f"Mean latency (ms):  {mean_lat:.2f}")
    print(f"p95 latency (ms):   {p95:.2f}")


if __name__ == "__main__":
    main()
