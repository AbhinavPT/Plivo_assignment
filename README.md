# PII NER Assignment

This project implements a DistilBERT-based token classification model for detecting PII entities in STT-style transcripts using BIO tagging and character-based span extraction. The model outputs entity spans with start/end offsets and a `pii` flag.

---

## 📊 Final Metrics

### Entity-wise Performance
| Entity        | Precision | Recall | F1 Score |
|---------------|-----------|--------|----------|
| CITY          | 1.000     | 1.000  | 1.000    |
| DATE          | 1.000     | 1.000  | 1.000    |
| EMAIL         | 1.000     | 1.000  | 1.000    |
| LOCATION      | 1.000     | 1.000  | 1.000    |
| PERSON_NAME   | 1.000     | 1.000  | 1.000    |
| PHONE         | 1.000     | 1.000  | 1.000    |

**Macro F1 Score:** 1.000  
**PII-only F1:** 1.000  
**Non-PII F1:** 1.000

### Latency (CPU, batch size = 1)
| Metric | Value |
|--------|--------|
| Mean Latency | **19.14 ms** |
| p95 Latency | **18.48 ms** |

---

## ⚙ Model Details

| Component | Value |
|-----------|--------|
| Model | distilbert-base-uncased |
| Tokenizer | distilbert-base-uncased |
| Approach | Fully learned, no regex or heuristic rules |
| Entities | CREDIT_CARD, PHONE, EMAIL, PERSON_NAME, DATE, CITY, LOCATION |

### Hyperparameters
| Parameter | Value |
|-----------|--------|
| Learning Rate | 5e-5 |
| Batch Size | 8 |
| Epochs | 3 |
| Max Length | 256 |

---

## 🚀 Commands to Run

### Training
python -m src.train --model_name distilbert-base-uncased --train data/train.jsonl --dev data/dev.jsonl --out_dir out/base --batch_size 8 --epochs 3 --lr 5e-5

### Prediction
python -m src.predict --model_dir out/base --input data/dev.jsonl --output out/dev_pred.json

### Evaluation
python -m src.eval_span_f1 --gold data/dev.jsonl --pred out/dev_pred.json

### Latency Measurement
python -m src.measure_latency --model_dir out/base --input data/dev.jsonl

---

## 📂 Output File
Location: `out/dev_pred.json`


---

## 📁 Repository Structure

src/  
data/  
out/  
requirements.txt  
README.md
