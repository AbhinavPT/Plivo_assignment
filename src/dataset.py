# src/dataset.py

import json
from typing import List, Dict, Any
from torch.utils.data import Dataset

from src.labels import LABELS, LABEL2ID  # use the same labels everywhere


class PIIDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 256, is_train: bool = True):
        self.items = []
        self.tokenizer = tokenizer
        self.label_list = LABELS
        self.label2id = LABEL2ID
        self.max_length = max_length
        self.is_train = is_train

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj["text"]
                entities = obj.get("entities", [])

                # 1) character-level BIO tags
                char_tags = ["O"] * len(text)
                for ent in entities:
                    s, e_idx, lab = ent["start"], ent["end"], ent["label"]
                    # basic span sanity checks
                    if s < 0 or e_idx > len(text) or s >= e_idx:
                        continue
                    char_tags[s] = f"B-{lab}"
                    for i in range(s + 1, e_idx):
                        char_tags[i] = f"I-{lab}"

                # 2) tokenize with offsets
                enc = tokenizer(
                    text,
                    return_offsets_mapping=True,
                    truncation=True,
                    max_length=self.max_length,
                    add_special_tokens=True,
                )
                offsets = enc["offset_mapping"]
                input_ids = enc["input_ids"]
                attention_mask = enc["attention_mask"]

                # 3) map char-level tags → token-level BIO tags
                bio_tags = []
                for (start, end) in offsets:
                    if start == end:
                        # special tokens (CLS/SEP, maybe padding)
                        bio_tags.append("O")
                    else:
                        if start < len(char_tags):
                            bio_tags.append(char_tags[start])
                        else:
                            bio_tags.append("O")

                # safety: if lengths mismatch for any reason, fall back to all "O"
                if len(bio_tags) != len(input_ids):
                    bio_tags = ["O"] * len(input_ids)

                # 4) convert BIO tags to label ids
                label_ids = [self.label2id.get(t, self.label2id["O"]) for t in bio_tags]

                self.items.append(
                    {
                        "id": obj["id"],
                        "text": text,
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": label_ids,
                        "offset_mapping": offsets,
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


def collate_batch(batch, pad_token_id: int, label_pad_id: int = -100):
    """
    Pads a batch to the same length.
    Returns plain Python lists; train.py will turn them into tensors.
    """
    input_ids_list = [x["input_ids"] for x in batch]
    attention_list = [x["attention_mask"] for x in batch]
    labels_list = [x["labels"] for x in batch]

    max_len = max(len(ids) for ids in input_ids_list)

    def pad(seq, pad_value, max_len):
        return seq + [pad_value] * (max_len - len(seq))

    input_ids = [pad(ids, pad_token_id, max_len) for ids in input_ids_list]
    attention_mask = [pad(am, 0, max_len) for am in attention_list]
    labels = [pad(lab, label_pad_id, max_len) for lab in labels_list]

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "ids": [x["id"] for x in batch],
        "texts": [x["text"] for x in batch],
        "offset_mapping": [x["offset_mapping"] for x in batch],
    }
    return out
