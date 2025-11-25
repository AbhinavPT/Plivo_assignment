# src/labels.py

LABELS = [
    "O",
    "B-CREDIT_CARD", "I-CREDIT_CARD",
    "B-PHONE", "I-PHONE",
    "B-EMAIL", "I-EMAIL",
    "B-PERSON_NAME", "I-PERSON_NAME",
    "B-DATE", "I-DATE",
    "B-CITY", "I-CITY",
    "B-LOCATION", "I-LOCATION",
]

# Entity type -> pii boolean
PII_MAP = {
    "CREDIT_CARD": True,
    "PHONE": True,
    "EMAIL": True,
    "PERSON_NAME": True,
    "DATE": True,
    "CITY": False,
    "LOCATION": False
}

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

def label_is_pii(label: str) -> bool:
    """
    label comes as entity type, not BIO-tag.
    Example: CREDIT_CARD (not B-CREDIT_CARD)
    """
    return PII_MAP.get(label, False)
