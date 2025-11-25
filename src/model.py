# src/model.py

from transformers import AutoConfig, AutoModelForTokenClassification


def create_model(model_name: str, num_labels: int):
    """
    Creates a token classification model (e.g. DistilBERT) with the
    correct number of labels for our BIO scheme.
    """
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)
    return model
