from typing import List

from gector.gec_model import GecBERTModel


def load_model():
    return GecBERTModel(vocab_path='model/vocabulary',
                        model_paths=['model/roberta-large_1_best_10k.th', 'model/xlnet-large_1_best_10k.th'],
                        max_len=50, min_len=3,
                        iterations=5,
                        min_error_probability=0.0,
                        lowercase_tokens=False,
                        model_name='roberta',
                        special_tokens_fix=1,
                        log=False,
                        confidence=0,
                        del_confidence=0,
                        is_ensemble=True,
                        weigths=None)


def process_batches(texts: List[str], model: GecBERTModel, batch_size=128):
    probabilities = []
    predicted_labels = []
    error_probabilities = []
    batch = []
    for i, sent in enumerate(texts):
        batch.append(sent.split())
        if len(batch) == batch_size or i == len(texts) - 1:
            batch_probabilities, batch_labels, batch_error_probs = model.get_tags(batch)
            probabilities.extend(batch_probabilities)
            predicted_labels.extend(batch_labels)
            error_probabilities.extend(batch_error_probs)
            batch = []
    return probabilities, predicted_labels, error_probabilities
